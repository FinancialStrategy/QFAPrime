from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from core.config import ProfessionalConfig, RunDiagnostics
from core.universes import flatten_universe_dict, get_universe_definition


@dataclass
class DataLoadResult:
    asset_prices: pd.DataFrame
    asset_returns: pd.DataFrame
    benchmark_prices: pd.Series
    benchmark_returns: pd.Series
    asset_metadata: pd.DataFrame


class ProfessionalDataManager:
    def __init__(self, config: ProfessionalConfig, diagnostics: Optional[RunDiagnostics] = None):
        self.config = config
        self.diagnostics = diagnostics or RunDiagnostics()

        self.asset_prices: pd.DataFrame = pd.DataFrame()
        self.asset_returns: pd.DataFrame = pd.DataFrame()
        self.benchmark_prices: pd.Series = pd.Series(dtype=float)
        self.benchmark_returns: pd.Series = pd.Series(dtype=float)
        self.asset_metadata: pd.DataFrame = pd.DataFrame()

    def _build_universe_metadata(self) -> pd.DataFrame:
        universe = get_universe_definition(self.config.selected_universe)
        flat = flatten_universe_dict(universe)

        rows: List[Dict[str, str]] = []
        for _, meta in flat.items():
            ticker = str(meta.get("ticker", "")).strip()
            if not ticker:
                continue

            rows.append(
                {
                    "ticker": ticker,
                    "name": str(meta.get("name", ticker)),
                    "bucket": str(meta.get("bucket", "Unknown")),
                    "asset_class": str(meta.get("asset_class", meta.get("bucket", "Unknown"))),
                    "region_type": str(meta.get("region_type", "Unknown")),
                    "category": str(meta.get("category", "Unknown")),
                }
            )

        metadata = pd.DataFrame(rows).drop_duplicates(subset=["ticker"]).reset_index(drop=True)

        if metadata.empty:
            metadata = pd.DataFrame(
                {
                    "ticker": self.config.default_symbols,
                    "name": self.config.default_symbols,
                    "bucket": ["Equities"] * len(self.config.default_symbols),
                    "asset_class": ["Equities"] * len(self.config.default_symbols),
                    "region_type": ["Unknown"] * len(self.config.default_symbols),
                    "category": ["Default"] * len(self.config.default_symbols),
                }
            )

        return metadata

    def _download_close_prices(self, tickers: List[str], start_date: str) -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame()

        try:
            raw = yf.download(
                tickers=tickers,
                start=start_date,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
            )
        except Exception as exc:
            self.diagnostics.add_error(f"Yahoo Finance download failed: {exc}")
            return pd.DataFrame()

        if raw is None or raw.empty:
            self.diagnostics.add_warning("Downloaded price data is empty.")
            return pd.DataFrame()

        # Single ticker case
        if len(tickers) == 1:
            if "Close" in raw.columns:
                prices = raw[["Close"]].copy()
                prices.columns = tickers
            else:
                prices = raw.copy()
                prices.columns = tickers[: len(prices.columns)]
            return self._clean_prices(prices)

        # Multi ticker case
        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" in raw.columns.get_level_values(-1):
                prices = raw.xs("Close", axis=1, level=-1, drop_level=True).copy()
            else:
                level0 = list(dict.fromkeys(raw.columns.get_level_values(0)))
                extracted = {}
                for t in level0:
                    sub = raw[t]
                    if "Close" in sub.columns:
                        extracted[t] = sub["Close"]
                prices = pd.DataFrame(extracted)
        else:
            prices = raw.copy()

        return self._clean_prices(prices)

    def _clean_prices(self, prices: pd.DataFrame) -> pd.DataFrame:
        if prices is None or prices.empty:
            return pd.DataFrame()

        clean = prices.copy()
        clean = clean.loc[:, ~clean.columns.duplicated()]
        clean = clean.sort_index()
        clean = clean.replace([np.inf, -np.inf], np.nan)
        clean = clean.ffill(limit=5)
        clean = clean.dropna(axis=0, how="all")
        clean = clean.dropna(axis=1, how="all")

        # Keep only columns with enough observations
        min_obs = max(20, int(self.config.min_observations))
        valid_cols = [c for c in clean.columns if clean[c].dropna().shape[0] >= min_obs]
        clean = clean[valid_cols] if valid_cols else pd.DataFrame(index=clean.index)

        return clean

    def _compute_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        if prices is None or prices.empty:
            return pd.DataFrame()

        if self.config.use_log_returns:
            returns = np.log(prices / prices.shift(1))
        else:
            returns = prices.pct_change()

        returns = returns.replace([np.inf, -np.inf], np.nan)
        returns = returns.dropna(how="all")
        return returns

    def _fallback_universe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ticker": self.config.default_symbols,
                "name": self.config.default_symbols,
                "bucket": ["Equities"] * len(self.config.default_symbols),
                "asset_class": ["Equities"] * len(self.config.default_symbols),
                "region_type": ["Unknown"] * len(self.config.default_symbols),
                "category": ["Fallback"] * len(self.config.default_symbols),
            }
        )

    def load(self) -> DataLoadResult:
        metadata = self._build_universe_metadata()
        if metadata.empty:
            metadata = self._fallback_universe()

        tickers = metadata["ticker"].dropna().astype(str).unique().tolist()

        asset_prices = self._download_close_prices(
            tickers=tickers,
            start_date=self.config.default_start_date,
        )

        if asset_prices.empty:
            self.diagnostics.add_warning(
                "Primary universe download returned empty data. Falling back to default symbols."
            )
            metadata = self._fallback_universe()
            tickers = metadata["ticker"].dropna().astype(str).unique().tolist()
            asset_prices = self._download_close_prices(
                tickers=tickers,
                start_date=self.config.default_start_date,
            )

        if asset_prices.empty:
            raise ValueError("Asset price download failed. No usable Yahoo Finance data was returned.")

        asset_prices = asset_prices.loc[:, [c for c in asset_prices.columns if c in tickers]]

        benchmark_df = self._download_close_prices(
            tickers=[self.config.benchmark_symbol],
            start_date=self.config.default_start_date,
        )
        if benchmark_df.empty or self.config.benchmark_symbol not in benchmark_df.columns:
            raise ValueError(f"Benchmark download failed for {self.config.benchmark_symbol}.")

        benchmark_prices = benchmark_df[self.config.benchmark_symbol].copy()

        common_index = asset_prices.index.intersection(benchmark_prices.index)
        asset_prices = asset_prices.loc[common_index].copy()
        benchmark_prices = benchmark_prices.loc[common_index].copy()

        asset_prices = asset_prices.ffill(limit=5).dropna(axis=0, how="all")
        benchmark_prices = benchmark_prices.ffill(limit=5).dropna()

        common_index = asset_prices.index.intersection(benchmark_prices.index)
        asset_prices = asset_prices.loc[common_index].copy()
        benchmark_prices = benchmark_prices.loc[common_index].copy()

        asset_returns = self._compute_returns(asset_prices)
        benchmark_returns = self._compute_returns(benchmark_prices.to_frame(name=self.config.benchmark_symbol))
        benchmark_returns = benchmark_returns[self.config.benchmark_symbol]

        common_return_index = asset_returns.index.intersection(benchmark_returns.index)
        asset_returns = asset_returns.loc[common_return_index].copy()
        benchmark_returns = benchmark_returns.loc[common_return_index].copy()

        # Remove columns with too many missing values after alignment
        asset_returns = asset_returns.dropna(axis=1, how="all")
        asset_returns = asset_returns.dropna(axis=0, how="all")

        valid_cols = asset_returns.columns.tolist()
        metadata = metadata[metadata["ticker"].isin(valid_cols)].copy().reset_index(drop=True)
        asset_prices = asset_prices[valid_cols].copy()

        if asset_returns.empty or benchmark_returns.empty:
            raise ValueError("Return calculation failed after alignment.")

        if asset_returns.shape[1] < 2:
            self.diagnostics.add_warning("Less than 2 assets available after cleaning.")

        self.asset_prices = asset_prices
        self.asset_returns = asset_returns
        self.benchmark_prices = benchmark_prices
        self.benchmark_returns = benchmark_returns
        self.asset_metadata = metadata

        self.diagnostics.add_info("num_assets", int(asset_returns.shape[1]))
        self.diagnostics.add_info("num_observations", int(asset_returns.shape[0]))
        self.diagnostics.add_info("benchmark_symbol", self.config.benchmark_symbol)

        return DataLoadResult(
            asset_prices=self.asset_prices,
            asset_returns=self.asset_returns,
            benchmark_prices=self.benchmark_prices,
            benchmark_returns=self.benchmark_returns,
            asset_metadata=self.asset_metadata,
        )

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
    def __init__(
        self,
        config: ProfessionalConfig,
        diagnostics: Optional[RunDiagnostics] = None,
    ):
        self.config = config
        self.diagnostics = diagnostics or RunDiagnostics()

        self.asset_prices: pd.DataFrame = pd.DataFrame()
        self.asset_returns: pd.DataFrame = pd.DataFrame()
        self.benchmark_prices: pd.Series = pd.Series(dtype=float)
        self.benchmark_returns: pd.Series = pd.Series(dtype=float)
        self.asset_metadata: pd.DataFrame = pd.DataFrame()

    # =========================
    # UNIVERSE HANDLING
    # =========================
    def _build_metadata(self) -> pd.DataFrame:
        universe = get_universe_definition(self.config.selected_universe)
        flat = flatten_universe_dict(universe)

        rows: List[Dict] = []
        for ticker, meta in flat.items():
            rows.append(meta)

        df = pd.DataFrame(rows)

        if df.empty:
            self.diagnostics.add_warning("Universe metadata is empty. Falling back.")
            df = pd.DataFrame({
                "ticker": self.config.default_symbols,
                "name": self.config.default_symbols,
                "bucket": ["Fallback"] * len(self.config.default_symbols),
                "asset_class": ["Equities"] * len(self.config.default_symbols),
                "region_type": ["Unknown"] * len(self.config.default_symbols),
                "category": ["Fallback"] * len(self.config.default_symbols),
            })

        return df.drop_duplicates(subset=["ticker"]).reset_index(drop=True)

    # =========================
    # DOWNLOAD
    # =========================
    def _download_prices(self, tickers: List[str]) -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame()

        try:
            data = yf.download(
                tickers=tickers,
                start=self.config.default_start_date,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
            )
        except Exception as e:
            self.diagnostics.add_error(f"Yahoo download failed: {e}")
            return pd.DataFrame()

        if data is None or data.empty:
            return pd.DataFrame()

        # SINGLE TICKER
        if len(tickers) == 1:
            if "Close" in data.columns:
                df = data[["Close"]].copy()
                df.columns = tickers
            else:
                df = data.copy()
            return df

        # MULTI TICKER
        if isinstance(data.columns, pd.MultiIndex):
            if "Close" in data.columns.get_level_values(-1):
                df = data.xs("Close", axis=1, level=-1)
            else:
                extracted = {}
                for t in tickers:
                    if t in data.columns.get_level_values(0):
                        sub = data[t]
                        if "Close" in sub.columns:
                            extracted[t] = sub["Close"]
                df = pd.DataFrame(extracted)
        else:
            df = data.copy()

        return df

    # =========================
    # CLEANING
    # =========================
    def _clean_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()
        df = df.loc[:, ~df.columns.duplicated()]
        df = df.sort_index()

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill(limit=5)
        df = df.dropna(axis=0, how="all")

        # min observation filter
        min_obs = max(30, self.config.min_observations)
        valid_cols = [c for c in df.columns if df[c].dropna().shape[0] >= min_obs]

        return df[valid_cols]

    def _compute_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        if prices.empty:
            return prices

        if self.config.use_log_returns:
            ret = np.log(prices / prices.shift(1))
        else:
            ret = prices.pct_change()

        ret = ret.replace([np.inf, -np.inf], np.nan)
        ret = ret.dropna(how="all")

        return ret

    # =========================
    # MAIN LOAD
    # =========================
    def load(self) -> DataLoadResult:
        metadata = self._build_metadata()

        tickers = metadata["ticker"].tolist()

        prices = self._download_prices(tickers)
        prices = self._clean_prices(prices)

        if prices.empty:
            raise ValueError("No asset price data downloaded.")

        # Benchmark
        bench_df = self._download_prices([self.config.benchmark_symbol])
        if bench_df.empty:
            raise ValueError("Benchmark download failed.")

        benchmark_prices = bench_df.iloc[:, 0]

        # ALIGN INDEX
        common_index = prices.index.intersection(benchmark_prices.index)
        prices = prices.loc[common_index]
        benchmark_prices = benchmark_prices.loc[common_index]

        # RETURNS
        asset_returns = self._compute_returns(prices)
        benchmark_returns = self._compute_returns(
            benchmark_prices.to_frame(name=self.config.benchmark_symbol)
        )
        benchmark_returns = benchmark_returns.iloc[:, 0]

        # ALIGN RETURNS
        common_idx = asset_returns.index.intersection(benchmark_returns.index)
        asset_returns = asset_returns.loc[common_idx]
        benchmark_returns = benchmark_returns.loc[common_idx]

        # FINAL CLEAN
        asset_returns = asset_returns.dropna(axis=1, how="all")
        asset_returns = asset_returns.dropna(axis=0, how="all")

        valid_cols = asset_returns.columns.tolist()
        prices = prices[valid_cols]
        metadata = metadata[metadata["ticker"].isin(valid_cols)].reset_index(drop=True)

        # SAVE
        self.asset_prices = prices
        self.asset_returns = asset_returns
        self.benchmark_prices = benchmark_prices
        self.benchmark_returns = benchmark_returns
        self.asset_metadata = metadata

        self.diagnostics.add_info("num_assets", len(valid_cols))
        self.diagnostics.add_info("num_observations", len(asset_returns))

        return DataLoadResult(
            asset_prices=prices,
            asset_returns=asset_returns,
            benchmark_prices=benchmark_prices,
            benchmark_returns=benchmark_returns,
            asset_metadata=metadata,
        )

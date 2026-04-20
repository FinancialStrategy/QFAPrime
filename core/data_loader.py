from __future__ import annotations

import time
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

    def _build_metadata(self) -> pd.DataFrame:
        universe = get_universe_definition(self.config.selected_universe)
        flat = flatten_universe_dict(universe)

        rows: List[Dict] = list(flat.values())
        df = pd.DataFrame(rows)

        if df.empty:
            self.diagnostics.add_warning("Universe metadata is empty. Falling back to default symbols.")
            df = pd.DataFrame({
                "ticker": self.config.default_symbols,
                "name": self.config.default_symbols,
                "bucket": ["Fallback"] * len(self.config.default_symbols),
                "asset_class": ["Equities"] * len(self.config.default_symbols),
                "region_type": ["Unknown"] * len(self.config.default_symbols),
                "category": ["Fallback"] * len(self.config.default_symbols),
            })

        return df.drop_duplicates(subset=["ticker"]).reset_index(drop=True)

    def _download_single_ticker(self, ticker: str, max_retries: int = 3) -> pd.Series:
        last_error = None

        for attempt in range(max_retries):
            try:
                data = yf.download(
                    tickers=ticker,
                    start=self.config.default_start_date,
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )

                if data is None or data.empty:
                    last_error = f"No data returned for {ticker}"
                    time.sleep(1.0 + attempt)
                    continue

                if "Close" in data.columns:
                    s = data["Close"].copy()
                else:
                    s = data.iloc[:, 0].copy()

                s = pd.to_numeric(s, errors="coerce").dropna()
                s.name = ticker

                if s.empty:
                    last_error = f"Empty close series for {ticker}"
                    time.sleep(1.0 + attempt)
                    continue

                return s

            except Exception as exc:
                last_error = str(exc)
                time.sleep(1.5 + attempt)

        self.diagnostics.add_warning(f"Ticker download failed for {ticker}: {last_error}")
        return pd.Series(dtype=float, name=ticker)

    def _download_prices(self, tickers: List[str]) -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame()

        series_list: List[pd.Series] = []

        for ticker in tickers:
            s = self._download_single_ticker(ticker)
            if not s.empty:
                series_list.append(s)

        if not series_list:
            return pd.DataFrame()

        df = pd.concat(series_list, axis=1)
        return df

    def _clean_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        out = df.copy()
        out = out.loc[:, ~out.columns.duplicated()]
        out = out.sort_index()
        out = out.replace([np.inf, -np.inf], np.nan)
        out = out.ffill(limit=5)
        out = out.dropna(axis=0, how="all")

        min_obs = max(30, int(self.config.min_observations))
        valid_cols = [c for c in out.columns if out[c].dropna().shape[0] >= min_obs]

        if not valid_cols:
            return pd.DataFrame(index=out.index)

        return out[valid_cols]

    def _compute_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        if prices.empty:
            return pd.DataFrame()

        if self.config.use_log_returns:
            ret = np.log(prices / prices.shift(1))
        else:
            ret = prices.pct_change(fill_method=None)

        ret = ret.replace([np.inf, -np.inf], np.nan)
        ret = ret.dropna(how="all")
        return ret

    def _get_benchmark_candidates(self) -> List[str]:
        candidates = [self.config.benchmark_symbol]

        fallback_map = {
            "^GSPC": ["SPY", "VTI", "QQQ"],
            "SPY": ["^GSPC", "VTI", "QQQ"],
            "QQQ": ["^NDX", "SPY", "VTI"],
            "GLD": ["GC=F", "IAU"],
        }

        for item in fallback_map.get(self.config.benchmark_symbol, []):
            if item not in candidates:
                candidates.append(item)

        return candidates

    def _download_benchmark(self) -> pd.Series:
        for candidate in self._get_benchmark_candidates():
            s = self._download_single_ticker(candidate)
            if not s.empty:
                if candidate != self.config.benchmark_symbol:
                    self.diagnostics.add_warning(
                        f"Primary benchmark {self.config.benchmark_symbol} failed. Using fallback benchmark {candidate}."
                    )
                self.config.benchmark_symbol = candidate
                return s

        return pd.Series(dtype=float, name=self.config.benchmark_symbol)

    def load(self) -> DataLoadResult:
        metadata = self._build_metadata()
        tickers = metadata["ticker"].astype(str).tolist()

        prices = self._download_prices(tickers)
        prices = self._clean_prices(prices)

        if prices.empty:
            raise ValueError("No asset price data downloaded.")

        benchmark_prices = self._download_benchmark()
        if benchmark_prices.empty:
            raise ValueError(f"Benchmark download failed for {self.config.benchmark_symbol} and fallback candidates.")

        common_index = prices.index.intersection(benchmark_prices.index)
        prices = prices.loc[common_index].copy()
        benchmark_prices = benchmark_prices.loc[common_index].copy()

        prices = self._clean_prices(prices)
        benchmark_prices = pd.to_numeric(benchmark_prices, errors="coerce").dropna()

        common_index = prices.index.intersection(benchmark_prices.index)
        prices = prices.loc[common_index].copy()
        benchmark_prices = benchmark_prices.loc[common_index].copy()

        asset_returns = self._compute_returns(prices)
        benchmark_returns_df = self._compute_returns(
            benchmark_prices.to_frame(name=self.config.benchmark_symbol)
        )

        if benchmark_returns_df.empty:
            raise ValueError("Benchmark returns are empty after cleaning/alignment.")

        benchmark_returns = benchmark_returns_df.iloc[:, 0]

        common_idx = asset_returns.index.intersection(benchmark_returns.index)
        asset_returns = asset_returns.loc[common_idx].copy()
        benchmark_returns = benchmark_returns.loc[common_idx].copy()

        asset_returns = asset_returns.dropna(axis=1, how="all")
        asset_returns = asset_returns.dropna(axis=0, how="all")

        valid_cols = asset_returns.columns.tolist()
        prices = prices[valid_cols].copy()
        metadata = metadata[metadata["ticker"].isin(valid_cols)].reset_index(drop=True)

        if len(valid_cols) < 2:
            raise ValueError("Not enough assets available after download/cleaning. At least 2 assets are required.")

        if asset_returns.empty:
            raise ValueError("Asset returns are empty after cleaning/alignment.")

        self.asset_prices = prices
        self.asset_returns = asset_returns
        self.benchmark_prices = benchmark_prices
        self.benchmark_returns = benchmark_returns
        self.asset_metadata = metadata

        dropped = sorted(set(tickers) - set(valid_cols))
        if dropped:
            self.diagnostics.add_warning(
                f"Dropped tickers after cleaning/download: {', '.join(dropped)}"
            )

        self.diagnostics.add_info("num_assets", len(valid_cols))
        self.diagnostics.add_info("num_observations", len(asset_returns))
        self.diagnostics.add_info("benchmark_symbol", self.config.benchmark_symbol)

        return DataLoadResult(
            asset_prices=self.asset_prices,
            asset_returns=self.asset_returns,
            benchmark_prices=self.benchmark_prices,
            benchmark_returns=self.benchmark_returns,
            asset_metadata=self.asset_metadata,
        )

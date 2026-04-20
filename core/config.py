# `core/config.py`

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd


@dataclass
class ProfessionalConfig:
    initial_capital: float = 10_000_000.0
    annual_trading_days: int = 252
    benchmark: str = "SPY"
    as_of_date: str = field(
        default_factory=lambda: (pd.Timestamp.today().normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    )
    lookback_years: int = 6
    risk_free_rate: float = 0.0425

    min_weight: float = 0.00
    max_weight: float = 0.20
    max_category_weight: float = 0.35
    allow_short: bool = False

    covariance_method: str = "ledoit_wolf"  # ledoit_wolf, sample, shrinkage
    expected_return_method: str = "ema_historical"  # ema_historical, historical_mean, capm
    confidence_levels: Tuple[float, ...] = (0.90, 0.95, 0.99)

    turnover_penalty: float = 0.005
    transaction_cost_bps: float = 10.0
    tracking_error_target: float = 0.06

    data_timeout_seconds: int = 30
    report_file: str = field(
        default_factory=lambda: f"qfa_professional_quant_platform_{datetime.today().strftime('%Y%m%d_%H%M')}.html"
    )

    selected_universe: str = "Institutional Multi-Asset"
    selected_region: str = "All"

    @property
    def start_date(self) -> str:
        return (pd.Timestamp(self.as_of_date) - pd.DateOffset(years=self.lookback_years)).strftime("%Y-%m-%d")

    @property
    def end_date(self) -> str:
        return self.as_of_date


@dataclass
class RunDiagnostics:
    dropped_assets: Dict[str, str] = field(default_factory=dict)
    info: List[str] = field(default_factory=list)
    benchmark_used: str | None = None
    covariance_repaired: bool = False
    covariance_method_used: str | None = None
    expected_return_method_used: str | None = None
    strategy_diagnostics: Dict[str, Dict] = field(default_factory=dict)

    def add_info(self, message: str) -> None:
        self.info.append(message)
```

---

# `core/universes.py`

```python
from __future__ import annotations

from typing import Dict, List


INSTITUTIONAL_MULTI_ASSET_UNIVERSE: Dict[str, Dict[str, str]] = {
    "US_Equities": {
        "SPDR S&P 500 ETF": "SPY",
        "Invesco QQQ Trust": "QQQ",
        "iShares Russell 2000 ETF": "IWM",
        "Vanguard Total Stock Market ETF": "VTI",
        "SPDR Dow Jones Industrial Average ETF": "DIA",
    },
    "International_Equities": {
        "Vanguard FTSE Developed Markets ETF": "VEA",
        "Vanguard FTSE Emerging Markets ETF": "VWO",
        "iShares MSCI Japan ETF": "EWJ",
        "iShares MSCI Eurozone ETF": "EZU",
    },
    "Fixed_Income": {
        "Vanguard Total Bond Market ETF": "BND",
        "iShares 20+ Year Treasury ETF": "TLT",
        "iShares 7-10 Year Treasury ETF": "IEF",
        "iShares iBoxx $ Investment Grade Corporate Bond ETF": "LQD",
        "iShares National Muni Bond ETF": "MUB",
        "iShares iBoxx $ High Yield Corporate Bond ETF": "HYG",
    },
    "Real_Assets": {
        "SPDR Gold Shares": "GLD",
        "iShares Silver Trust": "SLV",
        "Vanguard Real Estate ETF": "VNQ",
        "iShares S&P GSCI Commodity-Indexed Trust": "GSG",
        "Invesco DB Base Metals Fund": "DBB",
    },
    "Sectors": {
        "Technology Select Sector SPDR Fund": "XLK",
        "Financial Select Sector SPDR Fund": "XLF",
        "Health Care Select Sector SPDR Fund": "XLV",
        "Energy Select Sector SPDR Fund": "XLE",
        "Industrial Select Sector SPDR Fund": "XLI",
        "Consumer Staples Select Sector SPDR Fund": "XLP",
        "Consumer Discretionary Select Sector SPDR Fund": "XLY",
        "Utilities Select Sector SPDR Fund": "XLU",
    },
}


GLOBAL_MAJOR_INDICES_30: Dict[str, Dict[str, str]] = {
    "North_America": {
        "S&P 500": "^GSPC",
        "Nasdaq 100": "^NDX",
        "Dow Jones Industrial Average": "^DJI",
        "Russell 2000": "^RUT",
        "S&P/TSX Composite": "^GSPTSE",
        "Mexico IPC": "^MXX",
    },
    "Europe": {
        "BIST 100": "XU100.IS",
        "Euro Stoxx 50": "^STOXX50E",
        "FTSE 100": "^FTSE",
        "DAX": "^GDAXI",
        "CAC 40": "^FCHI",
        "AEX": "^AEX",
        "SMI": "^SSMI",
        "IBEX 35": "^IBEX",
        "FTSE MIB": "FTSEMIB.MI",
    },
    "Asia_Pacific": {
        "Nikkei 225": "^N225",
        "TOPIX": "^TOPX",
        "Hang Seng": "^HSI",
        "Shanghai Composite": "000001.SS",
        "CSI 300": "000300.SS",
        "Shenzhen Component": "399001.SZ",
        "KOSPI": "^KS11",
        "ASX 200": "^AXJO",
        "Nifty 50": "^NSEI",
        "Straits Times": "^STI",
    },
    "Emerging_Markets": {
        "Bovespa": "^BVSP",
        "MSCI Emerging Markets Proxy": "EEM",
        "MSCI Frontier Markets Proxy": "FM",
        "MSCI Turkey Proxy": "TUR",
        "South Africa Proxy": "EZA",
    },
}


UNIVERSE_REGISTRY: Dict[str, Dict[str, Dict[str, str]]] = {
    "Institutional Multi-Asset": INSTITUTIONAL_MULTI_ASSET_UNIVERSE,
    "Global Major Stock Indices": GLOBAL_MAJOR_INDICES_30,
}


def flatten_universe_dict(universe_dict: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for bucket, assets in universe_dict.items():
        for long_name, ticker in assets.items():
            out[ticker] = {
                "name": long_name,
                "bucket": bucket,
                "ticker": ticker,
            }
    return out


def get_universe_definition(universe_name: str) -> Dict[str, Dict[str, str]]:
    if universe_name not in UNIVERSE_REGISTRY:
        raise ValueError(f"Unknown universe: {universe_name}")
    return UNIVERSE_REGISTRY[universe_name]


def get_available_regions(universe_name: str) -> List[str]:
    definition = get_universe_definition(universe_name)
    return ["All", *list(definition.keys())]


def get_universe_tickers(universe_name: str, selected_region: str = "All") -> List[str]:
    definition = get_universe_definition(universe_name)
    tickers: List[str] = []

    for bucket, assets in definition.items():
        if selected_region != "All" and bucket != selected_region:
            continue
        tickers.extend(list(assets.values()))

    return tickers


def get_universe_metadata(universe_name: str, selected_region: str = "All") -> List[dict]:
    definition = get_universe_definition(universe_name)
    rows: List[dict] = []

    for bucket, assets in definition.items():
        if selected_region != "All" and bucket != selected_region:
            continue
        for long_name, ticker in assets.items():
            rows.append(
                {
                    "region_type": bucket,
                    "ticker": ticker,
                    "identity": long_name,
                }
            )

    return rows
```

---

# `core/data_loader.py`

```python
from __future__ import annotations

from contextlib import contextmanager
import signal
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from core.config import ProfessionalConfig, RunDiagnostics
from core.universes import get_universe_metadata, get_universe_tickers


@contextmanager
def timeout_context(seconds: int):
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    original_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)


class ProfessionalDataManager:
    def __init__(self, config: ProfessionalConfig, diagnostics: RunDiagnostics):
        self.config = config
        self.diagnostics = diagnostics

        self.asset_prices = pd.DataFrame()
        self.asset_returns = pd.DataFrame()
        self.benchmark_prices = pd.Series(dtype=float)
        self.benchmark_returns = pd.Series(dtype=float)
        self.data_quality = pd.DataFrame()
        self.asset_metadata = pd.DataFrame()

    def _download_single_ticker(self, ticker: str) -> Tuple[pd.Series | None, dict]:
        try:
            with timeout_context(self.config.data_timeout_seconds):
                t = yf.Ticker(ticker)
                df = t.history(
                    start=self.config.start_date,
                    end=self.config.end_date,
                    auto_adjust=True,
                )
                try:
                    info = t.info or {}
                except Exception:
                    info = {}

            if df.empty or "Close" not in df.columns:
                return None, {
                    "ticker": ticker,
                    "name": ticker,
                    "exchange": "",
                    "currency": "",
                    "type": "",
                }

            close = df["Close"].copy()
            if getattr(close.index, "tz", None) is not None:
                close.index = close.index.tz_localize(None)

            return close, {
                "ticker": ticker,
                "name": info.get("longName") or info.get("shortName") or ticker,
                "exchange": info.get("exchange") or "",
                "currency": info.get("currency") or "",
                "type": info.get("quoteType") or "",
            }

        except Exception:
            return None, {
                "ticker": ticker,
                "name": ticker,
                "exchange": "",
                "currency": "",
                "type": "",
            }

    def load(self) -> None:
        tickers = get_universe_tickers(self.config.selected_universe, self.config.selected_region)
        metadata_map = {
            row["ticker"]: row for row in get_universe_metadata(self.config.selected_universe, self.config.selected_region)
        }

        price_map: Dict[str, pd.Series] = {}
        quality_rows = []
        metadata_rows = []

        for ticker in tickers:
            close, live_meta = self._download_single_ticker(ticker)
            base_meta = metadata_map.get(ticker, {"region_type": "Unknown", "identity": ticker})

            metadata_rows.append(
                {
                    "region_type": base_meta.get("region_type", "Unknown"),
                    "ticker": ticker,
                    "identity": base_meta.get("identity", ticker),
                    "name": live_meta.get("name", ticker),
                    "exchange": live_meta.get("exchange", ""),
                    "currency": live_meta.get("currency", ""),
                    "type": live_meta.get("type", ""),
                }
            )

            if close is None:
                self.diagnostics.dropped_assets[ticker] = "No close history"
                continue

            valid_ratio = float(close.notna().mean())
            if valid_ratio < 0.80:
                self.diagnostics.dropped_assets[ticker] = f"Insufficient history ({valid_ratio:.1%})"
                continue

            price_map[ticker] = close.rename(ticker)
            quality_rows.append(
                {
                    "ticker": ticker,
                    "valid_ratio": valid_ratio,
                    "observations": int(close.notna().sum()),
                    "first_valid": close.first_valid_index(),
                    "last_valid": close.last_valid_index(),
                }
            )

        if len(price_map) < 5:
            raise ValueError("Too few assets survived data quality filtering.")

        prices = pd.concat(price_map.values(), axis=1).sort_index().ffill(limit=3).dropna()
        returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

        self.asset_prices = prices.loc[returns.index]
        self.asset_returns = returns
        self.data_quality = pd.DataFrame(quality_rows).sort_values("ticker")
        self.asset_metadata = pd.DataFrame(metadata_rows).sort_values(["region_type", "ticker"])

        self._load_benchmark()
        self._align_all()

        self.diagnostics.add_info(
            f"Loaded {self.asset_returns.shape[1]} investable assets from {self.config.selected_universe}."
        )

    def _load_benchmark(self) -> None:
        try:
            benchmark_series = yf.Ticker(self.config.benchmark).history(
                start=self.config.start_date,
                end=self.config.end_date,
                auto_adjust=True,
            )["Close"]

            if getattr(benchmark_series.index, "tz", None) is not None:
                benchmark_series.index = benchmark_series.index.tz_localize(None)

            self.benchmark_prices = benchmark_series
            self.benchmark_returns = benchmark_series.pct_change().dropna()
            self.diagnostics.benchmark_used = self.config.benchmark

        except Exception:
            proxy = self.asset_returns.mean(axis=1)
            self.benchmark_returns = proxy
            self.benchmark_prices = (1 + proxy).cumprod()
            self.diagnostics.benchmark_used = "EqualWeightProxy"

    def _align_all(self) -> None:
        common = self.asset_returns.index.intersection(self.benchmark_returns.index)
        self.asset_returns = self.asset_returns.loc[common]
        self.asset_prices = self.asset_prices.loc[common]
        self.benchmark_returns = self.benchmark_returns.loc[common]
        if not self.benchmark_prices.empty:
            self.benchmark_prices = self.benchmark_prices.loc[self.benchmark_prices.index.intersection(common)]
```

---

# `app.py`

```python
from __future__ import annotations

import streamlit as st
import pandas as pd

from core.config import ProfessionalConfig, RunDiagnostics
from core.data_loader import ProfessionalDataManager
from core.universes import get_available_regions


st.set_page_config(
    page_title="QFA Professional Quant Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.title("QFA Professional Quant Platform")
st.caption("Institutional modular Streamlit foundation")


with st.sidebar:
    st.header("Platform Controls")

    selected_universe = st.selectbox(
        "Universe Type",
        ["Institutional Multi-Asset", "Global Major Stock Indices"],
        index=0,
    )

    available_regions = get_available_regions(selected_universe)
    selected_region = st.selectbox(
        "Region Filter",
        available_regions,
        index=0,
    )

    benchmark = st.text_input("Benchmark", value="SPY")
    lookback_years = st.slider("Lookback Years", min_value=3, max_value=15, value=6, step=1)
    risk_free_rate = st.slider("Risk-Free Rate", min_value=0.00, max_value=0.10, value=0.0425, step=0.0025)

    min_weight = st.slider("Min Weight", min_value=0.00, max_value=0.10, value=0.00, step=0.01)
    max_weight = st.slider("Max Weight", min_value=0.05, max_value=0.50, value=0.20, step=0.01)
    max_category_weight = st.slider("Max Category Weight", min_value=0.10, max_value=0.80, value=0.35, step=0.01)

    covariance_method = st.selectbox(
        "Covariance Method",
        ["ledoit_wolf", "sample", "shrinkage"],
        index=0,
    )

    expected_return_method = st.selectbox(
        "Expected Return Method",
        ["ema_historical", "historical_mean", "capm"],
        index=0,
    )

    turnover_penalty = st.slider("Turnover Penalty", min_value=0.000, max_value=0.050, value=0.005, step=0.001)
    transaction_cost_bps = st.slider("Transaction Cost (bps)", min_value=0.0, max_value=50.0, value=10.0, step=1.0)
    tracking_error_target = st.slider("Tracking Error Target", min_value=0.01, max_value=0.20, value=0.06, step=0.01)

    run_data_preview = st.button("Load Foundation Data", type="primary")


config = ProfessionalConfig(
    selected_universe=selected_universe,
    selected_region=selected_region,
    benchmark=benchmark,
    lookback_years=lookback_years,
    risk_free_rate=risk_free_rate,
    min_weight=min_weight,
    max_weight=max_weight,
    max_category_weight=max_category_weight,
    covariance_method=covariance_method,
    expected_return_method=expected_return_method,
    turnover_penalty=turnover_penalty,
    transaction_cost_bps=transaction_cost_bps,
    tracking_error_target=tracking_error_target,
)

diagnostics = RunDiagnostics()

if run_data_preview:
    try:
        data = ProfessionalDataManager(config, diagnostics)
        data.load()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Universe", config.selected_universe)
        col2.metric("Region", config.selected_region)
        col3.metric("Assets Loaded", len(data.asset_returns.columns))
        col4.metric("Benchmark", diagnostics.benchmark_used or config.benchmark)

        st.subheader("Investment Universe Metadata")
        st.dataframe(data.asset_metadata, use_container_width=True)

        st.subheader("Data Quality")
        st.dataframe(data.data_quality, use_container_width=True)

        st.subheader("Asset Price Sample")
        st.dataframe(data.asset_prices.tail(10), use_container_width=True)

        st.subheader("Asset Return Sample")
        st.dataframe(data.asset_returns.tail(10), use_container_width=True)

        if diagnostics.dropped_assets:
            st.subheader("Dropped Assets")
            dropped_df = pd.DataFrame(
                [{"ticker": k, "reason": v} for k, v in diagnostics.dropped_assets.items()]
            )
            st.dataframe(dropped_df, use_container_width=True)

    except Exception as exc:
        st.error(f"Foundation load failed: {exc}")
else:
    st.info(
        "This is the foundation stage of the Streamlit migration. Use the sidebar to choose the universe and load the data preview."
    )
```

---

# `requirements.txt`

```txt
streamlit
numpy
pandas
scipy
plotly
yfinance
PyPortfolioOpt
statsmodels
cvxpy
ecos
osqp
```

---

# `core/__init__.py`

```python
# core package initializer
```

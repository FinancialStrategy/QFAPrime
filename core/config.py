from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd

from core.universes import UNIVERSE_REGISTRY, get_universe_tickers


@dataclass
class ProfessionalConfig:
    initial_capital: float = 100_000.0
    annual_trading_days: int = 252

    benchmark_symbol: str = "^GSPC"
    default_start_date: str = "2019-01-01"

    benchmark: str = "^GSPC"
    as_of_date: str = field(default_factory=lambda: pd.Timestamp.today().strftime("%Y-%m-%d"))
    lookback_years: int = 6

    risk_free_rate: float = 0.03

    # More diversification-friendly caps
    min_weight: float = 0.00
    max_weight: float = 0.12
    max_category_weight: float = 0.40
    allow_short: bool = False

    covariance_method: str = "ledoit_wolf"
    expected_return_method: str = "historical_mean"
    correlation_method: str = "pearson"
    use_log_returns: bool = False

    min_observations: int = 60
    rolling_window: int = 63
    confidence_levels: Tuple[float, ...] = (0.90, 0.95, 0.99)

    turnover_penalty: float = 0.005
    transaction_cost_bps: float = 10.0
    tracking_error_target: float = 0.06

    data_timeout_seconds: int = 30
    enable_caching: bool = True
    cache_size: int = 128

    n_efficient_frontier_points: int = 50
    finquant_mc_trials: int = 6000

    selected_universe: str = "institutional_multi_asset"
    asset_universe: Dict[str, List[str]] = field(default_factory=dict)

    report_file: str = field(
        default_factory=lambda: f"qfa_quant_platform_{datetime.today().strftime('%Y%m%d_%H%M')}.html"
    )

    def __post_init__(self) -> None:
        self.benchmark = self.benchmark_symbol

        if not self.default_start_date:
            inferred_start = pd.Timestamp(self.as_of_date) - pd.DateOffset(years=self.lookback_years)
            self.default_start_date = inferred_start.strftime("%Y-%m-%d")

        tickers = []
        if self.selected_universe in UNIVERSE_REGISTRY:
            tickers = get_universe_tickers(self.selected_universe)

        if len(tickers) < 2:
            tickers = get_universe_tickers("institutional_multi_asset")
            self.selected_universe = "institutional_multi_asset"

        self.asset_universe = {"SelectedUniverse": tickers}

        self.initial_capital = float(self.initial_capital)
        self.risk_free_rate = float(self.risk_free_rate)
        self.min_weight = float(self.min_weight)
        self.max_weight = float(self.max_weight)
        self.max_category_weight = float(self.max_category_weight)
        self.turnover_penalty = float(self.turnover_penalty)
        self.transaction_cost_bps = float(self.transaction_cost_bps)
        self.tracking_error_target = float(self.tracking_error_target)
        self.min_observations = int(self.min_observations)
        self.rolling_window = int(self.rolling_window)
        self.cache_size = int(self.cache_size)
        self.finquant_mc_trials = int(self.finquant_mc_trials)

        if self.max_weight < self.min_weight:
            self.max_weight = self.min_weight

        if self.max_category_weight < self.max_weight:
            self.max_category_weight = self.max_weight

        if self.min_observations < 20:
            self.min_observations = 20

        if self.rolling_window < 20:
            self.rolling_window = 20

    @property
    def start_date(self) -> str:
        return self.default_start_date

    @property
    def end_date(self) -> str:
        return self.as_of_date

    @property
    def assets(self) -> List[str]:
        out: List[str] = []
        for _, names in self.asset_universe.items():
            out.extend(names)
        out = [str(x).strip().upper() for x in out if str(x).strip()]
        return list(dict.fromkeys(out))

    @property
    def asset_categories(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for category, names in self.asset_universe.items():
            for ticker in names:
                mapping[str(ticker).strip().upper()] = str(category)
        return mapping

    def to_dict(self) -> Dict[str, object]:
        return {
            "initial_capital": self.initial_capital,
            "annual_trading_days": self.annual_trading_days,
            "benchmark_symbol": self.benchmark_symbol,
            "benchmark": self.benchmark,
            "default_start_date": self.default_start_date,
            "as_of_date": self.as_of_date,
            "lookback_years": self.lookback_years,
            "risk_free_rate": self.risk_free_rate,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "max_category_weight": self.max_category_weight,
            "allow_short": self.allow_short,
            "covariance_method": self.covariance_method,
            "expected_return_method": self.expected_return_method,
            "correlation_method": self.correlation_method,
            "use_log_returns": self.use_log_returns,
            "min_observations": self.min_observations,
            "rolling_window": self.rolling_window,
            "confidence_levels": self.confidence_levels,
            "turnover_penalty": self.turnover_penalty,
            "transaction_cost_bps": self.transaction_cost_bps,
            "tracking_error_target": self.tracking_error_target,
            "data_timeout_seconds": self.data_timeout_seconds,
            "enable_caching": self.enable_caching,
            "cache_size": self.cache_size,
            "n_efficient_frontier_points": self.n_efficient_frontier_points,
            "finquant_mc_trials": self.finquant_mc_trials,
            "selected_universe": self.selected_universe,
            "asset_universe": self.asset_universe,
            "report_file": self.report_file,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "assets": self.assets,
            "asset_categories": self.asset_categories,
        }

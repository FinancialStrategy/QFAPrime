from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RunDiagnostics:
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)
    strategy_diagnostics: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def add_warning(self, message: str) -> None:
        if message:
            self.warnings.append(str(message))

    def add_error(self, message: str) -> None:
        if message:
            self.errors.append(str(message))

    def add_info(self, key: str, value: Any) -> None:
        self.info[str(key)] = value

    def add_strategy_info(self, strategy_name: str, key: str, value: Any) -> None:
        strategy_name = str(strategy_name)
        key = str(key)

        if strategy_name not in self.strategy_diagnostics:
            self.strategy_diagnostics[strategy_name] = {}

        self.strategy_diagnostics[strategy_name][key] = value

    def get_strategy_info(self, strategy_name: str) -> Dict[str, Any]:
        return self.strategy_diagnostics.get(str(strategy_name), {})

    def summary(self) -> Dict[str, Any]:
        return {
            "warnings": self.warnings,
            "errors": self.errors,
            "info": self.info,
            "strategy_diagnostics": self.strategy_diagnostics,
        }


@dataclass
class ProfessionalConfig:
    app_title: str = "Multi-Asset Portfolio Analytics"

    risk_free_rate: float = 0.03
    trading_days: int = 252
    annual_trading_days: int = 252

    confidence_level: float = 0.95
    confidence_levels: List[float] = field(default_factory=lambda: [0.90, 0.95, 0.99])

    benchmark_symbol: str = "^GSPC"
    cash_symbol: str = "BIL"
    default_start_date: str = "2019-01-01"

    initial_capital: float = 100000.0
    min_observations: int = 60
    rolling_window: int = 63
    ewma_lambda: float = 0.94
    use_log_returns: bool = False
    allow_short: bool = False
    max_assets: int = 30
    selected_universe: str = "institutional_multi_asset"

    expected_return_method: str = "historical_mean"
    covariance_method: str = "sample_cov"
    correlation_method: str = "pearson"

    output_dir: str = "outputs"
    report_dir: str = "reports"
    chart_height: int = 520
    chart_width: Optional[int] = None

    default_symbols: List[str] = field(default_factory=lambda: [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"
    ])

    benchmark_map: Dict[str, str] = field(default_factory=lambda: {
        "US Equities": "^GSPC",
        "Gold": "GLD",
        "Turkey": "XU100.IS",
        "World": "URTH",
    })

    scenario_families: List[str] = field(default_factory=lambda: [
        "crisis",
        "inflation",
        "banking_stress",
        "sharp_rally",
        "sharp_selloff",
    ])

    optimizer_settings: Dict[str, Any] = field(default_factory=dict)
    data_settings: Dict[str, Any] = field(default_factory=dict)
    stress_test_settings: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.trading_days <= 0:
            self.trading_days = 252

        self.annual_trading_days = int(self.trading_days)

        cleaned_levels = []
        for level in self.confidence_levels:
            try:
                val = float(level)
                if 0 < val < 1:
                    cleaned_levels.append(val)
            except (TypeError, ValueError):
                continue

        if 0 < float(self.confidence_level) < 1 and self.confidence_level not in cleaned_levels:
            cleaned_levels.append(float(self.confidence_level))

        if not cleaned_levels:
            cleaned_levels = [0.90, 0.95, 0.99]

        self.confidence_levels = sorted(set(cleaned_levels))
        self.validate()

    def validate(self) -> None:
        if not 0 <= self.risk_free_rate <= 1:
            raise ValueError("risk_free_rate must be between 0 and 1.")
        if self.trading_days <= 0:
            raise ValueError("trading_days must be positive.")
        if self.annual_trading_days <= 0:
            raise ValueError("annual_trading_days must be positive.")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive.")
        if self.min_observations < 20:
            raise ValueError("min_observations must be at least 20.")
        if self.rolling_window < 20:
            raise ValueError("rolling_window must be at least 20.")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1.")
        if not self.confidence_levels:
            raise ValueError("confidence_levels cannot be empty.")
        if any((cl <= 0 or cl >= 1) for cl in self.confidence_levels):
            raise ValueError("All confidence_levels must be between 0 and 1.")
        if not 0 < self.ewma_lambda < 1:
            raise ValueError("ewma_lambda must be between 0 and 1.")
        if self.max_assets <= 0:
            raise ValueError("max_assets must be positive.")

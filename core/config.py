from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ProfessionalConfig:
    app_title: str = "Professional Portfolio Analytics"
    risk_free_rate: float = 0.03
    trading_days: int = 252
    benchmark_symbol: str = "^GSPC"
    cash_symbol: str = "BIL"
    default_start_date: str = "2019-01-01"
    min_observations: int = 60
    rolling_window: int = 63
    ewma_lambda: float = 0.94
    confidence_level: float = 0.95
    use_log_returns: bool = False
    allow_short: bool = False
    max_assets: int = 30

    default_symbols: List[str] = field(default_factory=lambda: [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"
    ])

    benchmark_map: Dict[str, str] = field(default_factory=lambda: {
        "US Equities": "^GSPC",
        "Gold": "GLD",
        "Turkey": "XU100.IS",
        "World": "URTH"
    })

    scenario_families: List[str] = field(default_factory=lambda: [
        "crisis",
        "inflation",
        "banking_stress",
        "sharp_rally",
        "sharp_selloff"
    ])

    output_dir: str = "outputs"
    report_dir: str = "reports"
    chart_height: int = 550
    chart_width: Optional[int] = None

    def validate(self) -> None:
        if not 0 <= self.risk_free_rate <= 1:
            raise ValueError("risk_free_rate must be between 0 and 1.")
        if self.trading_days <= 0:
            raise ValueError("trading_days must be positive.")
        if self.min_observations < 20:
            raise ValueError("min_observations must be at least 20.")
        if self.rolling_window < 20:
            raise ValueError("rolling_window must be at least 20.")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1.")
        if not 0 < self.ewma_lambda < 1:
            raise ValueError("ewma_lambda must be between 0 and 1.")

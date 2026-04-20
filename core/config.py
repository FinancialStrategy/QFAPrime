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


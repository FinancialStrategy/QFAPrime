from __future__ import annotations

import numpy as np
import pandas as pd

from pypfopt import expected_returns

from core.config import ProfessionalConfig, RunDiagnostics


class ExpectedReturnBuilder:
    def __init__(self, config: ProfessionalConfig, diagnostics: RunDiagnostics):
        self.config = config
        self.diagnostics = diagnostics

    def build(
        self,
        returns: pd.DataFrame,
        benchmark_returns: pd.Series,
    ) -> pd.Series:
        self.diagnostics.expected_return_method_used = self.config.expected_return_method
        price_like = (1 + returns).cumprod()

        if self.config.expected_return_method == "historical_mean":
            mu = expected_returns.mean_historical_return(
                price_like,
                frequency=self.config.annual_trading_days,
            )

        elif self.config.expected_return_method == "capm":
            benchmark_prices = (1 + benchmark_returns).cumprod().to_frame("benchmark")
            mu = expected_returns.capm_return(
                price_like,
                market_prices=benchmark_prices,
                risk_free_rate=self.config.risk_free_rate,
                frequency=self.config.annual_trading_days,
            )

        else:
            mu = expected_returns.ema_historical_return(
                price_like,
                frequency=self.config.annual_trading_days,
            )

        mu = mu.replace([np.inf, -np.inf], np.nan).dropna()
        return mu

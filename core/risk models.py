from __future__ import annotations

import numpy as np
import pandas as pd

from pypfopt import risk_models as pypfopt_risk_models

from core.config import ProfessionalConfig, RunDiagnostics


class RiskModelBuilder:
    def __init__(self, config: ProfessionalConfig, diagnostics: RunDiagnostics):
        self.config = config
        self.diagnostics = diagnostics

    def build_covariance(self, prices: pd.DataFrame) -> pd.DataFrame:
        self.diagnostics.covariance_method_used = self.config.covariance_method

        if self.config.covariance_method == "sample":
            cov = pypfopt_risk_models.sample_cov(
                prices,
                frequency=self.config.annual_trading_days,
            )
        elif self.config.covariance_method == "shrinkage":
            cov = pypfopt_risk_models.CovarianceShrinkage(
                prices,
                frequency=self.config.annual_trading_days,
            ).shrunk_covariance(0.2)
        else:
            cov = pypfopt_risk_models.CovarianceShrinkage(
                prices,
                frequency=self.config.annual_trading_days,
            ).ledoit_wolf()

        return self._nearest_psd(cov)

    def correlation_matrix(self, returns: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
        corr = returns.corr(method=method).replace([np.inf, -np.inf], np.nan)
        corr = corr.fillna(0.0)
        np.fill_diagonal(corr.values, 1.0)
        return corr

    def ewma_volatility(
        self,
        returns: pd.DataFrame,
        lambda_: float = 0.94,
    ) -> pd.Series:
        if returns.empty:
            return pd.Series(dtype=float)

        variances = pd.Series(index=returns.columns, dtype=float)
        for col in returns.columns:
            x = returns[col].dropna().values
            if len(x) < 2:
                variances[col] = np.nan
                continue
            var = np.var(x)
            for r in x:
                var = lambda_ * var + (1 - lambda_) * (r ** 2)
            variances[col] = np.sqrt(var) * np.sqrt(self.config.annual_trading_days)
        return variances

    def _nearest_psd(self, cov: pd.DataFrame) -> pd.DataFrame:
        vals, vecs = np.linalg.eigh(cov.values)
        min_eval = vals.min()

        if min_eval < 0:
            vals = np.clip(vals, 1e-10, None)
            cov = pd.DataFrame(
                vecs @ np.diag(vals) @ vecs.T,
                index=cov.index,
                columns=cov.columns,
            )
            self.diagnostics.covariance_repaired = True

        return cov

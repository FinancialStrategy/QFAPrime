from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from core.config import ProfessionalConfig, RunDiagnostics


@dataclass
class StrategyResult:
    name: str
    method: str
    weights: Dict[str, float]
    diagnostics: Dict = field(default_factory=dict)


class ProfessionalOptimizer:
    def __init__(
        self,
        mu: pd.Series,
        cov: pd.DataFrame,
        returns: pd.DataFrame,
        benchmark_returns: pd.Series,
        config: ProfessionalConfig,
        diagnostics: Optional[RunDiagnostics] = None,
        asset_categories: Optional[Dict[str, str]] = None,
        asset_metadata: Optional[pd.DataFrame] = None,
        current_weights: Optional[Dict[str, float]] = None,
        bl_controls: Optional[Dict] = None,
    ):
        self.mu = mu
        self.cov = cov
        self.returns = returns
        self.benchmark_returns = benchmark_returns
        self.config = config
        self.diagnostics = diagnostics or RunDiagnostics()
        self.asset_categories = asset_categories or {}
        self.asset_metadata = asset_metadata
        self.current_weights = current_weights or {}
        self.bl_controls = bl_controls or {}

    def _normalize(self, w: pd.Series) -> Dict[str, float]:
        w = w.fillna(0.0)

        if not self.config.allow_short:
            w = w.clip(lower=0.0)

        total = float(w.sum())
        if np.isclose(total, 0.0):
            w[:] = 1.0 / len(w)
        else:
            w = w / total

        return w.to_dict()

    def _equal_weight(self) -> Dict[str, float]:
        cols = list(self.returns.columns)
        if not cols:
            return {}
        return pd.Series(1.0 / len(cols), index=cols, dtype=float).to_dict()

    def _min_variance_proxy(self) -> Dict[str, float]:
        diag = pd.Series(np.diag(self.cov.values), index=self.cov.index, dtype=float)
        inv_risk = 1.0 / diag.replace(0.0, np.nan)
        inv_risk = inv_risk.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        inv_risk = inv_risk.reindex(self.returns.columns).fillna(0.0)
        return self._normalize(inv_risk)

    def _risk_parity_proxy(self) -> Dict[str, float]:
        vol = self.returns.std() * np.sqrt(self.config.annual_trading_days)
        inv_vol = 1.0 / vol.replace(0.0, np.nan)
        inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return self._normalize(inv_vol)

    def _max_return_proxy(self) -> Dict[str, float]:
        scores = self.mu.reindex(self.returns.columns).fillna(self.mu.mean())
        scores = scores - scores.min() + 1e-8
        return self._normalize(scores)

    def _tracking_error_proxy(self) -> Dict[str, float]:
        # If benchmark ticker is in universe, follow it
        cols = list(self.returns.columns)
        if not cols:
            return {}

        if self.config.benchmark_symbol in cols:
            w = pd.Series(0.0, index=cols, dtype=float)
            w[self.config.benchmark_symbol] = 1.0
            return w.to_dict()

        return self._equal_weight()

    def _black_litterman_proxy(self) -> StrategyResult:
        prior = self.mu.reindex(self.returns.columns).fillna(self.mu.mean())
        posterior = prior.copy()

        weights = self._normalize(posterior - posterior.min() + 1e-8)

        return StrategyResult(
            name="Black-Litterman",
            method="black_litterman",
            weights=weights,
            diagnostics={
                "prior_returns": prior.to_dict(),
                "posterior_returns": posterior.to_dict(),
                "bl_weight_output": weights,
            },
        )

    def run_all(self) -> Dict[str, StrategyResult]:
        strategies: Dict[str, StrategyResult] = {}

        try:
            eq = self._equal_weight()
            if eq:
                strategies["Equal Weight"] = StrategyResult(
                    name="Equal Weight",
                    method="equal_weight",
                    weights=eq,
                    diagnostics={},
                )
        except Exception as exc:
            self.diagnostics.add_warning(f"Equal Weight strategy failed: {exc}")

        try:
            mv = self._min_variance_proxy()
            if mv:
                strategies["Minimum Variance"] = StrategyResult(
                    name="Minimum Variance",
                    method="minimum_variance",
                    weights=mv,
                    diagnostics={},
                )
        except Exception as exc:
            self.diagnostics.add_warning(f"Minimum Variance strategy failed: {exc}")

        try:
            rp = self._risk_parity_proxy()
            if rp:
                strategies["Risk Parity"] = StrategyResult(
                    name="Risk Parity",
                    method="risk_parity",
                    weights=rp,
                    diagnostics={},
                )
        except Exception as exc:
            self.diagnostics.add_warning(f"Risk Parity strategy failed: {exc}")

        try:
            mr = self._max_return_proxy()
            if mr:
                strategies["Max Return Proxy"] = StrategyResult(
                    name="Max Return Proxy",
                    method="max_return_proxy",
                    weights=mr,
                    diagnostics={},
                )
        except Exception as exc:
            self.diagnostics.add_warning(f"Max Return Proxy strategy failed: {exc}")

        try:
            te = self._tracking_error_proxy()
            if te:
                strategies["Tracking Error Optimal"] = StrategyResult(
                    name="Tracking Error Optimal",
                    method="tracking_error_optimal",
                    weights=te,
                    diagnostics={
                        "tracking_error_target": 0.0,
                        "ex_ante_tracking_error": 0.0,
                    },
                )
        except Exception as exc:
            self.diagnostics.add_warning(f"Tracking Error strategy failed: {exc}")

        if self.bl_controls.get("enabled", False):
            try:
                bl_result = self._black_litterman_proxy()
                strategies[bl_result.name] = bl_result
            except Exception as exc:
                self.diagnostics.add_warning(f"Black-Litterman strategy failed: {exc}")

        return strategies

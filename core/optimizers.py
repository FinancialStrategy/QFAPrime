from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

from pypfopt import EfficientFrontier, objective_functions
from pypfopt.black_litterman import BlackLittermanModel, market_implied_risk_aversion

from core.config import ProfessionalConfig, RunDiagnostics


@dataclass
class StrategyResult:
    weights: Dict[str, float]
    method: str
    description: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class ProfessionalOptimizer:
    def __init__(
        self,
        mu: pd.Series,
        cov: pd.DataFrame,
        returns: pd.DataFrame,
        benchmark_returns: pd.Series,
        config: ProfessionalConfig,
        diagnostics: RunDiagnostics,
        asset_categories: Dict[str, str],
        asset_metadata: Optional[pd.DataFrame] = None,
        current_weights: Optional[Dict[str, float]] = None,
        bl_controls: Optional[Dict[str, Any]] = None,
    ):
        common = mu.index.intersection(cov.index).intersection(returns.columns)
        self.mu = mu.loc[common]
        self.cov = cov.loc[common, common]
        self.returns = returns[common]
        self.benchmark_returns = benchmark_returns.reindex(self.returns.index).dropna()
        self.returns = self.returns.loc[self.benchmark_returns.index]
        self.config = config
        self.diagnostics = diagnostics
        self.category_map = {k: v for k, v in asset_categories.items() if k in common}
        self.asset_metadata = asset_metadata.copy() if asset_metadata is not None else pd.DataFrame()
        self.bl_controls = bl_controls or {"enabled": False, "view_mode": "ticker", "views_payload": []}

        self.current_weights = pd.Series(
            current_weights or {a: 1 / len(common) for a in common}
        ).reindex(common).fillna(0.0)
        self.current_weights = self.current_weights / self.current_weights.sum()

    def run_all(self) -> Dict[str, StrategyResult]:
        methods = {
            "Max Sharpe": self.max_sharpe,
            "Minimum Volatility": self.min_volatility,
            "Equal Weight": self.equal_weight,
            "Inverse Volatility": self.inverse_volatility,
            "Equal Risk Contribution": self.equal_risk_contribution,
            "Maximum Diversification": self.maximum_diversification,
            "Hierarchical Risk Parity": self.hrp,
            "Black-Litterman": self.black_litterman,
            "Tracking Error Optimal": self.tracking_error_optimal,
        }

        out: Dict[str, StrategyResult] = {}
        for name, fn in methods.items():
            try:
                result = self._attach_costs(fn())
                self._validate(name, result)
                out[name] = result
                self.diagnostics.strategy_diagnostics[name] = result.diagnostics
            except Exception as exc:
                self.diagnostics.strategy_diagnostics[name] = {"error": str(exc)}

        if not out:
            raise ValueError("All strategies failed.")

        return out

    def max_sharpe(self) -> StrategyResult:
        ef = self._make_ef(self.mu, self.cov)
        ef.max_sharpe(risk_free_rate=self.config.risk_free_rate)
        return StrategyResult(
            self._norm(ef.clean_weights()),
            "max_sharpe",
            "Sharpe-optimal portfolio",
        )

    def min_volatility(self) -> StrategyResult:
        ef = self._make_ef(self.mu, self.cov)
        ef.min_volatility()
        return StrategyResult(
            self._norm(ef.clean_weights()),
            "min_volatility",
            "Minimum volatility portfolio",
        )

    def equal_weight(self) -> StrategyResult:
        n = len(self.mu)
        return StrategyResult(
            {a: 1 / n for a in self.mu.index},
            "equal_weight",
            "Equal-weight benchmark",
        )

    def inverse_volatility(self) -> StrategyResult:
        vol = np.sqrt(np.diag(self.cov.values))
        inv = 1 / np.maximum(vol, 1e-12)
        inv = inv / inv.sum()
        return StrategyResult(
            {a: float(w) for a, w in zip(self.mu.index, inv)},
            "inverse_volatility",
            "Inverse-volatility allocation",
        )

    def equal_risk_contribution(self) -> StrategyResult:
        n = len(self.mu)
        x0 = np.ones(n) / n
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(self.config.min_weight, self.config.max_weight)] * n

        def obj(w: np.ndarray) -> float:
            s = self.cov.values
            pv = float(w @ s @ w)
            if pv <= 0:
                return 1e6
            rc = w * (s @ w) / np.sqrt(pv)
            target = np.mean(rc)
            turnover = np.sum(np.abs(w - self.current_weights.values))
            return float(np.sum((rc - target) ** 2) + self.config.turnover_penalty * turnover)

        res = minimize(obj, x0=x0, bounds=bounds, constraints=cons, method="SLSQP")
        if not res.success:
            raise RuntimeError(res.message)

        return StrategyResult(
            self._norm({a: float(w) for a, w in zip(self.mu.index, res.x)}),
            "erc",
            "Equal risk contribution",
            {"optimizer_status": res.message},
        )

    def maximum_diversification(self) -> StrategyResult:
        n = len(self.mu)
        vols = np.sqrt(np.diag(self.cov.values))
        x0 = np.ones(n) / n
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(self.config.min_weight, self.config.max_weight)] * n

        def obj(w: np.ndarray) -> float:
            denom = float(np.sqrt(w @ self.cov.values @ w))
            if denom <= 0:
                return 1e6
            turnover = np.sum(np.abs(w - self.current_weights.values))
            return float(-(w @ vols) / denom + self.config.turnover_penalty * turnover)

        res = minimize(obj, x0=x0, bounds=bounds, constraints=cons, method="SLSQP")
        if not res.success:
            raise RuntimeError(res.message)

        return StrategyResult(
            self._norm({a: float(w) for a, w in zip(self.mu.index, res.x)}),
            "max_diversification",
            "Maximum diversification",
            {"optimizer_status": res.message},
        )

    def hrp(self) -> StrategyResult:
        corr = self.returns.corr().clip(-1, 1)
        dist = np.sqrt((1 - corr) / 2)
        condensed = squareform(dist.values, checks=False)
        link = hierarchy.linkage(condensed, method="ward")
        ordered_idx = self._get_quasi_diag(link)
        ordered_assets = corr.index[ordered_idx].tolist()
        weights = pd.Series(1.0 / len(ordered_assets), index=ordered_assets)

        def cluster_variance(items: List[str]) -> float:
            cov_slice = self.cov.loc[items, items]
            inv_diag = 1 / np.maximum(np.diag(cov_slice.values), 1e-12)
            parity_w = inv_diag / inv_diag.sum()
            return float(parity_w @ cov_slice.values @ parity_w)

        def bisect(items: List[str], w: pd.Series) -> pd.Series:
            if len(items) <= 1:
                return w
            mid = len(items) // 2
            left = items[:mid]
            right = items[mid:]
            v_left = cluster_variance(left)
            v_right = cluster_variance(right)
            alpha = 1 - v_left / (v_left + v_right) if (v_left + v_right) > 0 else 0.5
            w[left] *= alpha
            w[right] *= (1 - alpha)
            w = bisect(left, w)
            w = bisect(right, w)
            return w

        weights = bisect(ordered_assets, weights)
        weights = weights / weights.sum()

        return StrategyResult(
            self._norm(weights.to_dict()),
            "hrp",
            "Hierarchical Risk Parity",
            {"hrp_order": ordered_assets},
        )

    def black_litterman(self) -> StrategyResult:
        benchmark_prices = (1 + self.benchmark_returns).cumprod().rename("benchmark")
        delta = market_implied_risk_aversion(
            benchmark_prices.to_frame(),
            frequency=self.config.annual_trading_days,
        )
        w_mkt = pd.Series(1.0, index=self.mu.index) / len(self.mu)
        pi = delta * (self.cov @ w_mkt)

        abs_views, conf_map = self._resolve_black_litterman_views()

        if abs_views:
            view_confidences = [float(conf_map.get(k, 0.50)) for k in abs_views.keys()]
            bl = BlackLittermanModel(
                self.cov,
                pi=pi,
                absolute_views=abs_views,
                tau=0.05,
                view_confidences=view_confidences,
            )
        else:
            fallback_views = {}
            if "GLD" in self.mu.index:
                fallback_views["GLD"] = float(max(self.mu.loc["GLD"], 0.03))
            if "QQQ" in self.mu.index:
                fallback_views["QQQ"] = float(max(self.mu.loc["QQQ"], 0.05))

            if fallback_views:
                view_confidences = [0.60] * len(fallback_views)
                bl = BlackLittermanModel(
                    self.cov,
                    pi=pi,
                    absolute_views=fallback_views,
                    tau=0.05,
                    view_confidences=view_confidences,
                )
                abs_views = fallback_views
                conf_map = {k: 0.60 for k in fallback_views}
            else:
                bl = BlackLittermanModel(self.cov, pi=pi, tau=0.05)

        posterior_returns = bl.bl_returns()
        posterior_cov = bl.bl_cov()

        ef = self._make_ef(posterior_returns, posterior_cov)
        ef.max_sharpe(risk_free_rate=self.config.risk_free_rate)
        clean_weights = self._norm(ef.clean_weights())

        prior_series = pd.Series(pi).reindex(self.mu.index)
        posterior_series = pd.Series(posterior_returns).reindex(self.mu.index)

        diagnostics = {
            "views_used": abs_views,
            "view_confidences_used": conf_map,
            "prior_returns": prior_series.to_dict(),
            "posterior_returns": posterior_series.to_dict(),
            "posterior_covariance_trace": float(np.trace(posterior_cov.values)),
            "bl_weight_output": clean_weights,
            "bl_view_mode": self.bl_controls.get("view_mode", "ticker"),
        }

        return StrategyResult(
            clean_weights,
            "black_litterman",
            "Black-Litterman",
            diagnostics,
        )

    def tracking_error_optimal(self) -> StrategyResult:
        assets = list(self.mu.index)
        benchmark_proxy = self._benchmark_proxy_weights(assets)
        x0 = benchmark_proxy.values.copy()
        bounds = [(self.config.min_weight, self.config.max_weight)] * len(assets)
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        b = self.benchmark_returns.values
        r = self.returns[assets].values
        mu = self.mu.values

        def obj(w: np.ndarray) -> float:
            port = r @ w
            te = np.std(port - b) * np.sqrt(self.config.annual_trading_days)
            active_return = float(np.dot(w - benchmark_proxy.values, mu))
            turnover = np.sum(np.abs(w - self.current_weights.values))
            te_penalty = 0.0
            if te > self.config.tracking_error_target:
                te_penalty = 50.0 * (te - self.config.tracking_error_target) ** 2
            return -active_return + te_penalty + self.config.turnover_penalty * turnover

        res = minimize(obj, x0=x0, bounds=bounds, constraints=cons, method="SLSQP")
        if not res.success:
            raise RuntimeError(res.message)

        realized_te = float(np.std((r @ res.x) - b) * np.sqrt(self.config.annual_trading_days))
        weights = {a: float(w) for a, w in zip(assets, res.x)}

        return StrategyResult(
            self._norm(weights),
            "tracking_error_optimal",
            "Tracking-error-constrained active portfolio",
            {
                "tracking_error_target": self.config.tracking_error_target,
                "ex_ante_tracking_error": realized_te,
            },
        )

    def _resolve_black_litterman_views(self) -> tuple[Dict[str, float], Dict[str, float]]:
        enabled = bool(self.bl_controls.get("enabled", False))
        if not enabled:
            return {}, {}

        payload = self.bl_controls.get("views_payload", [])
        view_mode = self.bl_controls.get("view_mode", "ticker")

        if not payload:
            return {}, {}

        abs_views: Dict[str, float] = {}
        conf_map: Dict[str, float] = {}

        if view_mode == "ticker":
            for item in payload:
                tgt = item["target"]
                if tgt in self.mu.index:
                    abs_views[tgt] = float(item["expected_return"])
                    conf_map[tgt] = float(item["confidence"])
            return abs_views, conf_map

        # region-mapped views
        if self.asset_metadata is None or self.asset_metadata.empty:
            return {}, {}

        meta = self.asset_metadata[["ticker", "region_type"]].drop_duplicates()
        meta = meta[meta["ticker"].isin(self.mu.index)]

        for item in payload:
            region = item["target"]
            expected_return = float(item["expected_return"])
            confidence = float(item["confidence"])

            region_tickers = meta.loc[meta["region_type"] == region, "ticker"].tolist()
            for ticker in region_tickers:
                abs_views[ticker] = expected_return
                conf_map[ticker] = confidence

        return abs_views, conf_map

    def _make_ef(self, mu: pd.Series, cov: pd.DataFrame) -> EfficientFrontier:
        ef = EfficientFrontier(
            mu,
            cov,
            weight_bounds=(self.config.min_weight, self.config.max_weight),
        )
        ef.add_objective(objective_functions.L2_reg, gamma=0.02)
        self._add_category_constraints(ef)
        return ef

    def _add_category_constraints(self, ef: EfficientFrontier) -> None:
        categories: Dict[str, List[int]] = {}
        assets = list(self.mu.index)
        for i, asset in enumerate(assets):
            category = self.category_map.get(asset)
            if category is not None:
                categories.setdefault(category, []).append(i)

        for _, idxs in categories.items():
            ef.add_constraint(
                lambda w, idxs=idxs: self.config.max_category_weight - sum(w[i] for i in idxs)
            )

    def _attach_costs(self, result: StrategyResult) -> StrategyResult:
        wn = pd.Series(result.weights).reindex(self.mu.index).fillna(0.0)
        turnover = float(np.abs(wn - self.current_weights).sum())
        tc = turnover * (self.config.transaction_cost_bps / 10000.0) * self.config.initial_capital
        result.diagnostics.update({
            "turnover": turnover,
            "estimated_transaction_cost_usd": tc,
        })
        return result

    def _validate(self, name: str, result: StrategyResult) -> None:
        total = sum(result.weights.values())
        if not np.isclose(total, 1.0, atol=1e-4):
            raise ValueError(f"{name}: weights sum to {total:.6f}")

    def _benchmark_proxy_weights(self, assets: List[str]) -> pd.Series:
        if self.config.benchmark in assets:
            w = pd.Series(0.0, index=assets)
            w[self.config.benchmark] = 1.0
            return w
        if "SPY" in assets:
            w = pd.Series(0.0, index=assets)
            w["SPY"] = 1.0
            return w
        return pd.Series(1 / len(assets), index=assets)

    def _get_quasi_diag(self, link: np.ndarray) -> List[int]:
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = int(link[-1, 3])

        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix.loc[i] = link[j, 0]
            df1 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df1]).sort_index()
            sort_ix.index = range(sort_ix.shape[0])

        return sort_ix.tolist()

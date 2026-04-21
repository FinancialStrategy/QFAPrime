from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from scipy.optimize import minimize
from sklearn.decomposition import PCA

from core.universes import UNIVERSE_REGISTRY, get_universe_tickers


logger = logging.getLogger("QFA_QUANT_PLATFORM")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


@dataclass
class RunDiagnostics:
    warnings_list: List[str] = field(default_factory=list)
    errors_list: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)

    def add_warning(self, message: str) -> None:
        self.warnings_list.append(str(message))

    def add_error(self, message: str) -> None:
        self.errors_list.append(str(message))

    def add_info(self, message: str) -> None:
        self.info.append(str(message))

    def summary(self) -> Dict[str, List[str]]:
        return {
            "warnings": self.warnings_list,
            "errors": self.errors_list,
            "info": self.info,
        }


@dataclass
class StrategyResult:
    weights: Dict[str, float]
    method: str
    description: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def _normalize_weights_dict(weights: Dict[str, float]) -> Dict[str, float]:
    s = pd.Series(weights, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    total = float(s.sum())
    if abs(total) < 1e-12:
        n = len(s)
        return {k: 1.0 / n for k in s.index} if n > 0 else {}
    s = s / total
    return {k: float(v) for k, v in s.items()}


def _drawdown_series(return_series: pd.Series) -> pd.Series:
    wealth = (1.0 + return_series.fillna(0.0)).cumprod()
    peak = wealth.cummax()
    return wealth / peak - 1.0


class ProfessionalPortfolioEngine:
    def __init__(
        self,
        config,
        bl_controls: Optional[Dict[str, Any]] = None,
        scenario_controls: Optional[Dict[str, Any]] = None,
    ):
        self.config = config
        self.bl_controls = bl_controls or {}
        self.scenario_controls = scenario_controls or {}

        self.diagnostics = RunDiagnostics()
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.metrics_df = pd.DataFrame()
        self.strategy_df = pd.DataFrame()
        self.stress_df = pd.DataFrame()
        self.stress_table = pd.DataFrame()
        self.risk_contrib_df = pd.DataFrame()
        self.rolling_beta_df = pd.DataFrame()
        self.beta_summary_df = pd.DataFrame()
        self.factor_pca_df = pd.DataFrame()
        self.charts: Dict[str, Any] = {}
        self.finquant_charts: Dict[str, Any] = {}

        self.prices = pd.DataFrame()
        self.returns = pd.DataFrame()
        self.benchmark_returns = pd.Series(dtype=float)

    def _resolve_universe(self) -> List[str]:
        selected_universe = getattr(self.config, "selected_universe", "institutional_multi_asset")

        tickers = get_universe_tickers(selected_universe)

        if len(tickers) < 2:
            fallback = get_universe_tickers("institutional_multi_asset")
            self.diagnostics.add_warning(
                f"Universe '{selected_universe}' was invalid. Falling back to 'institutional_multi_asset'."
            )
            tickers = fallback
            self.config.selected_universe = "institutional_multi_asset"

        if len(tickers) < 2:
            raise ValueError(
                f"Selected universe '{selected_universe}' does not contain enough assets. Resolved tickers: {tickers}"
            )

        self.diagnostics.add_info(f"Resolved universe '{self.config.selected_universe}' with tickers: {tickers}")
        return tickers

    def _download_prices(self, tickers: List[str]) -> pd.DataFrame:
        price_map = {}
        for ticker in tickers:
            try:
                hist = yf.Ticker(ticker).history(
                    start=self.config.start_date,
                    end=self.config.end_date,
                    auto_adjust=True,
                )
                if hist.empty or "Close" not in hist.columns:
                    self.diagnostics.add_warning(f"{ticker}: no close history")
                    continue
                close = hist["Close"].copy()
                if getattr(close.index, "tz", None) is not None:
                    close.index = close.index.tz_localize(None)
                price_map[ticker] = close.rename(ticker)
                time.sleep(0.4 + random.uniform(0.1, 0.3))
            except Exception as exc:
                self.diagnostics.add_warning(f"{ticker}: {exc}")

        if len(price_map) < 2:
            raise ValueError(f"Too few assets downloaded successfully: {len(price_map)}")

        prices = pd.concat(price_map.values(), axis=1).sort_index().ffill(limit=3).dropna(how="all")
        return prices

    def _download_benchmark(self) -> pd.Series:
        try:
            b = yf.Ticker(self.config.benchmark).history(
                start=self.config.start_date,
                end=self.config.end_date,
                auto_adjust=True,
            )["Close"]
            if getattr(b.index, "tz", None) is not None:
                b.index = b.index.tz_localize(None)
            return b.pct_change().dropna()
        except Exception as exc:
            self.diagnostics.add_warning(f"Benchmark download failed: {exc}")
            return pd.Series(dtype=float)

    def _estimate_inputs(self):
        ann = 252
        mu = self.returns.mean() * ann
        cov = self.returns.cov() * ann

        vals, vecs = np.linalg.eigh(cov.values)
        if vals.min() < 0:
            vals = np.clip(vals, 1e-10, None)
            cov = pd.DataFrame(vecs @ np.diag(vals) @ vecs.T, index=cov.index, columns=cov.columns)

        return mu, cov

    def _portfolio_returns(self, returns: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
        w = pd.Series(weights).reindex(returns.columns).fillna(0.0)
        w = w / w.sum()
        return returns.mul(w, axis=1).sum(axis=1)

    def _build_strategies(self, mu: pd.Series, cov: pd.DataFrame) -> Dict[str, StrategyResult]:
        assets = list(mu.index)
        n = len(assets)

        equal_weight = {a: 1.0 / n for a in assets}

        inv_vol = 1 / np.maximum(np.sqrt(np.diag(cov.values)), 1e-12)
        inv_vol = inv_vol / inv_vol.sum()
        inverse_vol = {a: float(w) for a, w in zip(assets, inv_vol)}

        def min_var():
            x0 = np.repeat(1.0 / n, n)
            bounds = [(0, 1)] * n if not self.config.allow_short else [(-1, 1)] * n
            cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

            def obj(w):
                return float(w @ cov.values @ w)

            res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons)
            if not res.success:
                return equal_weight
            return _normalize_weights_dict({a: float(w) for a, w in zip(assets, res.x)})

        def max_sharpe():
            x0 = np.repeat(1.0 / n, n)
            bounds = [(0, 1)] * n if not self.config.allow_short else [(-1, 1)] * n
            cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

            def obj(w):
                port_ret = float(np.dot(w, mu.values))
                port_vol = float(np.sqrt(max(w @ cov.values @ w, 1e-12)))
                return -((port_ret - self.config.risk_free_rate) / port_vol)

            res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons)
            if not res.success:
                return equal_weight
            return _normalize_weights_dict({a: float(w) for a, w in zip(assets, res.x)})

        strategies = {
            "Max Sharpe": StrategyResult(max_sharpe(), "max_sharpe", "Sharpe-optimal portfolio"),
            "Minimum Volatility": StrategyResult(min_var(), "min_volatility", "Minimum volatility portfolio"),
            "Equal Weight": StrategyResult(equal_weight, "equal_weight", "Equal-weight benchmark"),
            "Inverse Volatility": StrategyResult(inverse_vol, "inverse_volatility", "Inverse-volatility allocation"),
        }

        return strategies

    def _calculate_metrics(self, pr: pd.Series, br: pd.Series) -> Dict[str, Any]:
        ann = 252
        aligned = pd.concat([pr, br], axis=1).dropna()
        aligned.columns = ["portfolio", "benchmark"]

        if aligned.empty:
            raise ValueError("Portfolio and benchmark series could not be aligned.")

        portfolio_values = (1 + aligned["portfolio"]).cumprod() * self.config.initial_capital
        benchmark_values = (1 + aligned["benchmark"]).cumprod() * self.config.initial_capital

        total_return_portfolio = portfolio_values.iloc[-1] / self.config.initial_capital - 1
        total_return_benchmark = benchmark_values.iloc[-1] / self.config.initial_capital - 1

        n_years = len(aligned) / ann
        annual_return_portfolio = (1 + total_return_portfolio) ** (1 / n_years) - 1 if n_years > 0 else 0.0
        annual_return_benchmark = (1 + total_return_benchmark) ** (1 / n_years) - 1 if n_years > 0 else 0.0

        volatility = float(aligned["portfolio"].std() * np.sqrt(ann))
        sharpe = (annual_return_portfolio - self.config.risk_free_rate) / volatility if volatility > 0 else np.nan

        downside = aligned["portfolio"][aligned["portfolio"] < 0]
        downside_dev = float(downside.std() * np.sqrt(ann)) if len(downside) > 1 else np.nan
        sortino = (annual_return_portfolio - self.config.risk_free_rate) / downside_dev if downside_dev and downside_dev > 0 else np.nan

        drawdown = _drawdown_series(aligned["portfolio"])
        max_drawdown = float(drawdown.min())
        calmar = annual_return_portfolio / abs(max_drawdown) if max_drawdown < 0 else np.nan

        covariance = aligned["portfolio"].cov(aligned["benchmark"])
        benchmark_variance = aligned["benchmark"].var()
        beta = float(covariance / benchmark_variance) if benchmark_variance > 0 else np.nan
        alpha = float(annual_return_portfolio - (self.config.risk_free_rate + beta * (annual_return_benchmark - self.config.risk_free_rate))) if pd.notna(beta) else np.nan

        tracking_diff = aligned["portfolio"] - aligned["benchmark"]
        tracking_error = float(tracking_diff.std() * np.sqrt(ann))
        information_ratio = float((tracking_diff.mean() * ann) / tracking_error) if tracking_error > 0 else np.nan

        win_rate = float((aligned["portfolio"] > 0).mean())
        win_rate_vs_benchmark = float((aligned["portfolio"] > aligned["benchmark"]).mean())
        profit_factor = _profit_factor(aligned["portfolio"])

        q95 = np.quantile(aligned["portfolio"], 0.05)
        tail95 = aligned["portfolio"][aligned["portfolio"] <= q95]
        aq95 = np.quantile(tracking_diff, 0.05)
        atail95 = tracking_diff[tracking_diff <= aq95]

        q99 = np.quantile(aligned["portfolio"], 0.01)
        tail99 = aligned["portfolio"][aligned["portfolio"] <= q99]

        return {
            "final_portfolio_value": float(portfolio_values.iloc[-1]),
            "final_benchmark_value": float(benchmark_values.iloc[-1]),
            "total_return_pct": float(total_return_portfolio),
            "total_return_benchmark_pct": float(total_return_benchmark),
            "excess_return_vs_benchmark_pct": float(total_return_portfolio - total_return_benchmark),
            "annual_return": float(annual_return_portfolio),
            "annual_return_benchmark": float(annual_return_benchmark),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe) if pd.notna(sharpe) else np.nan,
            "sortino_ratio": float(sortino) if pd.notna(sortino) else np.nan,
            "calmar_ratio": float(calmar) if pd.notna(calmar) else np.nan,
            "max_drawdown": float(max_drawdown),
            "alpha": alpha,
            "beta": beta,
            "tracking_error": float(tracking_error),
            "information_ratio": information_ratio,
            "win_rate": win_rate,
            "win_rate_vs_benchmark": win_rate_vs_benchmark,
            "profit_factor": profit_factor,
            "portfolio_values": portfolio_values,
            "benchmark_values": benchmark_values,
            "drawdown_series": drawdown,
            "portfolio_returns": aligned["portfolio"],
            "benchmark_returns": aligned["benchmark"],
            "var_95": float(-q95),
            "cvar_95": float(-tail95.mean()) if len(tail95) else np.nan,
            "relative_var_95": float(-aq95),
            "relative_cvar_95": float(-atail95.mean()) if len(atail95) else np.nan,
            "VaR_95": float(-q95),
            "CVaR_95": float(-tail95.mean()) if len(tail95) else np.nan,
            "VaR_99": float(-q99),
            "CVaR_99": float(-tail99.mean()) if len(tail99) else np.nan,
        }

    def _historical_stress_tests(self, pr: pd.Series, br: pd.Series) -> pd.DataFrame:
        scenarios = {
            "COVID Crash": ("2020-02-19", "2020-03-23"),
            "2022 Inflation Shock": ("2022-01-03", "2022-10-14"),
            "2023 Banking Stress": ("2023-03-08", "2023-03-31"),
            "2024 Q1 Rally": ("2024-01-02", "2024-03-28"),
        }

        rows = []
        for name, (s, e) in scenarios.items():
            mask = (pr.index >= pd.Timestamp(s)) & (pr.index <= pd.Timestamp(e))
            p = pr.loc[mask]
            b = br.reindex(p.index).dropna()
            p = p.reindex(b.index)
            if len(p) < 5:
                continue

            p_total = float((1 + p).prod() - 1)
            b_total = float((1 + b).prod() - 1)
            rows.append(
                {
                    "scenario": name,
                    "portfolio_return": p_total,
                    "benchmark_return": b_total,
                    "relative_return": p_total - b_total,
                    "duration_days": len(p),
                }
            )
        return pd.DataFrame(rows)

    def _compute_risk_contributions(self, weights: Dict[str, float], cov: pd.DataFrame) -> pd.DataFrame:
        assets = list(weights.keys())
        w = np.array([weights[a] for a in assets])
        S = cov.loc[assets, assets].values
        port_var = w @ S @ w

        if port_var <= 0:
            return pd.DataFrame()

        port_vol = np.sqrt(port_var)
        marginal_risk = (S @ w) / port_vol
        total_contrib = w * marginal_risk

        return pd.DataFrame(
            {
                "Asset": assets,
                "Weight": w,
                "Marginal Risk": marginal_risk,
                "Total Risk Contribution": total_contrib,
                "Contribution %": total_contrib / port_vol,
            }
        ).sort_values("Total Risk Contribution", ascending=False)

    def _rolling_beta(self, returns_df: pd.DataFrame, benchmark_returns: pd.Series, window: int) -> pd.DataFrame:
        aligned = pd.concat([returns_df, benchmark_returns.rename("benchmark")], axis=1).dropna()
        if aligned.empty:
            return pd.DataFrame()

        out = {}
        for col in returns_df.columns:
            pair = aligned[[col, "benchmark"]].dropna()
            if len(pair) < window:
                continue
            cov = pair[col].rolling(window).cov(pair["benchmark"])
            varb = pair["benchmark"].rolling(window).var()
            out[col] = cov / varb.replace(0, np.nan)
        return pd.DataFrame(out)

    def _rolling_beta_summary(self, rolling_beta_df: pd.DataFrame) -> pd.DataFrame:
        if rolling_beta_df.empty:
            return pd.DataFrame()
        return pd.DataFrame(
            {
                "mean_beta": rolling_beta_df.mean(),
                "min_beta": rolling_beta_df.min(),
                "max_beta": rolling_beta_df.max(),
                "latest_beta": rolling_beta_df.iloc[-1],
            }
        )

    def _factor_pca(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        clean = returns_df.dropna(how="any")
        if clean.shape[1] < 2 or clean.shape[0] < 20:
            return pd.DataFrame()

        n_components = min(3, clean.shape[1])
        pca = PCA(n_components=n_components)
        pca.fit(clean.values)

        loading_df = pd.DataFrame(
            pca.components_.T,
            index=clean.columns,
            columns=[f"PC{i+1}" for i in range(n_components)],
        ).reset_index().rename(columns={"index": "factor_or_asset"})

        explained_rows = []
        for i in range(n_components):
            row = {"factor_or_asset": f"PC{i+1}_explained"}
            for j in range(n_components):
                row[f"PC{j+1}"] = pca.explained_variance_ratio_[i] if i == j else np.nan
            explained_rows.append(row)

        return pd.concat([loading_df, pd.DataFrame(explained_rows)], ignore_index=True)

    def _build_charts(self, mu, cov):
        vis = {}
        if not self.metrics_df.empty:
            vis["dashboard"] = px.bar(
                self.metrics_df.reset_index(),
                x=self.metrics_df.reset_index().columns[0],
                y=[c for c in ["annual_return", "sharpe_ratio", "information_ratio"] if c in self.metrics_df.columns],
                barmode="group",
                title="Executive Strategy Dashboard",
            ).update_layout(template="plotly_white", height=560)

            vis["tracking_error"] = (
                px.bar(
                    self.metrics_df.reset_index(),
                    x=self.metrics_df.reset_index().columns[0],
                    y="tracking_error",
                    title="Tracking Error by Strategy",
                ).update_layout(template="plotly_white", height=450)
                if "tracking_error" in self.metrics_df.columns
                else None
            )

        self.charts = vis
        self.finquant_charts = {}

    def run(self):
        tickers = self._resolve_universe()

        self.prices = self._download_prices(tickers)
        self.returns = self.prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")

        if self.returns.shape[1] < 2:
            raise ValueError("Not enough valid assets after returns calculation.")

        self.benchmark_returns = self._download_benchmark()
        if self.benchmark_returns.empty:
            self.benchmark_returns = self.returns.mean(axis=1)
            self.diagnostics.add_warning("Benchmark fallback: using equal-weight proxy.")

        common = self.returns.index.intersection(self.benchmark_returns.index)
        self.returns = self.returns.loc[common]
        self.prices = self.prices.loc[common]
        self.benchmark_returns = self.benchmark_returns.loc[common]

        if len(self.returns) < self.config.min_observations:
            raise ValueError(
                f"Return matrix has only {len(self.returns)} rows, below minimum observations {self.config.min_observations}."
            )

        mu, cov = self._estimate_inputs()
        strategies = self._build_strategies(mu, cov)

        self.metrics = {}
        for name, result in strategies.items():
            pr = self._portfolio_returns(self.returns[mu.index], result.weights)
            m = self._calculate_metrics(pr, self.benchmark_returns)
            m["weights"] = pd.Series(result.weights)
            self.metrics[name] = m

        self.metrics_df = pd.DataFrame(self.metrics).T.sort_values("sharpe_ratio", ascending=False)

        self.strategy_df = pd.DataFrame(
            [
                {
                    "strategy": name,
                    "method": result.method,
                    "num_assets": len([w for w in result.weights.values() if w > 0.001]),
                    "max_weight": max(result.weights.values()),
                    "top_3_assets": ", ".join(
                        [asset for asset, weight in sorted(result.weights.items(), key=lambda x: x[1], reverse=True)[:3]]
                    ),
                }
                for name, result in strategies.items()
            ]
        )

        best_strategy = self.best_strategy_name()
        best_metrics = self.metrics[best_strategy]

        self.stress_df = self._historical_stress_tests(
            best_metrics["portfolio_returns"],
            best_metrics["benchmark_returns"],
        )
        self.stress_table = self.stress_df.copy()

        self.risk_contrib_df = self._compute_risk_contributions(
            strategies[best_strategy].weights,
            cov,
        )

        self.rolling_beta_df = self._rolling_beta(
            self.returns[mu.index],
            self.benchmark_returns,
            int(self.config.rolling_window),
        )
        self.beta_summary_df = self._rolling_beta_summary(self.rolling_beta_df)
        self.factor_pca_df = self._factor_pca(self.returns[mu.index])

        self._build_charts(mu, cov)

        self.diagnostics.add_info(f"Loaded {len(tickers)} requested tickers.")
        self.diagnostics.add_info(f"Retained {self.returns.shape[1]} valid assets.")
        return self

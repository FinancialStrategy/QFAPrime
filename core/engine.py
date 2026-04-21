from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import PCA

from core.data_loader import compute_returns, load_price_data
from core.universes import UNIVERSE_REGISTRY, get_universe_tickers


# =========================================================
# Diagnostics
# =========================================================
@dataclass
class Diagnostics:
    warnings_list: List[str] = field(default_factory=list)
    errors_list: List[str] = field(default_factory=list)
    info_list: List[str] = field(default_factory=list)

    def add_warning(self, message: str) -> None:
        self.warnings_list.append(str(message))

    def add_error(self, message: str) -> None:
        self.errors_list.append(str(message))

    def add_info(self, message: str) -> None:
        self.info_list.append(str(message))

    def summary(self) -> Dict[str, List[str]]:
        return {
            "warnings": self.warnings_list,
            "errors": self.errors_list,
            "info": self.info_list,
        }


# =========================================================
# Helpers
# =========================================================
def _annualization_factor() -> int:
    return 252


def _safe_div(a: float, b: float, default: float = np.nan) -> float:
    if b is None or pd.isna(b) or abs(float(b)) < 1e-12:
        return default
    return float(a) / float(b)


def _validate_input_frame(name: str, df: pd.DataFrame) -> None:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError(f"{name} is empty.")
    if df.shape[1] < 2:
        raise ValueError(f"{name} must contain at least two assets.")
    if df.shape[0] < 30:
        raise ValueError(f"{name} does not contain enough observations (need at least 30 rows).")


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    total = np.sum(w)
    if abs(total) < 1e-12:
        return np.repeat(1.0 / len(w), len(w))
    return w / total


def _portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    return returns.dot(weights)


def _drawdown_series(return_series: pd.Series) -> pd.Series:
    wealth = (1.0 + return_series.fillna(0.0)).cumprod()
    peak = wealth.cummax()
    return wealth / peak - 1.0


def _max_drawdown(return_series: pd.Series) -> float:
    dd = _drawdown_series(return_series)
    if dd.empty:
        return np.nan
    return float(dd.min())


def _historical_var(series: pd.Series, alpha: float = 0.05) -> float:
    s = series.dropna()
    if s.empty:
        return np.nan
    return float(np.quantile(s, alpha))


def _historical_cvar(series: pd.Series, alpha: float = 0.05) -> float:
    s = series.dropna()
    if s.empty:
        return np.nan
    var = np.quantile(s, alpha)
    tail = s[s <= var]
    if tail.empty:
        return np.nan
    return float(tail.mean())


def _profit_factor(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return np.nan
    gains = s[s > 0].sum()
    losses = -s[s < 0].sum()
    if losses <= 0:
        return np.nan
    return float(gains / losses)


def _tracking_error(active_returns: pd.Series) -> float:
    s = active_returns.dropna()
    if s.empty:
        return np.nan
    return float(s.std(ddof=1) * np.sqrt(_annualization_factor()))


def _information_ratio(active_returns: pd.Series) -> float:
    s = active_returns.dropna()
    if s.empty:
        return np.nan
    te_daily = s.std(ddof=1)
    if te_daily <= 0 or pd.isna(te_daily):
        return np.nan
    return float((s.mean() / te_daily) * np.sqrt(_annualization_factor()))


def _beta_alpha(portfolio_returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float) -> Tuple[float, float]:
    df = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if df.empty or df.shape[0] < 20:
        return np.nan, np.nan

    rp = df.iloc[:, 0]
    rb = df.iloc[:, 1]

    var_b = np.var(rb, ddof=1)
    if var_b <= 0 or pd.isna(var_b):
        return np.nan, np.nan

    beta = np.cov(rp, rb, ddof=1)[0, 1] / var_b

    ann = _annualization_factor()
    ann_rp = rp.mean() * ann
    ann_rb = rb.mean() * ann
    alpha = ann_rp - (risk_free_rate + beta * (ann_rb - risk_free_rate))
    return float(beta), float(alpha)


def _compute_performance_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    initial_capital: float,
    risk_free_rate: float,
) -> Dict[str, Any]:
    ann = _annualization_factor()

    pr = portfolio_returns.dropna()
    br = benchmark_returns.dropna()

    if pr.empty:
        raise ValueError("Portfolio return series is empty.")

    total_return = float((1.0 + pr).prod() - 1.0)
    total_benchmark_return = float((1.0 + br).prod() - 1.0) if not br.empty else np.nan

    annual_return = float(pr.mean() * ann)
    annual_benchmark_return = float(br.mean() * ann) if not br.empty else np.nan
    volatility = float(pr.std(ddof=1) * np.sqrt(ann))

    downside = pr[pr < 0]
    downside_vol = float(downside.std(ddof=1) * np.sqrt(ann)) if len(downside) > 1 else np.nan

    sharpe = _safe_div(annual_return - risk_free_rate, volatility)
    sortino = _safe_div(annual_return - risk_free_rate, downside_vol)

    mdd = _max_drawdown(pr)
    calmar = _safe_div(annual_return, abs(mdd)) if pd.notna(mdd) and mdd != 0 else np.nan

    dd_series = _drawdown_series(pr)
    bench_dd_series = _drawdown_series(br) if not br.empty else pd.Series(dtype=float)

    aligned = pd.concat([pr, br], axis=1).dropna()
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1] if not aligned.empty else pd.Series(dtype=float)

    beta, alpha = _beta_alpha(pr, br, risk_free_rate) if not br.empty else (np.nan, np.nan)
    te = _tracking_error(active) if not active.empty else np.nan
    ir = _information_ratio(active) if not active.empty else np.nan

    final_portfolio_value = float(initial_capital) * (1.0 + total_return)
    win_rate = float((pr > 0).mean()) if not pr.empty else np.nan
    profit_factor = _profit_factor(pr)

    return {
        "portfolio_returns": pr,
        "benchmark_returns": br,
        "drawdown_series": dd_series,
        "benchmark_drawdown_series": bench_dd_series,
        "total_return_pct": total_return,
        "total_return_benchmark_pct": total_benchmark_return,
        "excess_return_vs_benchmark_pct": total_return - total_benchmark_return if pd.notna(total_benchmark_return) else np.nan,
        "annual_return": annual_return,
        "annual_return_benchmark": annual_benchmark_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_drawdown": mdd,
        "alpha": alpha,
        "beta": beta,
        "tracking_error": te,
        "information_ratio": ir,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "final_portfolio_value": final_portfolio_value,
        "VaR_95": _historical_var(pr, 0.05),
        "CVaR_95": _historical_cvar(pr, 0.05),
        "VaR_99": _historical_var(pr, 0.01),
        "CVaR_99": _historical_cvar(pr, 0.01),
    }


def _equal_weight(n_assets: int) -> np.ndarray:
    return np.repeat(1.0 / n_assets, n_assets)


def _min_variance_weights(cov: pd.DataFrame, allow_short: bool) -> np.ndarray:
    n = cov.shape[0]
    x0 = np.repeat(1.0 / n, n)

    def objective(w: np.ndarray) -> float:
        return float(w.T @ cov.values @ w)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = None if allow_short else [(0.0, 1.0)] * n

    res = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )

    if not res.success:
        return x0

    return _normalize_weights(np.asarray(res.x, dtype=float))


def _max_sharpe_weights(mu: pd.Series, cov: pd.DataFrame, risk_free_rate: float, allow_short: bool) -> np.ndarray:
    n = len(mu)
    x0 = np.repeat(1.0 / n, n)

    def objective(w: np.ndarray) -> float:
        port_ret = float(np.dot(w, mu.values))
        port_var = float(w.T @ cov.values @ w)
        port_vol = float(np.sqrt(max(port_var, 1e-16)))
        sharpe = (port_ret - risk_free_rate) / port_vol
        return -sharpe

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = None if allow_short else [(0.0, 1.0)] * n

    res = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )

    if not res.success:
        return x0

    return _normalize_weights(np.asarray(res.x, dtype=float))


def _build_stress_table(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    selected_family: str = "All",
    minimum_severity_threshold: float = 0.0,
    quick_view: str = "All",
) -> pd.DataFrame:
    scenarios = [
        {"scenario_name": "Global Financial Crisis", "family": "Crisis", "shock": -0.20},
        {"scenario_name": "Pandemic Shock", "family": "Crisis", "shock": -0.15},
        {"scenario_name": "Inflation Shock", "family": "Inflation", "shock": -0.08},
        {"scenario_name": "Banking Stress", "family": "Banking_Stress", "shock": -0.10},
        {"scenario_name": "Sharp Rally", "family": "Sharp_Rally", "shock": 0.10},
        {"scenario_name": "Sharp Selloff", "family": "Sharp_Selloff", "shock": -0.12},
    ]

    base_port = float(portfolio_returns.mean()) if not portfolio_returns.empty else np.nan
    base_bench = float(benchmark_returns.mean()) if not benchmark_returns.empty else np.nan

    rows = []
    for s in scenarios:
        portfolio_return = base_port + s["shock"]
        benchmark_return = base_bench + 0.85 * s["shock"] if pd.notna(base_bench) else np.nan
        relative_return = portfolio_return - benchmark_return if pd.notna(benchmark_return) else np.nan
        severity_score = abs(s["shock"])

        rows.append(
            {
                "scenario_name": s["scenario_name"],
                "family": s["family"],
                "portfolio_return": portfolio_return,
                "benchmark_return": benchmark_return,
                "relative_return": relative_return,
                "severity_score": severity_score,
            }
        )

    df = pd.DataFrame(rows)

    if quick_view != "All":
        mapping = {
            "Crisis Only": "Crisis",
            "Inflation Only": "Inflation",
            "Banking Stress Only": "Banking_Stress",
            "Sharp Rally Only": "Sharp_Rally",
            "Sharp Selloff Only": "Sharp_Selloff",
        }
        fam = mapping.get(quick_view)
        if fam:
            df = df[df["family"] == fam].copy()

    if selected_family and selected_family != "All":
        df = df[df["family"].str.lower() == selected_family.lower()].copy()

    df = df[df["severity_score"] >= float(minimum_severity_threshold)].copy()
    df = df.sort_values(["severity_score", "relative_return"], ascending=[False, True]).reset_index(drop=True)
    return df


def _build_factor_pca_df(returns: pd.DataFrame) -> pd.DataFrame:
    if returns is None or returns.empty or returns.shape[1] < 2:
        return pd.DataFrame()

    clean = returns.dropna(how="any")
    if clean.shape[0] < 20:
        return pd.DataFrame()

    n_components = min(3, clean.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(clean.values)

    loading_df = pd.DataFrame(
        pca.components_.T,
        index=clean.columns,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )

    explained_df = pd.DataFrame(
        {
            "factor_or_asset": [f"PC{i+1}_explained" for i in range(n_components)],
            **{
                f"PC{i+1}": [pca.explained_variance_ratio_[i] if j == i else np.nan for j in range(n_components)]
                for i in range(n_components)
            },
        }
    )

    out = loading_df.reset_index().rename(columns={"index": "factor_or_asset"})
    out = pd.concat([out, explained_df], ignore_index=True)
    return out


# =========================================================
# Engine
# =========================================================
class ProfessionalPortfolioEngine:
    def __init__(
        self,
        config: Any,
        bl_controls: Optional[Dict[str, Any]] = None,
        scenario_controls: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.config = config
        self.bl_controls = bl_controls or {}
        self.scenario_controls = scenario_controls or {}

        self.diagnostics = Diagnostics()

        self.universe: List[str] = []
        self.benchmark_symbol: str = getattr(config, "benchmark_symbol", "^GSPC")

        self.prices: pd.DataFrame = pd.DataFrame()
        self.returns: pd.DataFrame = pd.DataFrame()
        self.benchmark_prices: pd.DataFrame = pd.DataFrame()
        self.benchmark_returns: pd.Series = pd.Series(dtype=float)

        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.metrics_df: pd.DataFrame = pd.DataFrame()
        self.strategy_df: pd.DataFrame = pd.DataFrame()
        self.stress_table: pd.DataFrame = pd.DataFrame()
        self.factor_pca_df: pd.DataFrame = pd.DataFrame()

    # -----------------------------------------------------
    # Universe resolution
    # -----------------------------------------------------
    def _resolve_universe(self) -> List[str]:
        selected_universe = getattr(self.config, "selected_universe", None)

        if selected_universe not in UNIVERSE_REGISTRY:
            raise ValueError(f"Selected universe '{selected_universe}' was not found in UNIVERSE_REGISTRY.")

        tickers = get_universe_tickers(selected_universe)

        if len(tickers) < 2:
            raise ValueError(
                f"Selected universe '{selected_universe}' does not contain enough assets. "
                f"Resolved tickers: {tickers}"
            )

        return tickers

    # -----------------------------------------------------
    # Data loading
    # -----------------------------------------------------
    def _load_market_data(self) -> None:
        self.universe = self._resolve_universe()
        start = str(getattr(self.config, "default_start_date", "2019-01-01"))
        end = pd.Timestamp.today().strftime("%Y-%m-%d")

        prices, meta = load_price_data(
            tickers=self.universe,
            start=start,
            end=end,
            auto_adjust=False,
            batch_size=3,
        )

        if meta.get("failed"):
            self.diagnostics.add_warning(
                "Some universe tickers could not be downloaded: " + ", ".join(meta["failed"])
            )

        if prices is None or prices.empty:
            raise ValueError(
                "Yahoo Finance download failed: no portfolio universe price data was retrieved. "
                "Possible reasons: rate limit, temporary Yahoo outage, or invalid tickers."
            )

        self.prices = prices.copy()
        self.returns = compute_returns(self.prices)

        bpx, bmeta = load_price_data(
            tickers=[self.benchmark_symbol],
            start=start,
            end=end,
            auto_adjust=False,
            batch_size=1,
        )

        if bmeta.get("failed"):
            self.diagnostics.add_warning(
                f"Benchmark download issue for {self.benchmark_symbol}: " + ", ".join(bmeta["failed"])
            )

        if bpx is None or bpx.empty or self.benchmark_symbol not in bpx.columns:
            self.diagnostics.add_warning(
                f"Benchmark data for {self.benchmark_symbol} could not be retrieved. "
                "Benchmark-relative metrics may be unavailable."
            )
            self.benchmark_prices = pd.DataFrame()
            self.benchmark_returns = pd.Series(dtype=float)
        else:
            self.benchmark_prices = bpx[[self.benchmark_symbol]].copy()
            benchmark_returns_df = compute_returns(self.benchmark_prices)
            if benchmark_returns_df.empty:
                self.benchmark_returns = pd.Series(dtype=float)
            else:
                self.benchmark_returns = benchmark_returns_df.iloc[:, 0]

        _validate_input_frame("Price data", self.prices)
        _validate_input_frame("Return matrix", self.returns)

        min_obs = int(getattr(self.config, "min_observations", 60))
        if self.returns.shape[0] < min_obs:
            raise ValueError(
                f"Return matrix has only {self.returns.shape[0]} rows, below the required minimum "
                f"observations of {min_obs}."
            )

        if not self.benchmark_returns.empty:
            aligned = pd.concat(
                [self.returns, self.benchmark_returns.rename("benchmark")],
                axis=1,
                join="inner",
            ).dropna()

            if aligned.empty or aligned.shape[0] < 30:
                self.diagnostics.add_warning(
                    "Benchmark series could not be aligned with portfolio returns; "
                    "benchmark-relative metrics may be partial."
                )
                self.benchmark_returns = pd.Series(dtype=float)
            else:
                self.returns = aligned.drop(columns=["benchmark"])
                self.benchmark_returns = aligned["benchmark"]

    # -----------------------------------------------------
    # Inputs
    # -----------------------------------------------------
    def _estimate_inputs(self) -> Tuple[pd.Series, pd.DataFrame]:
        ann = _annualization_factor()
        mu = self.returns.mean() * ann
        cov = self.returns.cov() * ann

        cov_values = cov.values
        eigvals, eigvecs = np.linalg.eigh(cov_values)
        eigvals = np.clip(eigvals, 1e-10, None)
        cov_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        cov = pd.DataFrame(cov_psd, index=cov.index, columns=cov.columns)

        return mu, cov

    def _build_strategy_weights(self, mu: pd.Series, cov: pd.DataFrame) -> Dict[str, np.ndarray]:
        allow_short = bool(getattr(self.config, "allow_short", False))
        rf = float(getattr(self.config, "risk_free_rate", 0.03))

        strategies: Dict[str, np.ndarray] = {
            "equal_weight": _equal_weight(len(mu)),
            "min_variance": _min_variance_weights(cov, allow_short=allow_short),
            "max_sharpe": _max_sharpe_weights(mu, cov, risk_free_rate=rf, allow_short=allow_short),
        }

        if self.bl_controls.get("enabled", False):
            mu_bl = 0.5 * mu + 0.5 * mu.mean()
            strategies["black_litterman_proxy"] = _max_sharpe_weights(
                mu_bl,
                cov,
                risk_free_rate=rf,
                allow_short=allow_short,
            )

        return strategies

    # -----------------------------------------------------
    # Main run
    # -----------------------------------------------------
    def run(self) -> "ProfessionalPortfolioEngine":
        self.metrics = {}
        self.metrics_df = pd.DataFrame()
        self.strategy_df = pd.DataFrame()
        self.stress_table = pd.DataFrame()
        self.factor_pca_df = pd.DataFrame()

        self._load_market_data()

        mu, cov = self._estimate_inputs()
        strategy_weights = self._build_strategy_weights(mu, cov)

        benchmark_returns = self.benchmark_returns.copy()
        if benchmark_returns.empty:
            benchmark_returns = pd.Series(index=self.returns.index, data=np.nan)

        metric_rows: List[Dict[str, Any]] = []
        strategy_rows: List[Dict[str, Any]] = []

        for strategy_name, weights in strategy_weights.items():
            pr = _portfolio_returns(self.returns, weights)

            metrics = _compute_performance_metrics(
                portfolio_returns=pr,
                benchmark_returns=benchmark_returns,
                initial_capital=float(getattr(self.config, "initial_capital", 100000.0)),
                risk_free_rate=float(getattr(self.config, "risk_free_rate", 0.03)),
            )

            metrics["weights"] = pd.Series(weights, index=self.returns.columns, name=strategy_name)

            stress_table = _build_stress_table(
                portfolio_returns=pr,
                benchmark_returns=benchmark_returns,
                selected_family=str(self.scenario_controls.get("selected_family", "All")),
                minimum_severity_threshold=float(self.scenario_controls.get("minimum_severity_threshold", 0.0)),
                quick_view=str(self.scenario_controls.get("quick_view", "All")),
            )
            metrics["stress_table"] = stress_table

            self.metrics[strategy_name] = metrics

            metric_rows.append(
                {
                    "strategy": strategy_name,
                    "annual_return": metrics.get("annual_return"),
                    "annual_return_benchmark": metrics.get("annual_return_benchmark"),
                    "volatility": metrics.get("volatility"),
                    "sharpe_ratio": metrics.get("sharpe_ratio"),
                    "sortino_ratio": metrics.get("sortino_ratio"),
                    "calmar_ratio": metrics.get("calmar_ratio"),
                    "max_drawdown": metrics.get("max_drawdown"),
                    "alpha": metrics.get("alpha"),
                    "beta": metrics.get("beta"),
                    "tracking_error": metrics.get("tracking_error"),
                    "information_ratio": metrics.get("information_ratio"),
                    "win_rate": metrics.get("win_rate"),
                    "profit_factor": metrics.get("profit_factor"),
                    "total_return_pct": metrics.get("total_return_pct"),
                    "total_return_benchmark_pct": metrics.get("total_return_benchmark_pct"),
                    "excess_return_vs_benchmark_pct": metrics.get("excess_return_vs_benchmark_pct"),
                    "VaR_95": metrics.get("VaR_95"),
                    "CVaR_95": metrics.get("CVaR_95"),
                    "VaR_99": metrics.get("VaR_99"),
                    "CVaR_99": metrics.get("CVaR_99"),
                    "final_portfolio_value": metrics.get("final_portfolio_value"),
                }
            )

            weights_s = pd.Series(weights, index=self.returns.columns)
            top_weights = weights_s.sort_values(ascending=False).head(5)

            strategy_rows.append(
                {
                    "strategy": strategy_name,
                    "top_holdings": ", ".join([f"{idx} ({val:.1%})" for idx, val in top_weights.items()]),
                    "n_assets": int((weights_s.abs() > 1e-8).sum()),
                }
            )

        if not metric_rows:
            raise ValueError("No strategy metrics were generated.")

        self.metrics_df = pd.DataFrame(metric_rows).set_index("strategy")
        self.strategy_df = pd.DataFrame(strategy_rows)

        best_name = self.best_strategy_name()
        self.stress_table = self.metrics[best_name].get("stress_table", pd.DataFrame())
        self.factor_pca_df = _build_factor_pca_df(self.returns)

        if self.returns.shape[1] < 3:
            self.diagnostics.add_warning("Factor PCA is limited because the universe contains fewer than 3 assets.")

        if self.benchmark_returns.empty:
            self.diagnostics.add_warning(
                "Benchmark-relative metrics are partially unavailable because benchmark data is missing."
            )

        self.diagnostics.add_info(
            f"Loaded {self.prices.shape[1]} assets and {self.prices.shape[0]} price observations."
        )
        self.diagnostics.add_info(
            f"Generated {len(self.metrics)} portfolio strategies."
        )

        return self

    # -----------------------------------------------------
    # Public helper
    # -----------------------------------------------------
    def best_strategy_name(self) -> str:
        if self.metrics_df is None or self.metrics_df.empty:
            raise ValueError("Strategy metrics are not available.")

        if "sharpe_ratio" not in self.metrics_df.columns:
            return str(self.metrics_df.index[0])

        ranked = self.metrics_df["sharpe_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
        if ranked.empty:
            return str(self.metrics_df.index[0])

        return str(ranked.idxmax())

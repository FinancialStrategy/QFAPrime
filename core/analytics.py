from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import scipy.stats as stats

from core.config import ProfessionalConfig


class AnalyticsEngine:
    def __init__(self, config: ProfessionalConfig):
        self.config = config

    def portfolio_returns(self, returns: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
        w = pd.Series(weights).reindex(returns.columns).fillna(0.0)
        w = w / w.sum()
        return returns.mul(w, axis=1).sum(axis=1)

    def portfolio_values(self, returns: pd.Series, initial_capital: float) -> pd.Series:
        return (1 + returns).cumprod() * initial_capital

    def calculate_var_family(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> Dict[str, float]:
        aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        aligned.columns = ["portfolio", "benchmark"]
        active = aligned["portfolio"] - aligned["benchmark"]

        out: Dict[str, float] = {}
        for cl in self.config.confidence_levels:
            q_port = np.quantile(aligned["portfolio"], 1 - cl)
            q_act = np.quantile(active, 1 - cl)

            tail_port = aligned["portfolio"][aligned["portfolio"] <= q_port]
            tail_act = active[active <= q_act]

            out[f"var_{int(cl * 100)}"] = float(-q_port)
            out[f"cvar_{int(cl * 100)}"] = float(-tail_port.mean()) if len(tail_port) else np.nan
            out[f"relative_var_{int(cl * 100)}"] = float(-q_act)
            out[f"relative_cvar_{int(cl * 100)}"] = float(-tail_act.mean()) if len(tail_act) else np.nan

        return out

    def calculate_all_metrics(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        initial_capital: float,
    ) -> Dict[str, Any]:
        aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        aligned.columns = ["portfolio", "benchmark"]

        portfolio_values = self.portfolio_values(aligned["portfolio"], initial_capital)
        benchmark_values = self.portfolio_values(aligned["benchmark"], initial_capital)

        final_portfolio_value = float(portfolio_values.iloc[-1])
        final_benchmark_value = float(benchmark_values.iloc[-1])

        total_return_portfolio = final_portfolio_value / initial_capital - 1
        total_return_benchmark = final_benchmark_value / initial_capital - 1

        n_years = len(aligned) / self.config.annual_trading_days
        annual_return_portfolio = (1 + total_return_portfolio) ** (1 / n_years) - 1 if n_years > 0 else 0.0
        annual_return_benchmark = (1 + total_return_benchmark) ** (1 / n_years) - 1 if n_years > 0 else 0.0

        daily_rf = self.config.risk_free_rate / self.config.annual_trading_days
        excess_returns = aligned["portfolio"] - daily_rf

        volatility = float(aligned["portfolio"].std() * np.sqrt(self.config.annual_trading_days))
        sharpe = (
            float((excess_returns.mean() / aligned["portfolio"].std()) * np.sqrt(self.config.annual_trading_days))
            if aligned["portfolio"].std() > 0
            else 0.0
        )

        downside = aligned["portfolio"][aligned["portfolio"] < 0]
        downside_vol = float(downside.std() * np.sqrt(self.config.annual_trading_days)) if len(downside) > 1 else np.nan
        sortino = (
            float((excess_returns.mean() * self.config.annual_trading_days) / downside_vol)
            if pd.notna(downside_vol) and downside_vol > 0
            else np.nan
        )

        cumulative = (1 + aligned["portfolio"]).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = cumulative / rolling_max - 1
        max_drawdown = float(drawdown.min())

        calmar = float(annual_return_portfolio / abs(max_drawdown)) if max_drawdown < 0 else np.nan

        covariance = aligned["portfolio"].cov(aligned["benchmark"])
        benchmark_variance = aligned["benchmark"].var()
        beta = float(covariance / benchmark_variance) if benchmark_variance > 0 else 1.0
        alpha = float(
            annual_return_portfolio
            - (self.config.risk_free_rate + beta * (annual_return_benchmark - self.config.risk_free_rate))
        )

        tracking_diff = aligned["portfolio"] - aligned["benchmark"]
        tracking_error = float(tracking_diff.std() * np.sqrt(self.config.annual_trading_days))
        information_ratio = (
            float((tracking_diff.mean() * self.config.annual_trading_days) / tracking_error)
            if tracking_error > 0
            else 0.0
        )

        win_rate = float((aligned["portfolio"] > 0).mean())
        gross_gain = aligned["portfolio"][aligned["portfolio"] > 0].sum()
        gross_loss = abs(aligned["portfolio"][aligned["portfolio"] < 0].sum())
        profit_factor = float(gross_gain / gross_loss) if gross_loss > 0 else np.nan

        var_family = self.calculate_var_family(aligned["portfolio"], aligned["benchmark"])

        return {
            "initial_capital": initial_capital,
            "final_portfolio_value": final_portfolio_value,
            "final_benchmark_value": final_benchmark_value,
            "total_profit_loss": final_portfolio_value - initial_capital,
            "total_return_pct": total_return_portfolio,
            "total_return_benchmark_pct": total_return_benchmark,
            "excess_return_vs_benchmark_pct": total_return_portfolio - total_return_benchmark,
            "annual_return": annual_return_portfolio,
            "annual_return_benchmark": annual_return_benchmark,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "max_drawdown": max_drawdown,
            "alpha": alpha,
            "beta": beta,
            "tracking_error": tracking_error,
            "information_ratio": information_ratio,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "skewness": float(stats.skew(aligned["portfolio"], nan_policy="omit")),
            "kurtosis": float(stats.kurtosis(aligned["portfolio"], nan_policy="omit", fisher=False)),
            "portfolio_values": portfolio_values,
            "benchmark_values": benchmark_values,
            "drawdown_series": drawdown,
            "portfolio_returns": aligned["portfolio"],
            "benchmark_returns": aligned["benchmark"],
            **var_family,
        }

    def rolling_tracking_error(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        window: int = 63,
        min_periods: int = 30,
    ) -> pd.Series:
        aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        aligned.columns = ["portfolio", "benchmark"]
        return (
            (aligned["portfolio"] - aligned["benchmark"])
            .rolling(window, min_periods=min_periods)
            .std()
            * np.sqrt(self.config.annual_trading_days)
        )

    def risk_contribution_table(
        self,
        weights: Dict[str, float],
        cov: pd.DataFrame,
    ) -> pd.DataFrame:
        w = pd.Series(weights).reindex(cov.index).fillna(0.0)
        sigma = cov.values
        port_var = float(w.values @ sigma @ w.values)
        if port_var <= 0:
            return pd.DataFrame()

        marginal = sigma @ w.values
        total_contrib = w.values * marginal / np.sqrt(port_var)
        pct_contrib = total_contrib / np.sum(total_contrib)

        return pd.DataFrame({
            "asset": cov.index,
            "weight": w.values,
            "marginal_risk": marginal,
            "risk_contribution": total_contrib,
            "pct_risk_contribution": pct_contrib,
        }).sort_values("pct_risk_contribution", ascending=False)

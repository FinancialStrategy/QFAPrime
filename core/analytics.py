from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

    def rolling_sharpe(
        self,
        portfolio_returns: pd.Series,
        window: int = 63,
        min_periods: int = 30,
    ) -> pd.Series:
        rf_daily = self.config.risk_free_rate / self.config.annual_trading_days
        excess = portfolio_returns - rf_daily

        rolling_mean = excess.rolling(window, min_periods=min_periods).mean()
        rolling_std = portfolio_returns.rolling(window, min_periods=min_periods).std()

        sharpe = (rolling_mean / rolling_std) * np.sqrt(self.config.annual_trading_days)
        return sharpe.replace([np.inf, -np.inf], np.nan)

    def rolling_beta(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        window: int = 63,
        min_periods: int = 30,
    ) -> pd.Series:
        aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        aligned.columns = ["portfolio", "benchmark"]

        cov = aligned["portfolio"].rolling(window, min_periods=min_periods).cov(aligned["benchmark"])
        var_b = aligned["benchmark"].rolling(window, min_periods=min_periods).var()
        beta = cov / var_b.replace(0, np.nan)
        return beta.replace([np.inf, -np.inf], np.nan)

    def rolling_information_ratio(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        window: int = 63,
        min_periods: int = 30,
    ) -> pd.Series:
        aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        aligned.columns = ["portfolio", "benchmark"]

        active = aligned["portfolio"] - aligned["benchmark"]
        active_mean = active.rolling(window, min_periods=min_periods).mean()
        active_std = active.rolling(window, min_periods=min_periods).std()

        ir = (active_mean / active_std) * np.sqrt(self.config.annual_trading_days)
        return ir.replace([np.inf, -np.inf], np.nan)

    def rolling_tracking_error(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        window: int = 63,
        min_periods: int = 30,
    ) -> pd.Series:
        aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        aligned.columns = ["portfolio", "benchmark"]

        te = (
            (aligned["portfolio"] - aligned["benchmark"])
            .rolling(window, min_periods=min_periods)
            .std()
            * np.sqrt(self.config.annual_trading_days)
        )
        return te.replace([np.inf, -np.inf], np.nan)

    def drawdown_series(self, returns: pd.Series) -> pd.Series:
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.cummax()
        return cumulative / rolling_max - 1

    def relative_drawdown_series(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> pd.DataFrame:
        aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        aligned.columns = ["portfolio", "benchmark"]

        p_dd = self.drawdown_series(aligned["portfolio"])
        b_dd = self.drawdown_series(aligned["benchmark"])
        rel_dd = p_dd - b_dd

        return pd.DataFrame({
            "portfolio_drawdown": p_dd,
            "benchmark_drawdown": b_dd,
            "relative_drawdown": rel_dd,
        })

    def active_return_series(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> pd.Series:
        aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        aligned.columns = ["portfolio", "benchmark"]
        return aligned["portfolio"] - aligned["benchmark"]

    def active_risk_contribution_by_region(
        self,
        asset_returns: pd.DataFrame,
        weights: Dict[str, float],
        benchmark_ticker: str,
        asset_metadata: pd.DataFrame,
    ) -> pd.DataFrame:
        if asset_returns.empty:
            return pd.DataFrame()

        assets = list(asset_returns.columns)
        w = pd.Series(weights).reindex(assets).fillna(0.0)

        b = pd.Series(0.0, index=assets)
        if benchmark_ticker in b.index:
            b[benchmark_ticker] = 1.0
        elif "SPY" in b.index:
            b["SPY"] = 1.0
        else:
            b[:] = 1 / len(b)

        active_w = w - b
        cov = asset_returns.cov() * self.config.annual_trading_days

        port_var = float(active_w.values @ cov.values @ active_w.values)
        if port_var <= 0:
            return pd.DataFrame()

        marginal = cov.values @ active_w.values
        contrib = active_w.values * marginal / np.sqrt(port_var)

        contrib_df = pd.DataFrame({
            "ticker": assets,
            "active_weight": active_w.values,
            "active_risk_contribution": contrib,
        })

        meta = asset_metadata[["ticker", "region_type"]].drop_duplicates().copy()
        merged = contrib_df.merge(meta, on="ticker", how="left")
        merged["region_type"] = merged["region_type"].fillna("Unknown")

        region_df = merged.groupby("region_type", as_index=False).agg(
            active_weight=("active_weight", "sum"),
            active_risk_contribution=("active_risk_contribution", "sum"),
        )

        total_arc = region_df["active_risk_contribution"].sum()
        if total_arc != 0:
            region_df["pct_active_risk_contribution"] = region_df["active_risk_contribution"] / total_arc
        else:
            region_df["pct_active_risk_contribution"] = np.nan

        region_df = region_df.sort_values("pct_active_risk_contribution", ascending=False)
        return region_df

    def pca_factor_analysis(
        self,
        returns: pd.DataFrame,
        n_components: int = 5,
    ) -> Dict[str, Any]:
        clean = returns.dropna(axis=1, how="any").copy()
        if clean.shape[1] < 2 or clean.shape[0] < 20:
            return {
                "explained_variance_ratio": pd.Series(dtype=float),
                "loadings": pd.DataFrame(),
                "scores": pd.DataFrame(),
                "interpretation": {},
            }

        n_components = min(n_components, clean.shape[1])

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(clean.values)

        pca = PCA(n_components=n_components)
        scores = pca.fit_transform(x_scaled)

        pcs = [f"PC{i+1}" for i in range(n_components)]
        loadings = pd.DataFrame(
            pca.components_.T,
            index=clean.columns,
            columns=pcs,
        )
        scores_df = pd.DataFrame(
            scores,
            index=clean.index,
            columns=pcs,
        )
        evr = pd.Series(pca.explained_variance_ratio_, index=pcs, name="explained_variance_ratio")

        interpretation = {}
        for pc in pcs[:3]:
            col = loadings[pc]
            same_sign_ratio = max((col > 0).mean(), (col < 0).mean())

            if same_sign_ratio > 0.75:
                interpretation[pc] = "Broad market factor: most assets move in the same direction."
            else:
                top_abs = col.abs().sort_values(ascending=False).head(5).index.tolist()
                interpretation[pc] = f"Differentiation factor: strongest loadings include {', '.join(top_abs)}."

        return {
            "explained_variance_ratio": evr,
            "loadings": loadings,
            "scores": scores_df,
            "interpretation": interpretation,
        }

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

        drawdown = self.drawdown_series(aligned["portfolio"])
        benchmark_drawdown = self.drawdown_series(aligned["benchmark"])
        relative_drawdown = drawdown - benchmark_drawdown
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
            "benchmark_drawdown_series": benchmark_drawdown,
            "relative_drawdown_series": relative_drawdown,
            "portfolio_returns": aligned["portfolio"],
            "benchmark_returns": aligned["benchmark"],
            "active_return_series": tracking_diff,
            **var_family,
        }

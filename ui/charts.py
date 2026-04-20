from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.config import ProfessionalConfig
from core.optimizers import StrategyResult


class StreamlitChartBuilder:
    def __init__(self, config: ProfessionalConfig):
        self.config = config

    def _base_layout(self) -> dict:
        return dict(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(size=11, color="#273444"),
            title_font=dict(size=15, color="#1f2d3d"),
            legend=dict(font=dict(size=10), orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(l=55, r=35, t=55, b=45),
            height=480,
        )

    def info_hub_table(self, asset_metadata: pd.DataFrame) -> go.Figure:
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(asset_metadata.columns),
                fill_color="#2d3e50",
                font=dict(color="white", size=10),
                align="left",
                height=26
            ),
            cells=dict(
                values=[asset_metadata[c] for c in asset_metadata.columns],
                fill_color="white",
                font=dict(color="#273444", size=10),
                align="left",
                height=24
            )
        )])
        fig.update_layout(
            title="Investment Universe Identity Map",
            font=dict(size=10),
            margin=dict(l=20, r=20, t=45, b=20),
            height=max(460, 26 * len(asset_metadata) + 120),
        )
        return fig

    def equity_curve_chart(
        self,
        portfolio_values: pd.Series,
        benchmark_values: pd.Series,
        strategy_name: str,
    ) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=portfolio_values.index,
            y=portfolio_values.values,
            mode="lines",
            name=f"{strategy_name} Portfolio",
            line=dict(color="#425b76", width=2.0),
            hovertemplate="Date: %{x}<br>Portfolio: $%{y:,.0f}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=benchmark_values.index,
            y=benchmark_values.values,
            mode="lines",
            name=f"Benchmark ({self.config.benchmark})",
            line=dict(color="#8c99a5", width=1.8, dash="dash"),
            hovertemplate="Date: %{x}<br>Benchmark: $%{y:,.0f}<extra></extra>"
        ))
        fig.update_layout(
            title="Portfolio vs Benchmark Equity Curve",
            yaxis_tickprefix="$",
            yaxis_tickformat=",.0f",
            **self._base_layout()
        )
        return fig

    def drawdown_chart(self, drawdown_series: pd.Series, strategy_name: str) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=drawdown_series.index,
            y=drawdown_series.values,
            mode="lines",
            fill="tozeroy",
            name=f"{strategy_name} Drawdown",
            line=dict(color="#6e7781", width=1.8),
            fillcolor="rgba(110,119,129,0.18)"
        ))
        fig.update_layout(
            title="Drawdown Analysis",
            yaxis_tickformat=".0%",
            **self._base_layout()
        )
        return fig

    def performance_dashboard(self, metrics_df: pd.DataFrame) -> go.Figure:
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("Annual Return", "Sharpe Ratio", "Max Drawdown", "Final Portfolio Value")
        )

        fig.add_trace(go.Bar(
            x=metrics_df.index,
            y=metrics_df["annual_return"],
            text=[f"{v:.2%}" for v in metrics_df["annual_return"]],
            textposition="outside",
            marker_color="#596d5f"
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            x=metrics_df.index,
            y=metrics_df["sharpe_ratio"],
            text=[f"{v:.2f}" for v in metrics_df["sharpe_ratio"]],
            textposition="outside",
            marker_color="#5e7286"
        ), row=1, col=2)

        fig.add_trace(go.Bar(
            x=metrics_df.index,
            y=metrics_df["max_drawdown"],
            text=[f"{v:.2%}" for v in metrics_df["max_drawdown"]],
            textposition="outside",
            marker_color="#7a6b6b"
        ), row=2, col=1)

        fig.add_trace(go.Bar(
            x=metrics_df.index,
            y=metrics_df["final_portfolio_value"] / 1e6,
            text=[f"${v/1e6:.1f}M" for v in metrics_df["final_portfolio_value"]],
            textposition="outside",
            marker_color="#7d8074"
        ), row=2, col=2)

        fig.update_layout(
            title="Executive Strategy Dashboard",
            showlegend=False,
            height=680,
            **self._base_layout()
        )
        return fig

    def optimization_chart(
        self,
        mu: pd.Series,
        cov: pd.DataFrame,
        strategies: Dict[str, StrategyResult],
        risk_free_rate: float
    ) -> go.Figure:
        assets = list(mu.index)
        rng = np.random.default_rng(42)

        vols = []
        rets = []
        for _ in range(900):
            w = rng.random(len(assets))
            w /= w.sum()
            vols.append(float(np.sqrt(w @ cov.values @ w)))
            rets.append(float(np.dot(w, mu.values)))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=vols,
            y=rets,
            mode="markers",
            name="Feasible Portfolios",
            opacity=0.22,
            marker=dict(color="#8d99a6", size=5)
        ))

        for name, result in strategies.items():
            w = pd.Series(result.weights).reindex(assets).fillna(0.0).values
            v = float(np.sqrt(w @ cov.values @ w))
            r = float(np.dot(w, mu.values))
            fig.add_trace(go.Scatter(
                x=[v],
                y=[r],
                mode="markers+text",
                text=[name],
                textposition="top center",
                name=name,
                marker=dict(size=9, color="#4d6175")
            ))

        max_sharpe_point = None
        try:
            from pypfopt import EfficientFrontier
            ef = EfficientFrontier(mu, cov)
            ef.max_sharpe(risk_free_rate=risk_free_rate)
            w = np.array(list(ef.clean_weights().values()))
            w = w / w.sum()
            max_sharpe_point = (
                float(np.sqrt(w @ cov.values @ w)),
                float(np.dot(w, mu.values))
            )
        except Exception:
            pass

        if max_sharpe_point is not None and max_sharpe_point[0] > 0:
            x = np.linspace(0, max(vols) * 1.15, 50)
            sr = (max_sharpe_point[1] - risk_free_rate) / max_sharpe_point[0]
            y = risk_free_rate + sr * x
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name="Capital Market Line",
                line=dict(color="#6f7c88", dash="dash")
            ))

        fig.update_layout(
            title="Portfolio Optimization Chart and Efficient Frontier Market Line",
            xaxis_title="Volatility",
            yaxis_title="Expected Return",
            **self._base_layout()
        )
        fig.update_xaxes(tickformat=".0%")
        fig.update_yaxes(tickformat=".0%")
        return fig

    def relative_frontier_chart(
        self,
        mu: pd.Series,
        cov: pd.DataFrame,
        strategies: Dict[str, StrategyResult],
        benchmark_proxy: pd.Series
    ) -> go.Figure:
        assets = list(mu.index)
        b = benchmark_proxy.reindex(assets).fillna(0.0).values
        rng = np.random.default_rng(42)

        xs = []
        ys = []
        for _ in range(700):
            w = rng.random(len(assets))
            w /= w.sum()
            active_vol = float(np.sqrt(max((w - b) @ cov.values @ (w - b), 0.0)))
            excess_ret = float(np.dot(w - b, mu.values))
            xs.append(active_vol)
            ys.append(excess_ret)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            name="Feasible Relative Portfolios",
            opacity=0.28,
            marker=dict(color="#98a3ad", size=5)
        ))

        for name, result in strategies.items():
            w = pd.Series(result.weights).reindex(assets).fillna(0.0).values
            active_vol = float(np.sqrt(max((w - b) @ cov.values @ (w - b), 0.0)))
            excess_ret = float(np.dot(w - b, mu.values))
            fig.add_trace(go.Scatter(
                x=[active_vol],
                y=[excess_ret],
                mode="markers+text",
                text=[name],
                textposition="top center",
                name=name,
                marker=dict(size=9, color="#55697c")
            ))

        fig.add_hline(y=0, line_color="#b0b8bf", line_width=1)
        fig.add_vline(x=0, line_color="#b0b8bf", line_width=1)

        fig.update_layout(
            title="Benchmark-Relative Efficient Frontier",
            xaxis_title="Active Risk (Tracking Error Proxy)",
            yaxis_title="Expected Excess Return vs Benchmark",
            **self._base_layout()
        )
        fig.update_xaxes(tickformat=".0%")
        fig.update_yaxes(tickformat=".0%")
        return fig

    def tracking_error_chart(self, metrics_df: pd.DataFrame) -> go.Figure:
        fig = go.Figure([
            go.Bar(
                x=metrics_df.index,
                y=metrics_df["tracking_error"],
                text=[f"{v:.2%}" for v in metrics_df["tracking_error"]],
                textposition="auto",
                marker_color="#6c786f"
            )
        ])
        fig.update_layout(
            title="Tracking Error by Strategy",
            yaxis_tickformat=".0%",
            **self._base_layout()
        )
        return fig

    def benchmark_vs_tracking_error_curve(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        strategy_name: str = "Tracking Error Optimal"
    ) -> go.Figure:
        aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        aligned.columns = ["portfolio", "benchmark"]

        if aligned.empty:
            fig = go.Figure()
            fig.add_annotation(text="No aligned data available.", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(title="Benchmark vs Tracking Error Optimal Dynamic Curve", **self._base_layout())
            return fig

        cum_portfolio = (1 + aligned["portfolio"]).cumprod() - 1
        cum_benchmark = (1 + aligned["benchmark"]).cumprod() - 1
        rolling_te_63d = (
            (aligned["portfolio"] - aligned["benchmark"])
            .rolling(63, min_periods=30)
            .std()
            * np.sqrt(self.config.annual_trading_days)
        )

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=cum_portfolio.index,
            y=cum_portfolio.values,
            mode="lines",
            name=f"{strategy_name} Cum Return",
            line=dict(color="#425b76", width=2.0),
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=cum_benchmark.index,
            y=cum_benchmark.values,
            mode="lines",
            name=f"Benchmark ({self.config.benchmark}) Cum Return",
            line=dict(color="#8c99a5", width=1.8, dash="dash"),
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=rolling_te_63d.index,
            y=rolling_te_63d.values,
            mode="lines",
            name="63D Rolling Tracking Error",
            line=dict(color="#596d5f", width=1.8, dash="dot"),
        ), secondary_y=True)

        fig.update_layout(
            title="Benchmark vs Tracking Error Optimal Dynamic Curve",
            hovermode="x unified",
            **self._base_layout()
        )
        fig.update_yaxes(title_text="Cumulative Return", tickformat=".0%", secondary_y=False)
        fig.update_yaxes(title_text="63D Rolling TE", tickformat=".0%", secondary_y=True)
        return fig

    def stress_test_chart(self, stress_df: pd.DataFrame) -> go.Figure:
        if stress_df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No stress scenarios available in current sample", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(title="Historical Stress Testing", **self._base_layout())
            return fig

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=stress_df["scenario"],
            y=stress_df["portfolio_return"],
            name="Portfolio",
            text=[f"{v:.2%}" for v in stress_df["portfolio_return"]],
            textposition="auto",
            marker_color="#5f7286"
        ))
        fig.add_trace(go.Bar(
            x=stress_df["scenario"],
            y=stress_df["benchmark_return"],
            name="Benchmark",
            text=[f"{v:.2%}" for v in stress_df["benchmark_return"]],
            textposition="auto",
            marker_color="#939ca5"
        ))
        fig.add_trace(go.Scatter(
            x=stress_df["scenario"],
            y=stress_df["relative_return"],
            name="Relative Return",
            mode="lines+markers",
            yaxis="y2",
            line=dict(color="#6c786f")
        ))
        fig.update_layout(
            title="Historical Stress Testing",
            barmode="group",
            yaxis_tickformat=".0%",
            yaxis2=dict(overlaying="y", side="right", tickformat=".0%"),
            **self._base_layout()
        )
        return fig

    def var_family_chart(self, metrics_df: pd.DataFrame, kind: str = "absolute") -> go.Figure:
        cols = ["var_95", "cvar_95"] if kind == "absolute" else ["relative_var_95", "relative_cvar_95"]
        title = "VaR / CVaR Figures" if kind == "absolute" else "Relative VaR Figures"

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=metrics_df.index,
            y=metrics_df[cols[0]],
            name=cols[0],
            text=[f"{v:.2%}" for v in metrics_df[cols[0]]],
            textposition="auto",
            marker_color="#6b7783"
        ))
        fig.add_trace(go.Bar(
            x=metrics_df.index,
            y=metrics_df[cols[1]],
            name=cols[1],
            text=[f"{v:.2%}" for v in metrics_df[cols[1]]],
            textposition="auto",
            marker_color="#98a3ad"
        ))
        fig.update_layout(
            title=title,
            barmode="group",
            yaxis_tickformat=".0%",
            **self._base_layout()
        )
        return fig

    def allocation_chart(self, weights: Dict[str, float]) -> go.Figure:
        s = pd.Series(weights).sort_values(ascending=False)
        fig = go.Figure([
            go.Bar(
                x=s.index[:12],
                y=s.values[:12],
                text=[f"{v:.1%}" for v in s.values[:12]],
                textposition="auto",
                marker_color="#5e7286"
            )
        ])
        fig.update_layout(
            title="Top Strategy Allocation",
            yaxis_tickformat=".0%",
            **self._base_layout()
        )
        return fig

    def monte_carlo_terminal_distribution(self, mc_result: Dict) -> go.Figure:
        terminal = pd.Series(mc_result["terminal_values"])
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=terminal,
            nbinsx=40,
            marker_color="#738292",
            opacity=0.85,
            name="Terminal Value Distribution"
        ))
        fig.update_layout(
            title="Monte Carlo Terminal Value Distribution",
            xaxis_title="Terminal Portfolio Value",
            yaxis_title="Frequency",
            **self._base_layout()
        )
        fig.update_xaxes(tickprefix="$", tickformat=",.0f")
        return fig

    def monte_carlo_paths_chart(self, mc_result: Dict, n_paths: int = 50) -> go.Figure:
        paths = mc_result["paths"]
        fig = go.Figure()
        max_paths = min(n_paths, paths.shape[1])

        for i in range(max_paths):
            fig.add_trace(go.Scatter(
                x=list(range(paths.shape[0])),
                y=paths[:, i],
                mode="lines",
                line=dict(width=1),
                opacity=0.20,
                showlegend=False,
                hoverinfo="skip"
            ))

        fig.update_layout(
            title="Monte Carlo Simulated Paths",
            xaxis_title="Day",
            yaxis_title="Portfolio Value",
            **self._base_layout()
        )
        fig.update_yaxes(tickprefix="$", tickformat=",.0f")
        return fig

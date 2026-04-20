from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.config import ProfessionalConfig
from core.engine import ProfessionalPortfolioEngine
from core.universes import UNIVERSE_REGISTRY


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Professional Portfolio Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# STYLING
# =========================================================
CUSTOM_CSS = """
<style>
    .main > div {
        padding-top: 1.0rem;
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 96rem;
    }

    .kpi-card {
        background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 16px;
        padding: 18px 18px 14px 18px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.18);
        min-height: 120px;
    }

    .kpi-title {
        font-size: 0.84rem;
        color: #cbd5e1;
        margin-bottom: 10px;
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    .kpi-value {
        font-size: 1.6rem;
        color: white;
        font-weight: 800;
        line-height: 1.15;
    }

    .kpi-sub {
        font-size: 0.82rem;
        color: #94a3b8;
        margin-top: 8px;
    }

    .section-title {
        font-size: 1.1rem;
        font-weight: 800;
        margin: 0.5rem 0 0.9rem 0;
        color: #0f172a;
    }

    .small-note {
        color: #64748b;
        font-size: 0.82rem;
    }

    .hero-wrap {
        padding: 0.4rem 0 0.8rem 0;
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 900;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        font-size: 0.96rem;
        color: #475569;
        margin-bottom: 0.1rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =========================================================
# HELPERS
# =========================================================
def fmt_pct(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"{x:.2%}"


def fmt_num(x: Optional[float], decimals: int = 2) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"{x:,.{decimals}f}"


def fmt_usd(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"${x:,.0f}"


def safe_series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x.dropna()
    return pd.Series(dtype=float)


def safe_df(x) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    return pd.DataFrame()


def render_kpi_card(title: str, value: str, sub: str = "") -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def cumulative_curve(returns: pd.Series) -> pd.Series:
    s = safe_series(returns)
    if s.empty:
        return s
    return (1 + s).cumprod() - 1


def make_line_chart(
    df: pd.DataFrame,
    title: str,
    yaxis_title: str = "",
    height: int = 520,
) -> go.Figure:
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                mode="lines",
                name=str(col),
            )
        )
    fig.update_layout(
        title=title,
        height=height,
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        xaxis_title="Date",
        yaxis_title=yaxis_title,
    )
    return fig


def make_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    color: Optional[str] = None,
    height: int = 520,
) -> go.Figure:
    fig = px.bar(
        df,
        x=x,
        y=y,
        color=color,
        title=title,
    )
    fig.update_layout(
        height=height,
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title=x,
        yaxis_title=y,
        showlegend=bool(color),
    )
    return fig


def build_strategy_return_frame(
    engine: ProfessionalPortfolioEngine,
) -> pd.DataFrame:
    curves = {}
    for strategy_name, metrics in engine.metrics.items():
        pr = safe_series(metrics.get("portfolio_returns"))
        if not pr.empty:
            curves[strategy_name] = cumulative_curve(pr)
    if not curves:
        return pd.DataFrame()
    return pd.DataFrame(curves)


def build_strategy_drawdown_frame(
    engine: ProfessionalPortfolioEngine,
) -> pd.DataFrame:
    dds = {}
    for strategy_name, metrics in engine.metrics.items():
        dd = safe_series(metrics.get("drawdown_series"))
        if not dd.empty:
            dds[strategy_name] = dd
    if not dds:
        return pd.DataFrame()
    return pd.DataFrame(dds)


def metric_columns_for_display() -> List[str]:
    return [
        "annual_return",
        "annual_return_benchmark",
        "volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "max_drawdown",
        "alpha",
        "beta",
        "tracking_error",
        "information_ratio",
        "win_rate",
        "profit_factor",
        "total_return_pct",
        "total_return_benchmark_pct",
        "excess_return_vs_benchmark_pct",
    ]


def format_metrics_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    pct_cols = [
        "annual_return",
        "annual_return_benchmark",
        "volatility",
        "max_drawdown",
        "tracking_error",
        "win_rate",
        "total_return_pct",
        "total_return_benchmark_pct",
        "excess_return_vs_benchmark_pct",
        "alpha",
    ]

    for c in pct_cols:
        if c in out.columns:
            out[c] = out[c].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")

    num_cols = [
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "beta",
        "information_ratio",
        "profit_factor",
    ]
    for c in num_cols:
        if c in out.columns:
            out[c] = out[c].map(lambda x: f"{x:,.3f}" if pd.notna(x) else "N/A")

    return out


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## Portfolio Gate")

    universe_options = list(UNIVERSE_REGISTRY.keys())
    selected_universe = st.selectbox(
        "Investment Universe",
        options=universe_options,
        index=universe_options.index("institutional_multi_asset")
        if "institutional_multi_asset" in universe_options
        else 0,
    )

    benchmark_symbol = st.text_input("Benchmark Symbol", value="^GSPC")
    start_date = st.text_input("Start Date", value="2019-01-01")

    initial_capital = st.number_input(
        "Initial Capital",
        min_value=1000.0,
        value=100000.0,
        step=1000.0,
    )

    risk_free_rate = st.number_input(
        "Risk-Free Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.03,
        step=0.005,
        format="%.3f",
    )

    use_log_returns = st.checkbox("Use Log Returns", value=False)

    st.markdown("---")
    st.markdown("### Black-Litterman")
    bl_enabled = st.checkbox("Enable Black-Litterman Proxy", value=False)

    st.markdown("---")
    st.markdown("### Stress Filters")
    selected_family = st.selectbox(
        "Scenario Family",
        options=["All", "Crisis", "Inflation", "Banking_Stress", "Sharp_Rally", "Sharp_Selloff"],
        index=0,
    )
    minimum_severity_threshold = st.slider(
        "Minimum Severity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
    )
    quick_view = st.selectbox(
        "Quick View",
        options=[
            "All",
            "Crisis Only",
            "Inflation Only",
            "Banking Stress Only",
            "Sharp Rally Only",
            "Sharp Selloff Only",
        ],
        index=0,
    )

    st.markdown("---")
    run_button = st.button("Run Professional Analytics", type="primary", use_container_width=True)


# =========================================================
# HERO
# =========================================================
st.markdown(
    """
    <div class="hero-wrap">
        <div class="hero-title">Professional Portfolio Analytics</div>
        <div class="hero-subtitle">
            Institutional multi-asset portfolio diagnostics, strategy comparison, stress testing, and factor analysis.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# ENGINE EXECUTION
# =========================================================
if "engine_result" not in st.session_state:
    st.session_state.engine_result = None
    st.session_state.engine_error = None

if run_button or st.session_state.engine_result is None:
    try:
        config = ProfessionalConfig(
            benchmark_symbol=benchmark_symbol,
            default_start_date=start_date,
            initial_capital=float(initial_capital),
            risk_free_rate=float(risk_free_rate),
            use_log_returns=bool(use_log_returns),
            selected_universe=selected_universe,
        )

        engine = ProfessionalPortfolioEngine(
            config=config,
            bl_controls={
                "enabled": bl_enabled,
                "view_mode": "ticker",
                "views_payload": [],
            },
            scenario_controls={
                "selected_family": selected_family,
                "minimum_severity_threshold": float(minimum_severity_threshold),
                "quick_view": quick_view,
            },
        )
        engine.run()

        st.session_state.engine_result = engine
        st.session_state.engine_error = None

    except Exception as exc:
        st.session_state.engine_result = None
        st.session_state.engine_error = str(exc)

engine: Optional[ProfessionalPortfolioEngine] = st.session_state.engine_result
engine_error = st.session_state.engine_error

if engine_error:
    st.error(f"Application error: {engine_error}")
    st.stop()

if engine is None:
    st.warning("No analysis output is available.")
    st.stop()


# =========================================================
# DIAGNOSTICS
# =========================================================
diag = engine.diagnostics.summary()
if diag.get("warnings"):
    with st.expander("Diagnostics Warnings", expanded=False):
        for w in diag["warnings"]:
            st.warning(str(w))

if diag.get("errors"):
    with st.expander("Diagnostics Errors", expanded=False):
        for e in diag["errors"]:
            st.error(str(e))


# =========================================================
# TOP KPIs
# =========================================================
best_name = engine.best_strategy_name()
best_metrics = engine.metrics.get(best_name, {})

col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    render_kpi_card("Best Strategy", best_name, "Top rank by Sharpe ratio")
with col2:
    render_kpi_card("Annual Return", fmt_pct(best_metrics.get("annual_return")), "Annualized")
with col3:
    render_kpi_card("Volatility", fmt_pct(best_metrics.get("volatility")), "Annualized")
with col4:
    render_kpi_card("Sharpe Ratio", fmt_num(best_metrics.get("sharpe_ratio"), 3), "Risk-adjusted return")
with col5:
    render_kpi_card("Max Drawdown", fmt_pct(best_metrics.get("max_drawdown")), "Peak-to-trough")
with col6:
    render_kpi_card("Final Portfolio Value", fmt_usd(best_metrics.get("final_portfolio_value")), "Based on initial capital")

st.markdown("")


# =========================================================
# TABS
# =========================================================
tab_overview, tab_strategies, tab_risk, tab_stress, tab_factors, tab_data = st.tabs(
    [
        "Overview",
        "Strategy Comparison",
        "Risk Analytics",
        "Stress Testing",
        "Factor PCA",
        "Data & Diagnostics",
    ]
)


# =========================================================
# OVERVIEW
# =========================================================
with tab_overview:
    st.markdown("### Executive Summary")

    best_portfolio_returns = safe_series(best_metrics.get("portfolio_returns"))
    best_benchmark_returns = safe_series(best_metrics.get("benchmark_returns"))
    best_drawdown = safe_series(best_metrics.get("drawdown_series"))
    best_benchmark_drawdown = safe_series(best_metrics.get("benchmark_drawdown_series"))

    curve_df = pd.DataFrame()
    if not best_portfolio_returns.empty:
        curve_df["Portfolio"] = cumulative_curve(best_portfolio_returns)
    if not best_benchmark_returns.empty:
        curve_df["Benchmark"] = cumulative_curve(best_benchmark_returns)

    if not curve_df.empty:
        fig = make_line_chart(
            curve_df,
            title=f"Cumulative Return: {best_name} vs Benchmark",
            yaxis_title="Cumulative Return",
            height=560,
        )
        st.plotly_chart(fig, use_container_width=True)

    dd_df = pd.DataFrame()
    if not best_drawdown.empty:
        dd_df["Portfolio Drawdown"] = best_drawdown
    if not best_benchmark_drawdown.empty:
        dd_df["Benchmark Drawdown"] = best_benchmark_drawdown

    if not dd_df.empty:
        fig = make_line_chart(
            dd_df,
            title=f"Drawdown Profile: {best_name} vs Benchmark",
            yaxis_title="Drawdown",
            height=520,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Best Strategy Metrics")
    summary_rows = {
        "Total Return": fmt_pct(best_metrics.get("total_return_pct")),
        "Benchmark Total Return": fmt_pct(best_metrics.get("total_return_benchmark_pct")),
        "Excess Return": fmt_pct(best_metrics.get("excess_return_vs_benchmark_pct")),
        "Annual Return": fmt_pct(best_metrics.get("annual_return")),
        "Annual Benchmark Return": fmt_pct(best_metrics.get("annual_return_benchmark")),
        "Sharpe Ratio": fmt_num(best_metrics.get("sharpe_ratio"), 3),
        "Sortino Ratio": fmt_num(best_metrics.get("sortino_ratio"), 3),
        "Calmar Ratio": fmt_num(best_metrics.get("calmar_ratio"), 3),
        "Beta": fmt_num(best_metrics.get("beta"), 3),
        "Alpha": fmt_pct(best_metrics.get("alpha")),
        "Tracking Error": fmt_pct(best_metrics.get("tracking_error")),
        "Information Ratio": fmt_num(best_metrics.get("information_ratio"), 3),
        "Win Rate": fmt_pct(best_metrics.get("win_rate")),
        "Profit Factor": fmt_num(best_metrics.get("profit_factor"), 3),
    }
    st.dataframe(
        pd.DataFrame({"Metric": list(summary_rows.keys()), "Value": list(summary_rows.values())}),
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# STRATEGY COMPARISON
# =========================================================
with tab_strategies:
    st.markdown("### Strategy Comparison Table")
    metrics_df = safe_df(engine.metrics_df)

    if not metrics_df.empty:
        display_cols = [c for c in metric_columns_for_display() if c in metrics_df.columns]
        display_df = format_metrics_df(metrics_df[display_cols].copy())
        st.dataframe(display_df, use_container_width=True)
    else:
        st.info("No strategy metrics are available.")

    strategy_curves = build_strategy_return_frame(engine)
    if not strategy_curves.empty:
        fig = make_line_chart(
            strategy_curves,
            title="Cumulative Return Comparison Across Strategies",
            yaxis_title="Cumulative Return",
            height=560,
        )
        st.plotly_chart(fig, use_container_width=True)

    strategy_dd = build_strategy_drawdown_frame(engine)
    if not strategy_dd.empty:
        fig = make_line_chart(
            strategy_dd,
            title="Drawdown Comparison Across Strategies",
            yaxis_title="Drawdown",
            height=520,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Strategy Diagnostics")
    st.dataframe(engine.strategy_df, use_container_width=True)


# =========================================================
# RISK ANALYTICS
# =========================================================
with tab_risk:
    st.markdown("### Risk Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card("Tracking Error", fmt_pct(best_metrics.get("tracking_error")), "Annualized")
    with c2:
        render_kpi_card("Information Ratio", fmt_num(best_metrics.get("information_ratio"), 3), "Active efficiency")
    with c3:
        render_kpi_card("Beta", fmt_num(best_metrics.get("beta"), 3), "Relative sensitivity")
    with c4:
        render_kpi_card("Alpha", fmt_pct(best_metrics.get("alpha")), "Annualized excess")

    portfolio_returns = safe_series(best_metrics.get("portfolio_returns"))
    benchmark_returns = safe_series(best_metrics.get("benchmark_returns"))

    if not portfolio_returns.empty and not benchmark_returns.empty:
        rolling_sharpe = engine.analytics.rolling_sharpe(
            portfolio_returns,
            window=engine.config.rolling_window,
            min_periods=30,
        )
        rolling_beta = engine.analytics.rolling_beta(
            portfolio_returns,
            benchmark_returns,
            window=engine.config.rolling_window,
            min_periods=30,
        )
        rolling_ir = engine.analytics.rolling_information_ratio(
            portfolio_returns,
            benchmark_returns,
            window=engine.config.rolling_window,
            min_periods=30,
        )
        rolling_te = engine.analytics.rolling_tracking_error(
            portfolio_returns,
            benchmark_returns,
            window=engine.config.rolling_window,
            min_periods=30,
        )

        for title, series, ytitle in [
            ("Rolling Sharpe Ratio", rolling_sharpe, "Sharpe"),
            ("Rolling Beta", rolling_beta, "Beta"),
            ("Rolling Information Ratio", rolling_ir, "Information Ratio"),
            ("Rolling Tracking Error", rolling_te, "Tracking Error"),
        ]:
            s = safe_series(series)
            if not s.empty:
                fig = make_line_chart(
                    pd.DataFrame({title: s}),
                    title=title,
                    yaxis_title=ytitle,
                    height=500,
                )
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Relative VaR / CVaR / Active Tail Metrics")
    tail_cols = [c for c in best_metrics.keys() if "var_" in c or "cvar_" in c]
    if tail_cols:
        tail_df = pd.DataFrame(
            {
                "Metric": tail_cols,
                "Value": [best_metrics.get(c) for c in tail_cols],
            }
        )
        tail_df["Value"] = tail_df["Value"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
        st.dataframe(tail_df, use_container_width=True, hide_index=True)

    rc_df = engine.risk_contributions.get(best_name, pd.DataFrame())
    if not rc_df.empty:
        st.markdown("### Risk Contribution")
        show_rc = rc_df.copy()
        for col in ["weight", "risk_contribution", "pct_risk_contribution"]:
            if col in show_rc.columns:
                show_rc[col] = show_rc[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
        if "marginal_risk" in show_rc.columns:
            show_rc["marginal_risk"] = show_rc["marginal_risk"].map(lambda x: f"{x:,.4f}" if pd.notna(x) else "N/A")
        st.dataframe(show_rc, use_container_width=True, hide_index=True)

        plot_df = rc_df.copy()
        if "asset" in plot_df.columns and "pct_risk_contribution" in plot_df.columns:
            fig = make_bar_chart(
                plot_df,
                x="asset",
                y="pct_risk_contribution",
                title=f"Risk Contribution by Asset: {best_name}",
                height=520,
            )
            st.plotly_chart(fig, use_container_width=True)


# =========================================================
# STRESS TESTING
# =========================================================
with tab_stress:
    st.markdown("### Historical Stress Dashboard")

    stress_df = engine.historical_stress.get(best_name, pd.DataFrame())
    stress_df = engine.filter_stress_dataframe(stress_df)

    if not stress_df.empty:
        worst_scenario = stress_df.iloc[0]
        stress_c1, stress_c2, stress_c3, stress_c4 = st.columns(4)

        with stress_c1:
            render_kpi_card("Worst Historical Scenario", str(worst_scenario.get("scenario", "N/A")), "Filtered view")
        with stress_c2:
            render_kpi_card("Average Severity", fmt_pct(stress_df["severity_score"].mean()), "Across filtered scenarios")
        with stress_c3:
            render_kpi_card("Worst Relative Return", fmt_pct(stress_df["relative_return"].min()), "Filtered scenarios")
        with stress_c4:
            render_kpi_card("Scenario Count", fmt_num(float(len(stress_df)), 0), "Passing filters")

        show_stress = stress_df.copy()
        for c in ["portfolio_return", "benchmark_return", "relative_return", "severity_score"]:
            if c in show_stress.columns:
                show_stress[c] = show_stress[c].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")

        st.dataframe(show_stress, use_container_width=True, hide_index=True)

        if "scenario" in stress_df.columns and "severity_score" in stress_df.columns:
            fig = make_bar_chart(
                stress_df,
                x="scenario",
                y="severity_score",
                color="family" if "family" in stress_df.columns else None,
                title="Scenario Severity Ranking",
                height=520,
            )
            st.plotly_chart(fig, use_container_width=True)

        selected_scenario = st.selectbox(
            "Scenario Path View",
            options=stress_df["scenario"].tolist(),
            key="stress_scenario_selector",
        )
        path_map = engine.historical_stress_paths.get(best_name, {})
        path_df = path_map.get(selected_scenario, pd.DataFrame())

        if not path_df.empty:
            fig = make_line_chart(
                path_df,
                title=f"Scenario Path: {selected_scenario}",
                yaxis_title="Cumulative Return",
                height=520,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No historical stress scenarios are available for the selected filters.")

    st.markdown("### Hypothetical Shock Analysis")
    hypo_df = engine.hypothetical_shocks.get(best_name, pd.DataFrame())
    if not hypo_df.empty:
        show_hypo = hypo_df.copy()
        if "shock" in show_hypo.columns:
            show_hypo["shock"] = show_hypo["shock"].map(lambda x: f"{x:.0%}" if pd.notna(x) else "N/A")
        if "portfolio_impact" in show_hypo.columns:
            show_hypo["portfolio_impact"] = show_hypo["portfolio_impact"].map(
                lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"
            )
        st.dataframe(show_hypo, use_container_width=True, hide_index=True)

    st.markdown("### Sharp Fluctuation Windows")
    sharp_df = engine.sharp_fluctuation_windows.get(best_name, pd.DataFrame())
    if not sharp_df.empty:
        show_sharp = sharp_df.copy()
        if "window_return" in show_sharp.columns:
            show_sharp["window_return"] = show_sharp["window_return"].map(
                lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"
            )
        st.dataframe(show_sharp, use_container_width=True, hide_index=True)


# =========================================================
# FACTOR PCA
# =========================================================
with tab_factors:
    st.markdown("### PCA Factor Analysis")
    pca_results = engine.pca_results or {}

    evr = safe_series(pca_results.get("explained_variance_ratio"))
    loadings = safe_df(pca_results.get("loadings"))
    scores = safe_df(pca_results.get("scores"))
    interpretation = pca_results.get("interpretation", {})

    if not evr.empty:
        evr_df = evr.reset_index()
        evr_df.columns = ["Principal Component", "Explained Variance Ratio"]
        fig = make_bar_chart(
            evr_df,
            x="Principal Component",
            y="Explained Variance Ratio",
            title="Explained Variance by Principal Component",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

    if not loadings.empty:
        st.markdown("### PCA Loadings")
        st.dataframe(loadings.round(4), use_container_width=True)

    if interpretation:
        st.markdown("### Factor Interpretation")
        interp_df = pd.DataFrame(
            {"Factor": list(interpretation.keys()), "Interpretation": list(interpretation.values())}
        )
        st.dataframe(interp_df, use_container_width=True, hide_index=True)

    if not scores.empty and {"PC1", "PC2"}.issubset(set(scores.columns)):
        scatter_df = scores[["PC1", "PC2"]].copy()
        scatter_df["Date"] = scatter_df.index.astype(str)
        fig = px.scatter(
            scatter_df,
            x="PC1",
            y="PC2",
            hover_name="Date",
            title="PC1 vs PC2 Score Scatter",
        )
        fig.update_layout(
            template="plotly_white",
            height=560,
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)


# =========================================================
# DATA / DIAGNOSTICS
# =========================================================
with tab_data:
    st.markdown("### Universe Metadata")
    st.dataframe(engine.data.asset_metadata, use_container_width=True, hide_index=True)

    st.markdown("### Asset Price Sample")
    st.dataframe(engine.data.asset_prices.tail(20), use_container_width=True)

    st.markdown("### Asset Return Sample")
    st.dataframe(engine.data.asset_returns.tail(20), use_container_width=True)

    st.markdown("### Benchmark Return Sample")
    st.dataframe(engine.data.benchmark_returns.tail(20).to_frame("benchmark_return"), use_container_width=True)

    st.markdown("### Diagnostics Summary")
    diag_df = pd.DataFrame(
        {
            "Key": list(diag.get("info", {}).keys()),
            "Value": list(diag.get("info", {}).values()),
        }
    )
    st.dataframe(diag_df, use_container_width=True, hide_index=True)

    st.markdown(
        f"""
        <div class="small-note">
            Universe: <b>{engine.config.selected_universe}</b> &nbsp;|&nbsp;
            Benchmark: <b>{engine.config.benchmark_symbol}</b> &nbsp;|&nbsp;
            Start Date: <b>{engine.config.default_start_date}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

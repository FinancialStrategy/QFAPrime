from __future__ import annotations

from typing import Dict, List, Optional

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
    page_title="QFA Prime Finance Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# STYLING (BAŞLIK ÇERÇEVEDEN ÇOK UZAK)
# =========================================================
CUSTOM_CSS = """
<style>
    .main > div {
        padding-top: 0.10rem;
    }

    .block-container {
        max-width: 100% !important;
        width: 100% !important;
        padding-top: 0.20rem !important;
        padding-left: 2.2rem !important;
        padding-right: 2.2rem !important;
        padding-bottom: 1.2rem !important;
    }

    section.main > div {
        max-width: 100% !important;
    }

    .stApp {
        max-width: 100% !important;
    }

    .hero-outer {
        width: 100%;
        display: block;
        margin: 0 0 0.75rem 0;
    }

    /* KART – ÜST PADDING ÇOK BÜYÜK (BAŞLIK ÇERÇEVEDEN UZAK) */
    .hero-shell {
        width: 100%;
        background: #ffffff;
        border: 2px solid #cbd5e1;
        border-radius: 16px;
        padding: 80px 24px 40px 24px;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
        overflow: visible;
    }

    .hero-title {
        display: block;
        width: 100%;
        font-size: 1.6rem !important;
        font-weight: 700;
        line-height: 1.3;
        color: #0f172a;
        text-align: left;
        margin: 0 0 16px 0;
        padding: 0;
        letter-spacing: -0.01em;
        word-break: break-word;
        overflow-wrap: break-word;
        white-space: normal;
    }

    .hero-subtitle {
        display: block;
        width: 100%;
        font-size: 1rem !important;
        font-weight: 500;
        line-height: 1.4;
        color: #475569;
        text-align: left;
        margin-top: 0.5rem;
        padding: 0;
        word-break: break-word;
        overflow-wrap: break-word;
        white-space: normal;
    }

    .kpi-card {
        background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 12px;
        padding: 12px 12px 10px 12px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.12);
        min-height: 94px;
    }

    .kpi-title {
        font-size: 0.68rem;
        color: #cbd5e1;
        margin-bottom: 6px;
        font-weight: 600;
        line-height: 1.2;
    }

    .kpi-value {
        font-size: 1.06rem;
        color: white;
        font-weight: 800;
        line-height: 1.1;
    }

    .kpi-sub {
        font-size: 0.68rem;
        color: #94a3b8;
        margin-top: 5px;
        line-height: 1.2;
    }

    .section-label {
        font-size: 0.90rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0.25rem 0 0.6rem 0;
    }

    .small-note {
        color: #64748b;
        font-size: 0.72rem;
    }

    /* MOBİL UYUM */
    @media (max-width: 1100px) {
        .block-container {
            padding-left: 1.2rem !important;
            padding-right: 1.2rem !important;
        }

        .hero-shell {
            padding: 50px 18px 30px 18px;
        }

        .hero-title {
            font-size: 1.3rem !important;
        }

        .hero-subtitle {
            font-size: 0.85rem !important;
        }
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


def safe_series(obj) -> pd.Series:
    if isinstance(obj, pd.Series):
        return obj.dropna()
    return pd.Series(dtype=float)


def safe_df(obj) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
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
    height: int = 500,
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
    height: int = 500,
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


def build_strategy_return_frame(engine: ProfessionalPortfolioEngine) -> pd.DataFrame:
    curves = {}
    for strategy_name, metrics in engine.metrics.items():
        pr = safe_series(metrics.get("portfolio_returns"))
        if not pr.empty:
            curves[strategy_name] = cumulative_curve(pr)
    if not curves:
        return pd.DataFrame()
    return pd.DataFrame(curves)


def build_strategy_drawdown_frame(engine: ProfessionalPortfolioEngine) -> pd.DataFrame:
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

    num_cols = [
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "beta",
        "information_ratio",
        "profit_factor",
    ]

    for c in pct_cols:
        if c in out.columns:
            out[c] = out[c].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")

    for c in num_cols:
        if c in out.columns:
            out[c] = out[c].map(lambda x: f"{x:,.3f}" if pd.notna(x) else "N/A")

    return out


def prepare_stress_display_table(df: pd.DataFrame) -> pd.DataFrame:
    show_df = df.copy()
    for col in ["portfolio_return", "benchmark_return", "relative_return", "severity_score"]:
        if col in show_df.columns:
            show_df[col] = show_df[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
    return show_df


def prepare_tail_metrics(best_metrics: Dict) -> pd.DataFrame:
    rows = []
    for key, val in best_metrics.items():
        key_str = str(key).lower()
        if "var_" in key_str or "cvar_" in key_str:
            rows.append({"Metric": key, "Value": val})

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["Value"] = out["Value"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
    return out


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## Portfolio Gate")

    available_universes = list(UNIVERSE_REGISTRY.keys())
    default_universe = (
        "institutional_multi_asset"
        if "institutional_multi_asset" in available_universes
        else available_universes[0]
    )

    selected_universe = st.selectbox(
        "Investment Universe",
        options=available_universes,
        index=available_universes.index(default_universe),
    )

    benchmark_symbol = st.text_input("Benchmark Symbol", value="^GSPC")
    default_start_date = st.text_input("Start Date", value="2019-01-01")

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

    min_observations = st.number_input(
        "Minimum Observations",
        min_value=20,
        value=60,
        step=5,
    )

    rolling_window = st.number_input(
        "Rolling Window",
        min_value=20,
        value=63,
        step=1,
    )

    use_log_returns = st.checkbox("Use Log Returns", value=False)
    allow_short = st.checkbox("Allow Short Selling", value=False)

    st.markdown("---")
    st.markdown("### Expected Return / Risk Settings")

    expected_return_method = st.selectbox(
        "Expected Return Method",
        options=["historical_mean"],
        index=0,
    )

    covariance_method = st.selectbox(
        "Covariance Method",
        options=["sample_cov"],
        index=0,
    )

    correlation_method = st.selectbox(
        "Correlation Method",
        options=["pearson"],
        index=0,
    )

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
# HEADER
# =========================================================
st.markdown(
    """
    <div class="hero-outer">
        <div class="hero-shell">
            <div class="hero-title">QFA Prime Finance Platform</div>
            <div class="hero-subtitle">
                Institutional portfolio analytics, risk diagnostics, stress testing, and factor intelligence
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SESSION STATE
# =========================================================
if "engine_result" not in st.session_state:
    st.session_state.engine_result = None

if "engine_error" not in st.session_state:
    st.session_state.engine_error = None


# =========================================================
# ENGINE EXECUTION
# =========================================================
if run_button or st.session_state.engine_result is None:
    try:
        config = ProfessionalConfig(
            benchmark_symbol=benchmark_symbol,
            default_start_date=default_start_date,
            initial_capital=float(initial_capital),
            risk_free_rate=float(risk_free_rate),
            min_observations=int(min_observations),
            rolling_window=int(rolling_window),
            use_log_returns=bool(use_log_returns),
            allow_short=bool(allow_short),
            selected_universe=selected_universe,
            expected_return_method=expected_return_method,
            covariance_method=covariance_method,
            correlation_method=correlation_method,
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
        for warning_msg in diag["warnings"]:
            st.warning(str(warning_msg))

if diag.get("errors"):
    with st.expander("Diagnostics Errors", expanded=False):
        for error_msg in diag["errors"]:
            st.error(str(error_msg))


# =========================================================
# BEST STRATEGY KPIS
# =========================================================
best_name = engine.best_strategy_name()
best_metrics = engine.metrics.get(best_name, {})

k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    render_kpi_card("Best Strategy", best_name, "Top rank by Sharpe ratio")
with k2:
    render_kpi_card("Annual Return", fmt_pct(best_metrics.get("annual_return")), "Annualized")
with k3:
    render_kpi_card("Volatility", fmt_pct(best_metrics.get("volatility")), "Annualized")
with k4:
    render_kpi_card("Sharpe Ratio", fmt_num(best_metrics.get("sharpe_ratio"), 3), "Risk-adjusted")
with k5:
    render_kpi_card("Max Drawdown", fmt_pct(best_metrics.get("max_drawdown")), "Peak-to-trough")
with k6:
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
    st.markdown('<div class="section-label">Executive Summary</div>', unsafe_allow_html=True)

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
            height=540,
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
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

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

    st.markdown('<div class="section-label">Best Strategy Metrics</div>', unsafe_allow_html=True)
    st.dataframe(
        pd.DataFrame({"Metric": list(summary_rows.keys()), "Value": list(summary_rows.values())}),
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# STRATEGY COMPARISON
# =========================================================
with tab_strategies:
    st.markdown('<div class="section-label">Strategy Comparison Table</div>', unsafe_allow_html=True)

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
            height=540,
        )
        st.plotly_chart(fig, use_container_width=True)

    strategy_dd = build_strategy_drawdown_frame(engine)
    if not strategy_dd.empty:
        fig = make_line_chart(
            strategy_dd,
            title="Drawdown Comparison Across Strategies",
            yaxis_title="Drawdown",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

    if not engine.strategy_df.empty:
        st.markdown('<div class="section-label">Strategy Diagnostics</div>', unsafe_allow_html=True)
        st.dataframe(engine.strategy_df, use_container_width=True)


# =========================================================
# RISK ANALYTICS
# =========================================================
with tab_risk:
    st.markdown('<div class="section-label">Risk Dashboard</div>', unsafe_allow_html=True)

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        render_kpi_card("Tracking Error", fmt_pct(best_metrics.get("tracking_error")), "Annualized")
    with r2:
        render_kpi_card("Information Ratio", fmt_num(best_metrics.get("information_ratio"), 3), "Active efficiency")
    with r3:
        render_kpi_card("Beta", fmt_num(best_metrics.get("beta"), 3), "Relative sensitivity")
    with r4:
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

        charts_to_show = [
            ("Rolling Sharpe Ratio", rolling_sharpe, "Sharpe"),
            ("Rolling Beta", rolling_beta, "Beta"),
            ("Rolling Information Ratio", rolling_ir, "Information Ratio"),
            ("Rolling Tracking Error", rolling_te, "Tracking Error"),
        ]

        for title, series_obj, ytitle in charts_to_show:
            s = safe_series(series_obj)
            if not s.empty:
                fig = make_line_chart(
                    pd.DataFrame({title: s}),
                    title=title,
                    yaxis_title=ytitle,
                    height=480,
                )
                st.plotly_chart(fig, use_container_width=True)

    tail_df = prepare_tail_metrics(best_metrics)
    if not tail_df.empty:
        st.markdown('<div class="section-label">Relative VaR / CVaR / Tail Metrics</div>', unsafe_allow_html=True)
        st.dataframe(tail_df, use_container_width=True, hide_index=True)

    rc_df = engine.risk_contributions.get(best_name, pd.DataFrame())
    if not rc_df.empty:
        st.markdown('<div class="section-label">Risk Contribution Table</div>', unsafe_allow_html=True)

        show_rc = rc_df.copy()
        for col in ["weight", "risk_contribution", "pct_risk_contribution"]:
            if col in show_rc.columns:
                show_rc[col] = show_rc[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")

        if "marginal_risk" in show_rc.columns:
            show_rc["marginal_risk"] = show_rc["marginal_risk"].map(
                lambda x: f"{x:,.4f}" if pd.notna(x) else "N/A"
            )

        st.dataframe(show_rc, use_container_width=True, hide_index=True)

        if "asset" in rc_df.columns and "pct_risk_contribution" in rc_df.columns:
            fig = make_bar_chart(
                rc_df,
                x="asset",
                y="pct_risk_contribution",
                title=f"Risk Contribution by Asset: {best_name}",
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)


# =========================================================
# STRESS TESTING
# =========================================================
with tab_stress:
    st.markdown('<div class="section-label">Historical Stress Dashboard</div>', unsafe_allow_html=True)

    stress_df = engine.historical_stress.get(best_name, pd.DataFrame())
    stress_df = engine.filter_stress_dataframe(stress_df)

    if not stress_df.empty:
        worst_scenario = stress_df.iloc[0]

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            render_kpi_card("Worst Historical Scenario", str(worst_scenario.get("scenario", "N/A")), "Filtered view")
        with s2:
            render_kpi_card("Average Severity", fmt_pct(stress_df["severity_score"].mean()), "Across filtered scenarios")
        with s3:
            render_kpi_card("Worst Relative Return", fmt_pct(stress_df["relative_return"].min()), "Filtered scenarios")
        with s4:
            render_kpi_card("Scenario Count", fmt_num(float(len(stress_df)), 0), "Passing filters")

        st.dataframe(
            prepare_stress_display_table(stress_df),
            use_container_width=True,
            hide_index=True,
        )

        if "scenario" in stress_df.columns and "severity_score" in stress_df.columns:
            fig = make_bar_chart(
                stress_df,
                x="scenario",
                y="severity_score",
                color="family" if "family" in stress_df.columns else None,
                title="Scenario Severity Ranking",
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

        scenario_options = stress_df["scenario"].astype(str).tolist()
        selected_scenario = st.selectbox(
            "Scenario Path View",
            options=scenario_options,
            key="stress_scenario_selector",
        )

        path_map = engine.historical_stress_paths.get(best_name, {})
        path_df = path_map.get(selected_scenario, pd.DataFrame())

        if not path_df.empty:
            fig = make_line_chart(
                path_df,
                title=f"Scenario Path: {selected_scenario}",
                yaxis_title="Cumulative Return",
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No historical stress scenarios are available for the selected filters.")

    hypo_df = engine.hypothetical_shocks.get(best_name, pd.DataFrame())
    if not hypo_df.empty:
        st.markdown('<div class="section-label">Hypothetical Shock Analysis</div>', unsafe_allow_html=True)
        show_hypo = hypo_df.copy()
        if "shock" in show_hypo.columns:
            show_hypo["shock"] = show_hypo["shock"].map(lambda x: f"{x:.0%}" if pd.notna(x) else "N/A")
        if "portfolio_impact" in show_hypo.columns:
            show_hypo["portfolio_impact"] = show_hypo["portfolio_impact"].map(
                lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"
            )
        st.dataframe(show_hypo, use_container_width=True, hide_index=True)

    sharp_df = engine.sharp_fluctuation_windows.get(best_name, pd.DataFrame())
    if not sharp_df.empty:
        st.markdown('<div class="section-label">Sharp Fluctuation Windows</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="section-label">PCA Factor Analysis</div>', unsafe_allow_html=True)

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
            height=460,
        )
        st.plotly_chart(fig, use_container_width=True)

    if not loadings.empty:
        st.markdown('<div class="section-label">PCA Loadings</div>', unsafe_allow_html=True)
        st.dataframe(loadings.round(4), use_container_width=True)

    if interpretation:
        st.markdown('<div class="section-label">Factor Interpretation</div>', unsafe_allow_html=True)
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
            height=500,
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)


# =========================================================
# DATA & DIAGNOSTICS
# =========================================================
with tab_data:
    st.markdown('<div class="section-label">Universe Metadata</div>', unsafe_allow_html=True)
    st.dataframe(engine.data.asset_metadata, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-label">Asset Price Sample</div>', unsafe_allow_html=True)
    st.dataframe(engine.data.asset_prices.tail(20), use_container_width=True)

    st.markdown('<div class="section-label">Asset Return Sample</div>', unsafe_allow_html=True)
    st.dataframe(engine.data.asset_returns.tail(20), use_container_width=True)

    st.markdown('<div class="section-label">Benchmark Return Sample</div>', unsafe_allow_html=True)
    st.dataframe(
        engine.data.benchmark_returns.tail(20).to_frame("benchmark_return"),
        use_container_width=True,
    )

    info_items = diag.get("info", {})
    if info_items:
        st.markdown('<div class="section-label">Diagnostics Summary</div>', unsafe_allow_html=True)
        diag_df = pd.DataFrame(
            {"Key": list(info_items.keys()), "Value": list(info_items.values())}
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

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.config import ProfessionalConfig
from core.engine import ProfessionalPortfolioEngine
from core.universes import UNIVERSE_REGISTRY


# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="QFA Prime Finance Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# Styling
# =========================================================
CUSTOM_CSS = """
<style>
.block-container {
    padding-top: 1.15rem;
    padding-bottom: 2rem;
    max-width: 1600px;
}
.qfa-hero {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 20px;
    padding: 1.5rem 1.5rem 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.qfa-hero h1 {
    font-size: 2rem;
    font-weight: 800;
    margin: 0 0 0.25rem 0;
    color: #0f172a;
}
.qfa-hero p {
    margin: 0;
    color: #334155;
    font-size: 1.02rem;
}
.qfa-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 1rem 1rem 0.85rem 1rem;
    min-height: 132px;
}
.qfa-card-title {
    color: #64748b;
    font-size: 0.90rem;
    margin-bottom: 0.4rem;
}
.qfa-card-value {
    color: #0f172a;
    font-size: 1.45rem;
    font-weight: 800;
    margin-bottom: 0.25rem;
}
.qfa-card-sub {
    color: #94a3b8;
    font-size: 0.82rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =========================================================
# Helpers
# =========================================================
def fmt_pct(x):
    if x is None or pd.isna(x):
        return "N/A"
    return f"{x:.2%}"


def fmt_num(x, decimals=2):
    if x is None or pd.isna(x):
        return "N/A"
    return f"{x:,.{decimals}f}"


def fmt_usd(x):
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


def cumulative_curve(returns: pd.Series) -> pd.Series:
    s = safe_series(returns)
    if s.empty:
        return s
    return (1.0 + s).cumprod() - 1.0


def render_kpi_card(title: str, value: str, sub: str = "") -> None:
    st.markdown(
        f"""
        <div class="qfa-card">
            <div class="qfa-card-title">{title}</div>
            <div class="qfa-card-value">{value}</div>
            <div class="qfa-card-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_line_chart(df: pd.DataFrame, title: str, yaxis_title: str = "", height: int = 500) -> go.Figure:
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
        template="plotly_white",
        height=height,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        xaxis_title="Date",
        yaxis_title=yaxis_title,
    )
    return fig


def make_bar_chart(df: pd.DataFrame, x: str, y: str, title: str, color: str | None = None, height: int = 500) -> go.Figure:
    fig = px.bar(df, x=x, y=y, color=color, title=title)
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title=x,
        yaxis_title=y,
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
        "VaR_95",
        "CVaR_95",
        "VaR_99",
        "CVaR_99",
    ]

    num_cols = [
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "beta",
        "information_ratio",
        "profit_factor",
        "final_portfolio_value",
    ]

    for c in pct_cols:
        if c in out.columns:
            out[c] = out[c].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")

    for c in num_cols:
        if c in out.columns:
            if c == "final_portfolio_value":
                out[c] = out[c].map(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
            else:
                out[c] = out[c].map(lambda x: f"{x:,.3f}" if pd.notna(x) else "N/A")

    return out


def prepare_stress_display_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    show_df = df.copy()
    for col in ["portfolio_return", "benchmark_return", "relative_return", "severity_score"]:
        if col in show_df.columns:
            show_df[col] = show_df[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
    return show_df


def prepare_tail_metrics(best_metrics: dict) -> pd.DataFrame:
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
# Sidebar
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
    st.caption(
        "Yahoo Finance on free cloud instances can throttle requests. "
        "If that happens, wait briefly and rerun with a smaller universe."
    )

    run_button = st.button(
        "Run Professional Analytics",
        type="primary",
        use_container_width=True,
    )


# =========================================================
# Header
# =========================================================
st.markdown(
    """
    <div class="qfa-hero">
        <h1>QFA Prime Finance Platform</h1>
        <p>Institutional portfolio analytics, risk diagnostics, stress testing, and factor intelligence</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Session state
# =========================================================
if "engine_result" not in st.session_state:
    st.session_state.engine_result = None

if "engine_error" not in st.session_state:
    st.session_state.engine_error = None

if "last_run_params" not in st.session_state:
    st.session_state.last_run_params = None


current_params = {
    "benchmark_symbol": benchmark_symbol,
    "default_start_date": default_start_date,
    "initial_capital": float(initial_capital),
    "risk_free_rate": float(risk_free_rate),
    "min_observations": int(min_observations),
    "rolling_window": int(rolling_window),
    "use_log_returns": bool(use_log_returns),
    "allow_short": bool(allow_short),
    "selected_universe": selected_universe,
    "expected_return_method": expected_return_method,
    "covariance_method": covariance_method,
    "correlation_method": correlation_method,
    "bl_enabled": bool(bl_enabled),
    "selected_family": selected_family,
    "minimum_severity_threshold": float(minimum_severity_threshold),
    "quick_view": quick_view,
}

if run_button:
    st.session_state.last_run_params = current_params.copy()


# =========================================================
# Initial state
# =========================================================
if st.session_state.last_run_params is None:
    st.info("Choose your settings in the sidebar, then click **Run Professional Analytics**.")
    st.stop()

run_params = st.session_state.last_run_params


# =========================================================
# Run engine
# =========================================================
with st.spinner("Downloading data and running portfolio analytics..."):
    try:
        config = ProfessionalConfig(
            benchmark_symbol=run_params["benchmark_symbol"],
            default_start_date=run_params["default_start_date"],
            initial_capital=run_params["initial_capital"],
            risk_free_rate=run_params["risk_free_rate"],
            min_observations=run_params["min_observations"],
            rolling_window=run_params["rolling_window"],
            use_log_returns=run_params["use_log_returns"],
            allow_short=run_params["allow_short"],
            selected_universe=run_params["selected_universe"],
            expected_return_method=run_params["expected_return_method"],
            covariance_method=run_params["covariance_method"],
            correlation_method=run_params["correlation_method"],
        )

        engine = ProfessionalPortfolioEngine(
            config=config,
            bl_controls={
                "enabled": run_params["bl_enabled"],
                "view_mode": "ticker",
                "views_payload": [],
            },
            scenario_controls={
                "selected_family": run_params["selected_family"],
                "minimum_severity_threshold": run_params["minimum_severity_threshold"],
                "quick_view": run_params["quick_view"],
            },
        )

        engine.run()
        st.session_state.engine_result = engine
        st.session_state.engine_error = None

    except Exception as exc:
        st.session_state.engine_result = None
        st.session_state.engine_error = str(exc)


engine = st.session_state.engine_result
engine_error = st.session_state.engine_error


# =========================================================
# Error handling
# =========================================================
if engine_error:
    st.error(f"Application error: {engine_error}")

    lowered = engine_error.lower()
    if "rate limit" in lowered or "too many requests" in lowered:
        st.warning(
            "Yahoo Finance appears to be throttling requests. "
            "Wait 30-60 seconds and rerun, preferably with a smaller universe."
        )

    if "does not contain enough assets" in lowered:
        st.warning(
            "The selected universe is invalid. Please verify that the selected universe "
            "contains at least two tickers in core/universes.py."
        )

    st.stop()

if engine is None:
    st.warning("No analysis output is available.")
    st.stop()


# =========================================================
# Diagnostics
# =========================================================
diag = engine.diagnostics.summary()

if diag.get("warnings"):
    with st.expander("Diagnostics Warnings", expanded=False):
        for warning_msg in diag["warnings"]:
            st.warning(str(warning_msg))

if diag.get("errors"):
    with st.expander("Diagnostics Errors", expanded=True):
        for error_msg in diag["errors"]:
            st.error(str(error_msg))


# =========================================================
# Best strategy / KPIs
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
    render_kpi_card(
        "Final Portfolio Value",
        fmt_usd(best_metrics.get("final_portfolio_value")),
        "Based on initial capital",
    )

st.markdown("")


# =========================================================
# Tabs
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
# Overview
# =========================================================
with tab_overview:
    st.subheader("Executive Summary")

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

    st.subheader("Best Strategy Metrics")
    st.dataframe(
        pd.DataFrame({"Metric": list(summary_rows.keys()), "Value": list(summary_rows.values())}),
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# Strategy comparison
# =========================================================
with tab_strategies:
    st.subheader("Strategy Comparison Table")

    metrics_df = safe_df(engine.metrics_df)
    if not metrics_df.empty:
        display_df = format_metrics_df(metrics_df.copy())
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

    strategy_df = safe_df(engine.strategy_df)
    if not strategy_df.empty:
        st.subheader("Strategy Diagnostics")
        st.dataframe(strategy_df, use_container_width=True)


# =========================================================
# Risk analytics
# =========================================================
with tab_risk:
    st.subheader("Risk Dashboard")

    tail_df = prepare_tail_metrics(best_metrics)
    if not tail_df.empty:
        st.dataframe(tail_df, use_container_width=True, hide_index=True)
    else:
        st.info("Tail metrics are not available for the best strategy.")

    weights = best_metrics.get("weights")
    if isinstance(weights, pd.Series) and not weights.empty:
        weights_df = (
            weights.sort_values(ascending=False)
            .reset_index()
            .rename(columns={"index": "Asset", 0: "Weight"})
        )
        fig = make_bar_chart(
            weights_df,
            x="Asset",
            y="Weight",
            title=f"Portfolio Weights - {best_name}",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)


# =========================================================
# Stress testing
# =========================================================
with tab_stress:
    st.subheader("Stress Testing")

    stress_df = safe_df(getattr(engine, "stress_table", pd.DataFrame()))
    if stress_df.empty:
        stress_df = safe_df(best_metrics.get("stress_table"))

    if not stress_df.empty:
        st.dataframe(prepare_stress_display_table(stress_df), use_container_width=True)

        if "scenario_name" in stress_df.columns and "relative_return" in stress_df.columns:
            stress_plot_df = stress_df.copy()
            fig = make_bar_chart(
                stress_plot_df,
                x="scenario_name",
                y="relative_return",
                title="Scenario Relative Return Impact",
                color="family" if "family" in stress_plot_df.columns else None,
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Stress testing output is not available.")


# =========================================================
# Factor PCA
# =========================================================
with tab_factors:
    st.subheader("Factor PCA")

    factor_df = safe_df(getattr(engine, "factor_pca_df", pd.DataFrame()))
    if not factor_df.empty:
        st.dataframe(factor_df, use_container_width=True)
    else:
        st.info("Factor PCA output is not available.")


# =========================================================
# Data & diagnostics
# =========================================================
with tab_data:
    st.subheader("Data & Diagnostics")

    st.write("Selected universe:", run_params["selected_universe"])
    st.write("Universe tickers:", UNIVERSE_REGISTRY.get(run_params["selected_universe"], []))
    st.write("Benchmark:", run_params["benchmark_symbol"])
    st.write("Start date:", run_params["default_start_date"])

    if hasattr(engine, "prices") and isinstance(engine.prices, pd.DataFrame) and not engine.prices.empty:
        st.markdown("**Price History Preview**")
        st.dataframe(engine.prices.tail(10), use_container_width=True)

    if hasattr(engine, "returns") and isinstance(engine.returns, pd.DataFrame) and not engine.returns.empty:
        st.markdown("**Return Matrix Preview**")
        st.dataframe(engine.returns.tail(10), use_container_width=True)

    st.markdown("**Diagnostics Summary**")
    st.json(diag)

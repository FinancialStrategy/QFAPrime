import inspect
import traceback
from typing import Any, Dict, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.config import ProfessionalConfig
from core.engine import ProfessionalPortfolioEngine
from core.universes import UNIVERSE_REGISTRY


st.set_page_config(
    page_title="QFA Prime Finance Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
.block-container {
    padding-top: 1.0rem;
    padding-bottom: 2rem;
    max-width: 1680px;
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
    margin-bottom: 0.35rem;
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
.small-note {
    color: #475569;
    font-size: 0.93rem;
    line-height: 1.58;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


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


def make_line_chart(df: pd.DataFrame, title: str, yaxis_title: str = "", height: int = 520) -> go.Figure:
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=str(col)))
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


def make_bar_chart(df: pd.DataFrame, x: str, y: str, title: str, color: Optional[str] = None, height: int = 500) -> go.Figure:
    fig = px.bar(df, x=x, y=y, color=color, title=title)
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def build_strategy_return_frame(engine: Any) -> pd.DataFrame:
    curves = {}
    for strategy_name, metrics in getattr(engine, "metrics", {}).items():
        pr = safe_series(metrics.get("portfolio_returns"))
        if not pr.empty:
            curves[strategy_name] = cumulative_curve(pr)
    return pd.DataFrame(curves) if curves else pd.DataFrame()


def build_strategy_drawdown_frame(engine: Any) -> pd.DataFrame:
    dds = {}
    for strategy_name, metrics in getattr(engine, "metrics", {}).items():
        dd = safe_series(metrics.get("drawdown_series"))
        if not dd.empty:
            dds[strategy_name] = dd
    return pd.DataFrame(dds) if dds else pd.DataFrame()


def format_metrics_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    pct_cols = [
        "annual_return", "annual_return_benchmark", "volatility", "max_drawdown",
        "tracking_error", "win_rate", "win_rate_vs_benchmark",
        "total_return_pct", "total_return_benchmark_pct", "excess_return_vs_benchmark_pct",
        "alpha", "VaR_95", "CVaR_95", "VaR_99", "CVaR_99",
        "var_95", "cvar_95", "relative_var_95", "relative_cvar_95",
        "portfolio_return", "benchmark_return", "relative_return", "severity_score"
    ]
    num_cols = [
        "sharpe_ratio", "sortino_ratio", "calmar_ratio", "beta",
        "information_ratio", "profit_factor", "final_portfolio_value"
    ]

    for c in pct_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")

    for c in num_cols:
        if c in out.columns:
            s = pd.to_numeric(out[c], errors="coerce")
            if c == "final_portfolio_value":
                out[c] = s.map(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
            else:
                out[c] = s.map(lambda x: f"{x:,.3f}" if pd.notna(x) else "N/A")

    return out


def prepare_stress_display_table(df: pd.DataFrame) -> pd.DataFrame:
    show_df = df.copy()
    for col in ["portfolio_return", "benchmark_return", "relative_return", "severity_score"]:
        if col in show_df.columns:
            show_df[col] = pd.to_numeric(show_df[col], errors="coerce").map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
    return show_df


def prepare_tail_metrics(best_metrics: Dict) -> pd.DataFrame:
    rows = []
    for key, val in best_metrics.items():
        key_str = str(key).lower()
        if "var" in key_str or "cvar" in key_str:
            rows.append({"Metric": key, "Value": val})
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["Value"] = pd.to_numeric(out["Value"], errors="coerce").map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
    return out


def show_plotly(fig):
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)


def get_chart_dict(engine: Any) -> Dict[str, Any]:
    charts = getattr(engine, "charts", {})
    return charts if isinstance(charts, dict) else {}


def get_finquant_chart_dict(engine: Any) -> Dict[str, Any]:
    charts = getattr(engine, "finquant_charts", {})
    return charts if isinstance(charts, dict) else {}


def call_constructor_flex(cls, payload: Dict[str, Any]):
    sig = inspect.signature(cls)
    valid = {k: v for k, v in payload.items() if k in sig.parameters}
    return cls(**valid)


st.markdown(
    """
    <div class="qfa-hero">
        <h1>QFA Prime Finance Platform</h1>
        <p>Institutional portfolio analytics, portfolio optimization, FinQuant diagnostics, stress testing, benchmark-relative analysis, and factor intelligence</p>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown("## Portfolio Gate")

    available_universes = list(UNIVERSE_REGISTRY.keys())
    default_universe = "institutional_multi_asset" if "institutional_multi_asset" in available_universes else available_universes[0]

    selected_universe = st.selectbox(
        "Investment Universe",
        options=available_universes,
        index=available_universes.index(default_universe),
    )

    st.caption("Universe composition")
    st.code(", ".join(UNIVERSE_REGISTRY.get(selected_universe, [])), language=None)

    benchmark_symbol = st.text_input("Benchmark Symbol", value="^GSPC")
    default_start_date = st.text_input("Start Date", value="2019-01-01")

    initial_capital = st.number_input("Initial Capital", min_value=1000.0, value=100000.0, step=1000.0)
    risk_free_rate = st.number_input("Risk-Free Rate", min_value=0.0, max_value=1.0, value=0.03, step=0.005, format="%.3f")
    min_observations = st.number_input("Minimum Observations", min_value=20, value=60, step=5)
    rolling_window = st.number_input("Rolling Window", min_value=20, value=63, step=1)

    use_log_returns = st.checkbox("Use Log Returns", value=False)
    allow_short = st.checkbox("Allow Short Selling", value=False)

    st.markdown("---")
    st.markdown("### Model Settings")

    expected_return_method = st.selectbox(
        "Expected Return Method",
        options=["historical_mean", "ema_historical", "capm"],
        index=0,
    )

    covariance_method = st.selectbox(
        "Covariance Method",
        options=["sample_cov", "sample", "shrinkage", "ledoit_wolf"],
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
    minimum_severity_threshold = st.slider("Minimum Severity Threshold", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    quick_view = st.selectbox(
        "Quick View",
        options=[
            "All", "Crisis Only", "Inflation Only",
            "Banking Stress Only", "Sharp Rally Only", "Sharp Selloff Only",
        ],
        index=0,
    )

    st.markdown("---")
    st.caption(
        "Yahoo Finance on cloud instances can throttle requests. "
        "Use smaller universes and rerun after a short pause if needed."
    )

    run_button = st.button("Run Professional Analytics", type="primary", use_container_width=True)


if "engine_result" not in st.session_state:
    st.session_state.engine_result = None
if "engine_error" not in st.session_state:
    st.session_state.engine_error = None
if "last_run_params" not in st.session_state:
    st.session_state.last_run_params = None
if "has_run" not in st.session_state:
    st.session_state.has_run = False


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
    st.session_state.has_run = True
    st.session_state.engine_result = None
    st.session_state.engine_error = None

if not st.session_state.has_run:
    st.info("Choose your settings in the sidebar, then click **Run Professional Analytics**.")
    st.stop()

run_params = st.session_state.last_run_params


if st.session_state.engine_result is None and st.session_state.engine_error is None:
    with st.spinner("Downloading data and running portfolio analytics..."):
        try:
            config_payload = {
                "benchmark_symbol": run_params["benchmark_symbol"],
                "benchmark": run_params["benchmark_symbol"],
                "default_start_date": run_params["default_start_date"],
                "start_date": run_params["default_start_date"],
                "initial_capital": run_params["initial_capital"],
                "risk_free_rate": run_params["risk_free_rate"],
                "min_observations": run_params["min_observations"],
                "rolling_window": run_params["rolling_window"],
                "use_log_returns": run_params["use_log_returns"],
                "allow_short": run_params["allow_short"],
                "selected_universe": run_params["selected_universe"],
                "expected_return_method": run_params["expected_return_method"],
                "covariance_method": run_params["covariance_method"],
                "correlation_method": run_params["correlation_method"],
            }

            config = call_constructor_flex(ProfessionalConfig, config_payload)

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
            st.session_state.engine_error = f"{exc}\n\n{traceback.format_exc()}"


engine = st.session_state.engine_result
engine_error = st.session_state.engine_error

if engine_error:
    st.error("Application error")
    st.code(engine_error)
    lowered = engine_error.lower()

    if "rate limit" in lowered or "too many requests" in lowered:
        st.warning("Yahoo Finance appears to be throttling requests. Wait briefly and rerun with a smaller universe.")

    if "does not contain enough assets" in lowered:
        st.warning("The selected universe is invalid. Please verify that the universe contains at least two valid tickers.")

    st.stop()

if engine is None:
    st.warning("No analysis output is available.")
    st.stop()


diag = engine.diagnostics.summary() if hasattr(engine, "diagnostics") else {}

if isinstance(diag, dict) and diag.get("warnings"):
    with st.expander("Diagnostics Warnings", expanded=False):
        for warning_msg in diag["warnings"]:
            st.warning(str(warning_msg))

if isinstance(diag, dict) and diag.get("errors"):
    with st.expander("Diagnostics Errors", expanded=True):
        for error_msg in diag["errors"]:
            st.error(str(error_msg))


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


tabs = st.tabs([
    "Overview",
    "Info Hub",
    "Executive Dashboard",
    "Portfolio Optimization",
    "Relative Frontier",
    "Tracking Error",
    "Stress Testing",
    "Risk Analytics",
    "Rolling Beta",
    "FinQuant",
    "Factor PCA",
    "Data & Diagnostics",
])

(
    tab_overview,
    tab_info_hub,
    tab_dashboard,
    tab_optimization,
    tab_relative,
    tab_te,
    tab_stress,
    tab_risk,
    tab_beta,
    tab_finquant,
    tab_factors,
    tab_data,
) = tabs


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
        show_plotly(make_line_chart(curve_df, f"Cumulative Return: {best_name} vs Benchmark", "Cumulative Return", 540))

    dd_df = pd.DataFrame()
    if not best_drawdown.empty:
        dd_df["Portfolio Drawdown"] = best_drawdown
    if not best_benchmark_drawdown.empty:
        dd_df["Benchmark Drawdown"] = best_benchmark_drawdown

    if not dd_df.empty:
        show_plotly(make_line_chart(dd_df, f"Drawdown Profile: {best_name} vs Benchmark", "Drawdown", 500))

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


with tab_info_hub:
    st.subheader("Investment Universe Identity Map")
    charts = getattr(engine, "charts", {}) if isinstance(getattr(engine, "charts", {}), dict) else {}
    info_hub_chart = charts.get("info_hub")
    if info_hub_chart is not None:
        show_plotly(info_hub_chart)

    data_obj = getattr(engine, "data", None)
    if data_obj is not None and hasattr(data_obj, "asset_metadata"):
        asset_metadata = safe_df(data_obj.asset_metadata)
        if not asset_metadata.empty:
            st.dataframe(asset_metadata, use_container_width=True)


with tab_dashboard:
    st.subheader("Executive Strategy Dashboard")
    charts = getattr(engine, "charts", {}) if isinstance(getattr(engine, "charts", {}), dict) else {}
    dashboard_chart = charts.get("dashboard")
    radar_chart = charts.get("radar")

    if dashboard_chart is not None:
        show_plotly(dashboard_chart)
    if radar_chart is not None:
        show_plotly(radar_chart)

    strategy_df = safe_df(getattr(engine, "strategy_df", pd.DataFrame()))
    if not strategy_df.empty:
        st.dataframe(strategy_df, use_container_width=True)


with tab_optimization:
    st.subheader("Portfolio Optimization")
    charts = getattr(engine, "charts", {}) if isinstance(getattr(engine, "charts", {}), dict) else {}
    opt_chart = charts.get("optimization")
    if opt_chart is not None:
        show_plotly(opt_chart)

    metrics_df = safe_df(getattr(engine, "metrics_df", pd.DataFrame()))
    if not metrics_df.empty:
        st.dataframe(format_metrics_df(metrics_df.copy()), use_container_width=True)


with tab_relative:
    st.subheader("Benchmark-Relative Frontier")
    charts = getattr(engine, "charts", {}) if isinstance(getattr(engine, "charts", {}), dict) else {}
    rel_chart = charts.get("relative_frontier")
    if rel_chart is not None:
        show_plotly(rel_chart)


with tab_te:
    st.subheader("Tracking Error")
    charts = getattr(engine, "charts", {}) if isinstance(getattr(engine, "charts", {}), dict) else {}
    te_chart = charts.get("tracking_error")
    benchmark_vs_te = charts.get("benchmark_vs_te")

    if te_chart is not None:
        show_plotly(te_chart)
    if benchmark_vs_te is not None:
        show_plotly(benchmark_vs_te)


with tab_stress:
    st.subheader("Stress Testing")
    charts = getattr(engine, "charts", {}) if isinstance(getattr(engine, "charts", {}), dict) else {}
    stress_chart = charts.get("stress")
    if stress_chart is not None:
        show_plotly(stress_chart)

    stress_df = safe_df(getattr(engine, "stress_table", pd.DataFrame()))
    if stress_df.empty:
        stress_df = safe_df(getattr(engine, "stress_df", pd.DataFrame()))

    if not stress_df.empty:
        st.dataframe(prepare_stress_display_table(stress_df), use_container_width=True)


with tab_risk:
    st.subheader("Risk Analytics")
    charts = getattr(engine, "charts", {}) if isinstance(getattr(engine, "charts", {}), dict) else {}
    abs_var_chart = charts.get("absolute_var")
    rel_var_chart = charts.get("relative_var")
    risk_contrib_chart = charts.get("risk_contrib")
    allocation_chart = charts.get("allocation")

    if abs_var_chart is not None:
        show_plotly(abs_var_chart)
    if rel_var_chart is not None:
        show_plotly(rel_var_chart)
    if risk_contrib_chart is not None:
        show_plotly(risk_contrib_chart)
    if allocation_chart is not None:
        show_plotly(allocation_chart)

    tail_df = prepare_tail_metrics(best_metrics)
    if not tail_df.empty:
        st.dataframe(tail_df, use_container_width=True, hide_index=True)

    risk_contrib_df = safe_df(getattr(engine, "risk_contrib_df", pd.DataFrame()))
    if not risk_contrib_df.empty:
        st.dataframe(risk_contrib_df, use_container_width=True)


with tab_beta:
    st.subheader("Rolling Beta")
    charts = getattr(engine, "charts", {}) if isinstance(getattr(engine, "charts", {}), dict) else {}
    rolling_beta_chart = charts.get("rolling_beta")
    if rolling_beta_chart is not None:
        show_plotly(rolling_beta_chart)

    rolling_beta_df = safe_df(getattr(engine, "rolling_beta_df", pd.DataFrame()))
    beta_summary_df = safe_df(getattr(engine, "beta_summary_df", pd.DataFrame()))

    if not rolling_beta_df.empty:
        st.dataframe(rolling_beta_df.tail(20), use_container_width=True)
    if not beta_summary_df.empty:
        st.dataframe(beta_summary_df, use_container_width=True)


with tab_finquant:
    st.subheader("FinQuant")
    fq = getattr(engine, "finquant_charts", {}) if isinstance(getattr(engine, "finquant_charts", {}), dict) else {}
    fq_chart = fq.get("ef_chart")
    fq_min_tbl = fq.get("min_vol_table")
    fq_max_tbl = fq.get("max_sharpe_table")

    if fq_chart is not None:
        show_plotly(fq_chart)

    c1, c2 = st.columns(2)
    with c1:
        if fq_min_tbl is not None:
            show_plotly(fq_min_tbl)
    with c2:
        if fq_max_tbl is not None:
            show_plotly(fq_max_tbl)


with tab_factors:
    st.subheader("Factor PCA")

    st.markdown("""
### What is PCA (Principal Component Analysis)?

Principal Component Analysis (PCA) is a statistical technique used to identify the **underlying common drivers** of asset returns within a portfolio.

Instead of analyzing each asset individually, PCA reduces the complexity of the portfolio into a smaller number of **independent factors (principal components)** that explain most of the market movements.

### Why is PCA Important in Portfolio Analysis?

PCA helps answer critical institutional questions:

- Is the portfolio truly diversified?
- Are multiple assets actually driven by the same underlying risk factor?
- What are the dominant sources of risk in the portfolio?
- Are hedging assets behaving as expected?

### How to Interpret the Results

**PC1**  
Represents the most important factor and explains the largest share of portfolio variance. This is often interpreted as the broad market or risk-on / risk-off factor.

**PC2**  
Represents the second independent driver of returns. This may reflect interest rate sensitivity, style rotation, or asset class divergence.

**PC3**  
Represents a third independent source of movement, which may capture more specific themes such as commodities, inflation, or defensive behavior.

### Factor Loadings

The values in the PCA table are factor loadings:

- Large positive values: strong positive relationship with the factor
- Large negative values: inverse relationship
- Values near zero: weak relationship

### Explained Variance

If one principal component explains most of the variance, the portfolio may look diversified on paper while still being dominated by a single hidden risk factor.

### Summary

PCA transforms complex portfolio behavior into a smaller set of common risk drivers and helps users understand:

- true diversification,
- hidden concentration,
- structural portfolio exposure.
""")

    st.info("Key Insight: if PC1 explains most of the variance, your portfolio may appear diversified but is actually driven by one dominant factor.")

    factor_df = safe_df(getattr(engine, "factor_pca_df", pd.DataFrame()))
    if not factor_df.empty:
        st.dataframe(factor_df, use_container_width=True)


with tab_data:
    st.subheader("Data & Diagnostics")
    st.write("Selected universe:", run_params["selected_universe"])
    st.write("Universe tickers:", UNIVERSE_REGISTRY.get(run_params["selected_universe"], []))
    st.write("Benchmark:", run_params["benchmark_symbol"])
    st.write("Start date:", run_params["default_start_date"])

    metrics_df = safe_df(getattr(engine, "metrics_df", pd.DataFrame()))
    if not metrics_df.empty:
        st.dataframe(format_metrics_df(metrics_df.copy()), use_container_width=True)

    data_obj = getattr(engine, "data", None)
    if data_obj is not None:
        asset_metadata = safe_df(getattr(data_obj, "asset_metadata", pd.DataFrame()))
        data_quality = safe_df(getattr(data_obj, "data_quality", pd.DataFrame()))
        if not asset_metadata.empty:
            st.markdown("**Asset Metadata**")
            st.dataframe(asset_metadata, use_container_width=True)
        if not data_quality.empty:
            st.markdown("**Data Quality**")
            st.dataframe(data_quality, use_container_width=True)

    if hasattr(engine, "prices") and isinstance(engine.prices, pd.DataFrame) and not engine.prices.empty:
        st.markdown("**Price History Preview**")
        st.dataframe(engine.prices.tail(10), use_container_width=True)

    if hasattr(engine, "returns") and isinstance(engine.returns, pd.DataFrame) and not engine.returns.empty:
        st.markdown("**Return Matrix Preview**")
        st.dataframe(engine.returns.tail(10), use_container_width=True)

    st.markdown("**Diagnostics Summary**")
    st.json(diag if isinstance(diag, dict) else {})

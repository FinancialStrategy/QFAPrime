from __future__ import annotations

import streamlit as st

from core.engine import ProfessionalPortfolioEngine
from ui.sidebar import render_sidebar, render_run_controls, render_black_litterman_controls
from ui.kpis import render_full_kpi_panel
from ui.tables import (
    show_asset_metadata_table,
    show_data_quality_table,
    show_metrics_table,
    show_strategy_table,
)
from ui.charts import StreamlitChartBuilder
from ui.theme import apply_theme


st.set_page_config(
    page_title="QFA Professional Quant Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()

st.title("QFA Professional Quant Platform")
st.caption("Institutional modular Streamlit application")

config = render_sidebar()
bl_controls = render_black_litterman_controls(config)
run_clicked = render_run_controls()

if "engine" not in st.session_state:
    st.session_state["engine"] = None

if "last_run_config" not in st.session_state:
    st.session_state["last_run_config"] = None


def _run_engine() -> None:
    engine = ProfessionalPortfolioEngine(config=config, bl_controls=bl_controls)
    engine.run()
    st.session_state["engine"] = engine
    st.session_state["last_run_config"] = {
        "selected_universe": config.selected_universe,
        "selected_region": config.selected_region,
        "benchmark": config.benchmark,
        "lookback_years": config.lookback_years,
        "risk_free_rate": config.risk_free_rate,
        "min_weight": config.min_weight,
        "max_weight": config.max_weight,
        "max_category_weight": config.max_category_weight,
        "covariance_method": config.covariance_method,
        "expected_return_method": config.expected_return_method,
        "turnover_penalty": config.turnover_penalty,
        "transaction_cost_bps": config.transaction_cost_bps,
        "tracking_error_target": config.tracking_error_target,
        "initial_capital": config.initial_capital,
        "bl_controls": bl_controls,
    }


if run_clicked:
    with st.spinner("Running institutional analysis..."):
        try:
            _run_engine()
            st.success("Analysis completed successfully.")
        except Exception as exc:
            st.session_state["engine"] = None
            st.error(f"Run failed: {exc}")


engine = st.session_state["engine"]

if engine is None:
    st.info(
        "Set your controls in the sidebar and click "
        "**Run Institutional Analysis** to build the full platform outputs."
    )

    st.markdown("### Platform Scope")
    c1, c2, c3 = st.columns(3)
    c1.markdown(
        """
        **Universe Layer**
        - Institutional Multi-Asset
        - Global Major Stock Indices
        - Regional filtering
        """
    )
    c2.markdown(
        """
        **Optimization Layer**
        - Max Sharpe
        - Min Volatility
        - HRP / ERC
        - Black-Litterman
        - Tracking Error Optimization
        """
    )
    c3.markdown(
        """
        **Risk Layer**
        - VaR / CVaR / Relative VaR
        - Stress testing
        - Scenario families
        - Monte Carlo
        - Tracking error analytics
        """
    )

    st.markdown("### Current Configuration Preview")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Universe", config.selected_universe)
    p2.metric("Region", config.selected_region)
    p3.metric("Benchmark", config.benchmark)
    p4.metric("Lookback Years", config.lookback_years)

    st.markdown("### Black-Litterman Preview")
    b1, b2, b3 = st.columns(3)
    b1.metric("BL Enabled", "Yes" if bl_controls.get("enabled") else "No")
    b2.metric("BL View Mode", bl_controls.get("view_mode", "ticker"))
    b3.metric("BL View Count", len(bl_controls.get("views_payload", [])))

else:
    chart_builder = StreamlitChartBuilder(engine.config)
    best_strategy = engine.best_strategy_name()
    best_metrics = engine.metrics[best_strategy]
    tracking_strategy = engine.tracking_error_strategy_name()
    tracking_metrics = engine.metrics[tracking_strategy]

    render_full_kpi_panel(
        best_strategy,
        best_metrics,
        engine.config.initial_capital,
    )

    st.markdown("### Platform Overview")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Executive Snapshot",
            "Universe Preview",
            "Optimization Snapshot",
            "Tracking Error Snapshot",
            "Tables",
        ]
    )

    with tab1:
        st.plotly_chart(
            chart_builder.performance_dashboard(engine.metrics_df),
            use_container_width=True,
        )
        st.plotly_chart(
            chart_builder.equity_curve_chart(
                best_metrics["portfolio_values"],
                best_metrics["benchmark_values"],
                best_strategy,
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            chart_builder.drawdown_chart(
                best_metrics["drawdown_series"],
                best_strategy,
            ),
            use_container_width=True,
        )

    with tab2:
        st.plotly_chart(
            chart_builder.info_hub_table(engine.data.asset_metadata),
            use_container_width=True,
        )

    with tab3:
        st.plotly_chart(
            chart_builder.optimization_chart(
                engine.mu,
                engine.cov,
                engine.strategies,
                engine.config.risk_free_rate,
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            chart_builder.allocation_chart(engine.strategies[best_strategy].weights),
            use_container_width=True,
        )

        bl_strategy_name = None
        for name, obj in engine.strategies.items():
            if obj.method == "black_litterman":
                bl_strategy_name = name
                break

        if bl_strategy_name is not None:
            bl_diag = engine.strategies[bl_strategy_name].diagnostics
            st.markdown("#### BL Snapshot")
            c1, c2, c3 = st.columns(3)
            c1.metric("BL View Mode", bl_diag.get("bl_view_mode", "N/A"))
            c2.metric("BL Views Used", len(bl_diag.get("views_used", {})))
            c3.metric("BL Posterior Cov Trace", f"{bl_diag.get('posterior_covariance_trace', float('nan')):.4f}")

    with tab4:
        st.plotly_chart(
            chart_builder.tracking_error_chart(engine.metrics_df),
            use_container_width=True,
        )
        st.plotly_chart(
            chart_builder.benchmark_vs_tracking_error_curve(
                tracking_metrics["portfolio_returns"],
                tracking_metrics["benchmark_returns"],
                tracking_strategy,
            ),
            use_container_width=True,
        )

    with tab5:
        show_metrics_table(engine.metrics_df)
        show_strategy_table(engine.strategy_df)
        show_asset_metadata_table(engine.data.asset_metadata)
        show_data_quality_table(engine.data.data_quality)

    st.markdown("### Run Summary")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Assets Loaded", len(engine.data.asset_returns.columns))
    s2.metric("Strategies Built", len(engine.strategies))
    s3.metric("Benchmark Used", engine.diagnostics.benchmark_used or engine.config.benchmark)
    s4.metric("Covariance Method", engine.diagnostics.covariance_method_used or engine.config.covariance_method)

    if engine.diagnostics.dropped_assets:
        with st.expander("Dropped Assets"):
            dropped_rows = [
                {"ticker": ticker, "reason": reason}
                for ticker, reason in engine.diagnostics.dropped_assets.items()
            ]
            st.dataframe(dropped_rows, use_container_width=True)

    with st.expander("Diagnostics"):
        st.write(
            {
                "benchmark_used": engine.diagnostics.benchmark_used,
                "covariance_method_used": engine.diagnostics.covariance_method_used,
                "expected_return_method_used": engine.diagnostics.expected_return_method_used,
                "covariance_repaired": engine.diagnostics.covariance_repaired,
                "bl_controls": bl_controls,
            }
        )

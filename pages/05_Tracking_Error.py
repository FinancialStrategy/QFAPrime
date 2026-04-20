from __future__ import annotations

import pandas as pd
import streamlit as st

from core.engine import ProfessionalPortfolioEngine
from ui.charts import StreamlitChartBuilder
from ui.tables import show_dataframe
from ui.theme import apply_theme


apply_theme()
st.title("Tracking Error")

if "engine" not in st.session_state or st.session_state["engine"] is None:
    st.warning("Run the platform from the main app first.")
else:
    engine: ProfessionalPortfolioEngine = st.session_state["engine"]
    chart_builder = StreamlitChartBuilder(engine.config)

    tracking_name = engine.tracking_error_strategy_name()
    tracking_metrics = engine.metrics[tracking_name]
    tracking_weights = engine.strategies[tracking_name].weights

    rolling_te_63 = engine.analytics.rolling_tracking_error(
        tracking_metrics["portfolio_returns"],
        tracking_metrics["benchmark_returns"],
        window=63,
    )

    rolling_ir_63 = engine.analytics.rolling_information_ratio(
        tracking_metrics["portfolio_returns"],
        tracking_metrics["benchmark_returns"],
        window=63,
    )

    region_arc_df = engine.analytics.active_risk_contribution_by_region(
        asset_returns=engine.data.asset_returns,
        weights=tracking_weights,
        benchmark_ticker=engine.config.benchmark,
        asset_metadata=engine.data.asset_metadata,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tracking Error Strategy", tracking_name)
    c2.metric("Ex-Ante Tracking Error", f"{engine.strategies[tracking_name].diagnostics.get('ex_ante_tracking_error', float('nan')):.2%}")
    c3.metric("TE Target", f"{engine.config.tracking_error_target:.2%}")
    c4.metric("Information Ratio", f"{tracking_metrics['information_ratio']:.2f}" if pd.notna(tracking_metrics["information_ratio"]) else "N/A")

    st.plotly_chart(
        chart_builder.tracking_error_chart(engine.metrics_df),
        use_container_width=True,
    )

    st.plotly_chart(
        chart_builder.benchmark_vs_tracking_error_curve(
            tracking_metrics["portfolio_returns"],
            tracking_metrics["benchmark_returns"],
            tracking_name,
        ),
        use_container_width=True,
    )

    st.plotly_chart(
        chart_builder.tracking_error_band_chart(
            rolling_te_63=rolling_te_63,
            te_target=engine.config.tracking_error_target,
            tolerance=0.01,
        ),
        use_container_width=True,
    )

    st.plotly_chart(
        chart_builder.rolling_information_ratio_chart(rolling_ir_63, None),
        use_container_width=True,
    )

    st.plotly_chart(
        chart_builder.active_risk_contribution_region_chart(region_arc_df),
        use_container_width=True,
    )

    show_dataframe("Active Risk Contribution by Region", region_arc_df)

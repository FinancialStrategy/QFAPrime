from __future__ import annotations

import streamlit as st

from core.engine import ProfessionalPortfolioEngine
from ui.charts import StreamlitChartBuilder
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

    st.plotly_chart(chart_builder.tracking_error_chart(engine.metrics_df), use_container_width=True)
    st.plotly_chart(
        chart_builder.benchmark_vs_tracking_error_curve(
            tracking_metrics["portfolio_returns"],
            tracking_metrics["benchmark_returns"],
            tracking_name,
        ),
        use_container_width=True,
    )

    rolling_te = engine.analytics.rolling_tracking_error(
        tracking_metrics["portfolio_returns"],
        tracking_metrics["benchmark_returns"],
        window=63,
    )
    st.line_chart(rolling_te, use_container_width=True)

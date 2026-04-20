from __future__ import annotations

import streamlit as st

from core.engine import ProfessionalPortfolioEngine
from ui.charts import StreamlitChartBuilder
from ui.kpis import render_full_kpi_panel
from ui.theme import apply_theme


apply_theme()
st.title("Executive Dashboard")

if "engine" not in st.session_state or st.session_state["engine"] is None:
    st.warning("Run the platform from the main app first.")
else:
    engine: ProfessionalPortfolioEngine = st.session_state["engine"]
    chart_builder = StreamlitChartBuilder(engine.config)

    best = engine.best_strategy_name()
    best_metrics = engine.metrics[best]

    render_full_kpi_panel(best, best_metrics, engine.config.initial_capital)

    st.plotly_chart(chart_builder.performance_dashboard(engine.metrics_df), use_container_width=True)
    st.plotly_chart(
        chart_builder.equity_curve_chart(
            best_metrics["portfolio_values"],
            best_metrics["benchmark_values"],
            best,
        ),
        use_container_width=True,
    )
    st.plotly_chart(
        chart_builder.drawdown_chart(best_metrics["drawdown_series"], best),
        use_container_width=True,
    )

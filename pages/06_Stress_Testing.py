from __future__ import annotations

import streamlit as st

from core.engine import ProfessionalPortfolioEngine
from ui.charts import StreamlitChartBuilder
from ui.tables import show_stress_table
from ui.theme import apply_theme


apply_theme()
st.title("Historical Stress Testing")

if "engine" not in st.session_state or st.session_state["engine"] is None:
    st.warning("Run the platform from the main app first.")
else:
    engine: ProfessionalPortfolioEngine = st.session_state["engine"]
    chart_builder = StreamlitChartBuilder(engine.config)

    best = engine.best_strategy_name()
    stress_df = engine.historical_stress[best]

    st.plotly_chart(chart_builder.stress_test_chart(stress_df), use_container_width=True)
    show_stress_table(stress_df)

    st.markdown("#### Hypothetical Shock Table")
    st.dataframe(engine.hypothetical_shocks[best], use_container_width=True)

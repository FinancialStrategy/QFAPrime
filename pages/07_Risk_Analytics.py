from __future__ import annotations

import streamlit as st

from core.engine import ProfessionalPortfolioEngine
from ui.charts import StreamlitChartBuilder
from ui.tables import show_metrics_table, show_risk_contribution_table
from ui.theme import apply_theme


apply_theme()
st.title("Risk Analytics")

if "engine" not in st.session_state or st.session_state["engine"] is None:
    st.warning("Run the platform from the main app first.")
else:
    engine: ProfessionalPortfolioEngine = st.session_state["engine"]
    chart_builder = StreamlitChartBuilder(engine.config)

    st.plotly_chart(chart_builder.var_family_chart(engine.metrics_df, kind="absolute"), use_container_width=True)
    st.plotly_chart(chart_builder.var_family_chart(engine.metrics_df, kind="relative"), use_container_width=True)

    best = engine.best_strategy_name()
    show_risk_contribution_table(engine.risk_contributions[best])
    show_metrics_table(engine.metrics_df)

from __future__ import annotations

import streamlit as st

from core.engine import ProfessionalPortfolioEngine
from ui.charts import StreamlitChartBuilder
from ui.theme import apply_theme


apply_theme()
st.title("Monte Carlo")

if "engine" not in st.session_state or st.session_state["engine"] is None:
    st.warning("Run the platform from the main app first.")
else:
    engine: ProfessionalPortfolioEngine = st.session_state["engine"]
    chart_builder = StreamlitChartBuilder(engine.config)

    best = engine.best_strategy_name()
    mc = engine.monte_carlo_results[best]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean Terminal", f"${mc['mean_terminal']:,.0f}")
    c2.metric("Median Terminal", f"${mc['median_terminal']:,.0f}")
    c3.metric("Prob. Loss", f"{mc['prob_loss']:.2%}")
    c4.metric("Prob. Gain", f"{mc['prob_gain']:.2%}")

    st.plotly_chart(chart_builder.monte_carlo_terminal_distribution(mc), use_container_width=True)
    st.plotly_chart(chart_builder.monte_carlo_paths_chart(mc), use_container_width=True)

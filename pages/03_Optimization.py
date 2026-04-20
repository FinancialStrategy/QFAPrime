from __future__ import annotations

import streamlit as st

from core.engine import ProfessionalPortfolioEngine
from ui.charts import StreamlitChartBuilder
from ui.tables import show_strategy_table
from ui.theme import apply_theme


apply_theme()
st.title("Optimization")

if "engine" not in st.session_state or st.session_state["engine"] is None:
    st.warning("Run the platform from the main app first.")
else:
    engine: ProfessionalPortfolioEngine = st.session_state["engine"]
    chart_builder = StreamlitChartBuilder(engine.config)

    st.plotly_chart(
        chart_builder.optimization_chart(
            engine.mu,
            engine.cov,
            engine.strategies,
            engine.config.risk_free_rate,
        ),
        use_container_width=True,
    )

    best = engine.best_strategy_name()
    st.plotly_chart(
        chart_builder.allocation_chart(engine.strategies[best].weights),
        use_container_width=True,
    )

    show_strategy_table(engine.strategy_df)

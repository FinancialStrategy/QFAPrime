from __future__ import annotations

import streamlit as st

from core.engine import ProfessionalPortfolioEngine
from ui.charts import StreamlitChartBuilder
from ui.tables import show_stress_table, show_dataframe
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

    st.markdown("## Detailed Historical Stress Paths")
    path_map = engine.historical_stress_paths.get(best, {})
    if path_map:
        scenario_name = st.selectbox("Select Historical Scenario", list(path_map.keys()))
        selected_path = path_map.get(scenario_name)
        if selected_path is not None and not selected_path.empty:
            st.plotly_chart(
                chart_builder.stress_detail_chart(selected_path, scenario_name),
                use_container_width=True,
            )
            show_dataframe("Selected Stress Path Data", selected_path.tail(50))
    else:
        st.info("No detailed historical stress paths available.")

    st.markdown("## Sharp Fluctuation Windows")
    fluctuation_df = engine.sharp_fluctuation_windows.get(best)
    show_dataframe("Sharp Drop / Rally Detection", fluctuation_df)

    st.markdown("## Hypothetical Shock Table")
    st.dataframe(engine.hypothetical_shocks[best], use_container_width=True)

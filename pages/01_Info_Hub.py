from __future__ import annotations

import streamlit as st

from core.engine import ProfessionalPortfolioEngine
from ui.charts import StreamlitChartBuilder
from ui.tables import show_asset_metadata_table, show_data_quality_table
from ui.theme import apply_theme


apply_theme()
st.title("Info Hub")

if "engine" not in st.session_state or st.session_state["engine"] is None:
    st.warning("Run the platform from the main app first.")
else:
    engine: ProfessionalPortfolioEngine = st.session_state["engine"]
    chart_builder = StreamlitChartBuilder(engine.config)

    st.plotly_chart(chart_builder.info_hub_table(engine.data.asset_metadata), use_container_width=True)
    show_asset_metadata_table(engine.data.asset_metadata)
    show_data_quality_table(engine.data.data_quality)

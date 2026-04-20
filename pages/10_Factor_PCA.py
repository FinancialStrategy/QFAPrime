from __future__ import annotations

import streamlit as st

from core.engine import ProfessionalPortfolioEngine
from ui.charts import StreamlitChartBuilder
from ui.tables import show_dataframe
from ui.theme import apply_theme


apply_theme()
st.title("Factor PCA Interpretation")

if "engine" not in st.session_state or st.session_state["engine"] is None:
    st.warning("Run the platform from the main app first.")
else:
    engine: ProfessionalPortfolioEngine = st.session_state["engine"]
    chart_builder = StreamlitChartBuilder(engine.config)

    pca_results = engine.pca_results
    evr = pca_results.get("explained_variance_ratio")
    loadings = pca_results.get("loadings")
    interpretation = pca_results.get("interpretation", {})

    if evr is None or len(evr) == 0 or loadings is None or loadings.empty:
        st.info("PCA results are not available for this run.")
    else:
        st.plotly_chart(
            chart_builder.pca_explained_variance_chart(evr),
            use_container_width=True,
        )

        st.plotly_chart(
            chart_builder.pca_loadings_heatmap(loadings),
            use_container_width=True,
        )

        show_dataframe("PCA Loadings Table", loadings)

        st.markdown("## Interpretation Notes")
        for pc, note in interpretation.items():
            st.markdown(f"**{pc}:** {note}")

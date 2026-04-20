from __future__ import annotations

import streamlit as st

from core.engine import ProfessionalPortfolioEngine
from exports.html_report import build_html_report
from ui.theme import apply_theme


apply_theme()
st.title("Report Export")

if "engine" not in st.session_state or st.session_state["engine"] is None:
    st.warning("Run the platform from the main app first.")
else:
    engine: ProfessionalPortfolioEngine = st.session_state["engine"]

    html_report = build_html_report(engine)

    st.download_button(
        label="Download HTML Report",
        data=html_report,
        file_name=engine.config.report_file,
        mime="text/html",
        use_container_width=True,
    )

    st.download_button(
        label="Download Metrics CSV",
        data=engine.metrics_df.to_csv(index=True).encode("utf-8"),
        file_name="qfa_metrics.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.download_button(
        label="Download Strategy Details CSV",
        data=engine.strategy_df.to_csv(index=False).encode("utf-8"),
        file_name="qfa_strategy_details.csv",
        mime="text/csv",
        use_container_width=True,
    )

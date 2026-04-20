from __future__ import annotations

import pandas as pd
import streamlit as st

from core.engine import ProfessionalPortfolioEngine
from ui.charts import StreamlitChartBuilder
from ui.theme import apply_theme


apply_theme()
st.title("Benchmark Relative")

if "engine" not in st.session_state or st.session_state["engine"] is None:
    st.warning("Run the platform from the main app first.")
else:
    engine: ProfessionalPortfolioEngine = st.session_state["engine"]
    chart_builder = StreamlitChartBuilder(engine.config)

    benchmark_proxy = pd.Series(0.0, index=engine.mu.index)
    if engine.config.benchmark in benchmark_proxy.index:
        benchmark_proxy[engine.config.benchmark] = 1.0
    elif "SPY" in benchmark_proxy.index:
        benchmark_proxy["SPY"] = 1.0
    else:
        benchmark_proxy[:] = 1 / len(benchmark_proxy)

    st.plotly_chart(
        chart_builder.relative_frontier_chart(
            engine.mu,
            engine.cov,
            engine.strategies,
            benchmark_proxy,
        ),
        use_container_width=True,
    )

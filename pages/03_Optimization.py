from __future__ import annotations

import pandas as pd
import streamlit as st

from core.engine import ProfessionalPortfolioEngine
from ui.charts import StreamlitChartBuilder
from ui.tables import show_strategy_table, show_dataframe
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

    st.markdown("## Black-Litterman Posterior Analysis")

    bl_strategy_name = None
    for name, obj in engine.strategies.items():
        if obj.method == "black_litterman":
            bl_strategy_name = name
            break

    if bl_strategy_name is None:
        st.info("Black-Litterman strategy was not available in this run.")
    else:
        bl_result = engine.strategies[bl_strategy_name]
        bl_diag = bl_result.diagnostics

        views_used = bl_diag.get("views_used", {})
        conf_used = bl_diag.get("view_confidences_used", {})
        prior_returns = pd.Series(bl_diag.get("prior_returns", {}), name="prior")
        posterior_returns = pd.Series(bl_diag.get("posterior_returns", {}), name="posterior")
        bl_weights = pd.Series(bl_diag.get("bl_weight_output", {}), name="posterior_weight").sort_values(ascending=False)

        if not prior_returns.empty and not posterior_returns.empty:
            prior_posterior_df = pd.concat([prior_returns, posterior_returns], axis=1)
            prior_posterior_df["delta"] = prior_posterior_df["posterior"] - prior_posterior_df["prior"]
            prior_posterior_df = prior_posterior_df.sort_values("posterior", ascending=False)

            st.plotly_chart(
                chart_builder.prior_vs_posterior_return_chart(
                    prior_returns=prior_returns,
                    posterior_returns=posterior_returns,
                ),
                use_container_width=True,
            )

            show_dataframe("Prior vs Posterior Returns Table", prior_posterior_df)

        if not bl_weights.empty:
            show_dataframe("BL Posterior Weights", bl_weights.to_frame())

        if engine.bl_prior_returns is not None and engine.bl_posterior_returns is not None and engine.bl_posterior_cov is not None and engine.bl_weights is not None:
            st.plotly_chart(
                chart_builder.posterior_frontier_chart(
                    prior_returns=engine.bl_prior_returns,
                    posterior_returns=engine.bl_posterior_returns,
                    posterior_cov=engine.bl_posterior_cov,
                    bl_weights=engine.bl_weights,
                ),
                use_container_width=True,
            )

        views_table = pd.DataFrame(
            [
                {
                    "target": k,
                    "expected_return_view": v,
                    "confidence": conf_used.get(k, None),
                }
                for k, v in views_used.items()
            ]
        )
        show_dataframe("BL Views Used", views_table)

        st.markdown("### BL Diagnostics")
        c1, c2, c3 = st.columns(3)
        c1.metric("BL View Mode", bl_diag.get("bl_view_mode", "N/A"))
        c2.metric("Number of Views", len(views_used))
        c3.metric("Posterior Cov Trace", f"{bl_diag.get('posterior_covariance_trace', float('nan')):.4f}")

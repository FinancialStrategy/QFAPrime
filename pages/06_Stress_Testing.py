from __future__ import annotations

import pandas as pd
import streamlit as st

from core.engine import ProfessionalPortfolioEngine
from core.scenarios import rank_scenario_severity, summarize_scenario_families
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
    raw_stress_df = engine.historical_stress[best]
    stress_df = engine.filter_stress_dataframe(raw_stress_df)

    family_summary_df = summarize_scenario_families(stress_df)
    ranked_df = rank_scenario_severity(stress_df)

    # ============================================================
    # STRESS DASHBOARD KPI PANEL
    # ============================================================
    st.markdown("## Stress Dashboard")

    if stress_df is None or stress_df.empty:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Worst Historical Scenario", "N/A")
        c2.metric("Average Severity Score", "N/A")
        c3.metric("Worst Relative Return", "N/A")
        c4.metric("Worst Portfolio Drawdown", "N/A")
        c5.metric("Scenarios Passing Filters", "0")
    else:
        worst_scenario_row = stress_df.loc[stress_df["severity_score"].idxmax()]
        avg_severity = float(stress_df["severity_score"].mean())
        worst_relative_return = float(stress_df["relative_return"].min())
        worst_portfolio_drawdown = float(stress_df["portfolio_max_drawdown"].min())
        scenario_count = int(len(stress_df))

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Worst Historical Scenario", str(worst_scenario_row["scenario"]))
        c2.metric("Average Severity Score", f"{avg_severity:.3f}")
        c3.metric("Worst Relative Return", f"{worst_relative_return:.2%}")
        c4.metric("Worst Portfolio Drawdown", f"{worst_portfolio_drawdown:.2%}")
        c5.metric("Scenarios Passing Filters", f"{scenario_count}")

    # ============================================================
    # ACTIVE FILTERS
    # ============================================================
    st.markdown("## Active Scenario Filters")
    f1, f2, f3 = st.columns(3)
    f1.metric("Family Filter", engine.scenario_controls.get("selected_family", "All"))
    f2.metric("Min Severity", f"{engine.scenario_controls.get('minimum_severity_threshold', 0.0):.2f}")
    f3.metric("Quick View", engine.scenario_controls.get("quick_view", "All"))

    # ============================================================
    # SCENARIO FAMILY SUMMARY
    # ============================================================
    st.markdown("## Scenario Family Summary")
    st.plotly_chart(
        chart_builder.scenario_family_summary_chart(family_summary_df),
        use_container_width=True,
    )
    show_dataframe("Scenario Family Summary Table", family_summary_df)

    # ============================================================
    # HISTORICAL STRESS OVERVIEW
    # ============================================================
    st.markdown("## Historical Stress Overview")
    st.plotly_chart(chart_builder.stress_test_chart(stress_df), use_container_width=True)
    show_stress_table(stress_df)

    # ============================================================
    # EVENT SEVERITY RANKING
    # ============================================================
    st.markdown("## Event Severity Ranking")
    st.plotly_chart(
        chart_builder.scenario_severity_ranking_chart(ranked_df, top_n=10),
        use_container_width=True,
    )
    show_dataframe("Severity Ranking Table", ranked_df)

    # ============================================================
    # DETAILED HISTORICAL STRESS PATHS
    # ============================================================
    st.markdown("## Detailed Historical Stress Paths")
    path_map = engine.historical_stress_paths.get(best, {})
    available_scenarios = [s for s in stress_df["scenario"].tolist() if s in path_map]

    if available_scenarios:
        scenario_name = st.selectbox("Select Historical Scenario", available_scenarios)
        selected_path = path_map.get(scenario_name)

        if selected_path is not None and not selected_path.empty:
            st.plotly_chart(
                chart_builder.stress_detail_chart(selected_path, scenario_name),
                use_container_width=True,
            )
            show_dataframe("Selected Stress Path Data", selected_path.tail(50))

            row_match = stress_df.loc[stress_df["scenario"] == scenario_name]
            if not row_match.empty:
                row = row_match.iloc[0]
                region_impact_df = engine.analytics.scenario_regional_impact_decomposition(
                    asset_returns=engine.data.asset_returns,
                    weights=engine.strategies[best].weights,
                    asset_metadata=engine.data.asset_metadata,
                    start_date=row["start_date"],
                    end_date=row["end_date"],
                )

                st.markdown("## Regional Scenario Impact Decomposition")
                st.plotly_chart(
                    chart_builder.regional_scenario_impact_chart(region_impact_df, scenario_name),
                    use_container_width=True,
                )
                show_dataframe("Regional Scenario Impact Table", region_impact_df)
    else:
        st.info("No detailed historical stress paths available under the current scenario filters.")

    # ============================================================
    # SHARP FLUCTUATION WINDOWS
    # ============================================================
    st.markdown("## Sharp Fluctuation Windows")
    fluctuation_df = engine.sharp_fluctuation_windows.get(best)
    show_dataframe("Sharp Drop / Rally Detection", fluctuation_df)

    # ============================================================
    # HYPOTHETICAL SHOCK TABLE
    # ============================================================
    st.markdown("## Hypothetical Shock Table")
    st.dataframe(engine.hypothetical_shocks[best], use_container_width=True)

    # ============================================================
    # SCENARIO EXPORT TABLES
    # ============================================================
    st.markdown("## Scenario Export Tables")
    export_choice = st.selectbox(
        "Select Export Table",
        [
            "Historical Stress",
            "Scenario Family Summary",
            "Severity Ranking",
            "Hypothetical Shocks",
            "Sharp Fluctuation Windows",
        ],
    )

    export_map = {
        "Historical Stress": stress_df,
        "Scenario Family Summary": family_summary_df,
        "Severity Ranking": ranked_df,
        "Hypothetical Shocks": engine.hypothetical_shocks[best],
        "Sharp Fluctuation Windows": fluctuation_df,
    }
    export_df = export_map[export_choice]

    if export_df is not None and not export_df.empty:
        st.download_button(
            label=f"Download {export_choice} CSV",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{export_choice.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

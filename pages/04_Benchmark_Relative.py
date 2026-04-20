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

    best = engine.best_strategy_name()
    best_metrics = engine.metrics[best]

    benchmark_proxy = pd.Series(0.0, index=engine.mu.index)
    if engine.config.benchmark in benchmark_proxy.index:
        benchmark_proxy[engine.config.benchmark] = 1.0
    elif "SPY" in benchmark_proxy.index:
        benchmark_proxy["SPY"] = 1.0
    else:
        benchmark_proxy[:] = 1 / len(benchmark_proxy)

    rel_dd_df = engine.analytics.relative_drawdown_series(
        best_metrics["portfolio_returns"],
        best_metrics["benchmark_returns"],
    )

    rolling_beta_63 = engine.analytics.rolling_beta(
        best_metrics["portfolio_returns"],
        best_metrics["benchmark_returns"],
        window=63,
    )
    rolling_beta_126 = engine.analytics.rolling_beta(
        best_metrics["portfolio_returns"],
        best_metrics["benchmark_returns"],
        window=126,
        min_periods=60,
    )

    rolling_sharpe_63 = engine.analytics.rolling_sharpe(
        best_metrics["portfolio_returns"],
        window=63,
    )
    rolling_sharpe_126 = engine.analytics.rolling_sharpe(
        best_metrics["portfolio_returns"],
        window=126,
        min_periods=60,
    )

    rolling_ir_63 = engine.analytics.rolling_information_ratio(
        best_metrics["portfolio_returns"],
        best_metrics["benchmark_returns"],
        window=63,
    )
    rolling_ir_126 = engine.analytics.rolling_information_ratio(
        best_metrics["portfolio_returns"],
        best_metrics["benchmark_returns"],
        window=126,
        min_periods=60,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Alpha", f"{best_metrics['alpha']:.2%}" if pd.notna(best_metrics["alpha"]) else "N/A")
    c2.metric("Beta", f"{best_metrics['beta']:.2f}" if pd.notna(best_metrics["beta"]) else "N/A")
    c3.metric("Information Ratio", f"{best_metrics['information_ratio']:.2f}" if pd.notna(best_metrics["information_ratio"]) else "N/A")
    c4.metric("Excess Return vs Benchmark", f"{best_metrics['excess_return_vs_benchmark_pct']:.2%}")

    st.plotly_chart(
        chart_builder.relative_frontier_chart(
            engine.mu,
            engine.cov,
            engine.strategies,
            benchmark_proxy,
        ),
        use_container_width=True,
    )

    st.plotly_chart(
        chart_builder.relative_drawdown_chart(rel_dd_df, best),
        use_container_width=True,
    )

    st.plotly_chart(
        chart_builder.rolling_beta_chart(rolling_beta_63, rolling_beta_126),
        use_container_width=True,
    )

    st.plotly_chart(
        chart_builder.rolling_sharpe_chart(rolling_sharpe_63, rolling_sharpe_126),
        use_container_width=True,
    )

    st.plotly_chart(
        chart_builder.rolling_information_ratio_chart(rolling_ir_63, rolling_ir_126),
        use_container_width=True,
    )

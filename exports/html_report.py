from __future__ import annotations

import pandas as pd
import plotly.io as pio

from core.engine import ProfessionalPortfolioEngine
from ui.charts import StreamlitChartBuilder


def _table_html(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "<p>No data available</p>"
    return df.round(4).to_html(classes="data-table", border=0)


def build_html_report(engine: ProfessionalPortfolioEngine) -> str:
    chart_builder = StreamlitChartBuilder(engine.config)

    best = engine.best_strategy_name()
    best_metrics = engine.metrics[best]
    tracking_name = engine.tracking_error_strategy_name()
    tracking_metrics = engine.metrics[tracking_name]

    import pandas as pd
    benchmark_proxy = pd.Series(0.0, index=engine.mu.index)
    if engine.config.benchmark in benchmark_proxy.index:
        benchmark_proxy[engine.config.benchmark] = 1.0
    elif "SPY" in benchmark_proxy.index:
        benchmark_proxy["SPY"] = 1.0
    else:
        benchmark_proxy[:] = 1 / len(benchmark_proxy)

    charts = {
        "info_hub": chart_builder.info_hub_table(engine.data.asset_metadata),
        "dashboard": chart_builder.performance_dashboard(engine.metrics_df),
        "equity": chart_builder.equity_curve_chart(
            best_metrics["portfolio_values"],
            best_metrics["benchmark_values"],
            best,
        ),
        "drawdown": chart_builder.drawdown_chart(best_metrics["drawdown_series"], best),
        "optimization": chart_builder.optimization_chart(
            engine.mu,
            engine.cov,
            engine.strategies,
            engine.config.risk_free_rate,
        ),
        "frontier_relative": chart_builder.relative_frontier_chart(
            engine.mu,
            engine.cov,
            engine.strategies,
            benchmark_proxy,
        ),
        "tracking_error": chart_builder.tracking_error_chart(engine.metrics_df),
        "benchmark_vs_te": chart_builder.benchmark_vs_tracking_error_curve(
            tracking_metrics["portfolio_returns"],
            tracking_metrics["benchmark_returns"],
            tracking_name,
        ),
        "stress": chart_builder.stress_test_chart(engine.historical_stress[best]),
        "var_abs": chart_builder.var_family_chart(engine.metrics_df, kind="absolute"),
        "var_rel": chart_builder.var_family_chart(engine.metrics_df, kind="relative"),
        "allocation": chart_builder.allocation_chart(engine.strategies[best].weights),
        "mc_terminal": chart_builder.monte_carlo_terminal_distribution(engine.monte_carlo_results[best]),
        "mc_paths": chart_builder.monte_carlo_paths_chart(engine.monte_carlo_results[best]),
    }

    chart_html = {k: pio.to_html(v, include_plotlyjs=False, full_html=False) for k, v in charts.items()}

    best_final_value = float(engine.metrics_df.iloc[0]["final_portfolio_value"]) if not engine.metrics_df.empty else engine.config.initial_capital
    best_return = float(engine.metrics_df.iloc[0]["total_return_pct"]) if not engine.metrics_df.empty else 0.0
    best_sharpe = f"{float(engine.metrics_df.iloc[0]['sharpe_ratio']):.2f}" if not engine.metrics_df.empty else "0.00"

    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset='utf-8'>
<title>QFA Professional Quant Platform</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
* {{
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}}
body {{
  font-family: "Segoe UI", Arial, sans-serif;
  background: #f4f6f8;
  color: #23313f;
  font-size: 11px;
  line-height: 1.35;
}}
.header {{
  background: linear-gradient(135deg, #2e3f4f 0%, #43596d 100%);
  color: white;
  padding: 24px 34px;
}}
.header h1 {{
  font-size: 24px;
  font-weight: 600;
}}
.header p {{
  margin-top: 8px;
  font-size: 11px;
}}
.container {{
  max-width: 1640px;
  margin: 0 auto;
  padding: 18px 22px;
}}
.kpi-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
  gap: 14px;
  margin-bottom: 18px;
}}
.kpi-card {{
  background: white;
  border: 1px solid #dde3e8;
  border-radius: 10px;
  padding: 14px 16px;
}}
.kpi-label {{
  font-size: 10px;
  text-transform: uppercase;
  color: #75808b;
  margin-bottom: 6px;
}}
.kpi-value {{
  font-size: 20px;
  font-weight: 600;
}}
.kpi-sub {{
  font-size: 10px;
  color: #6e7781;
  margin-top: 6px;
}}
.chart-card {{
  background: white;
  border: 1px solid #dde3e8;
  border-radius: 10px;
  margin-bottom: 16px;
  overflow: hidden;
}}
.chart-header {{
  padding: 10px 14px;
  border-bottom: 1px solid #e5eaef;
  background: #f8fafb;
}}
.chart-header h3 {{
  font-size: 13px;
  font-weight: 600;
}}
.chart-body {{
  padding: 12px 14px;
}}
.data-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 10px;
}}
.data-table th {{
  background: #32485d;
  color: white;
  padding: 8px 9px;
  text-align: left;
  font-size: 10px;
}}
.data-table td {{
  padding: 7px 9px;
  border-bottom: 1px solid #e4e8ec;
}}
.footer {{
  background: #2f3e4c;
  color: white;
  text-align: center;
  padding: 16px;
  border-radius: 10px;
  margin-top: 18px;
  font-size: 10px;
}}
</style>
</head>
<body>
<div class="header">
  <h1>QFA Professional Quant Platform</h1>
  <p>
    Initial Capital: ${engine.config.initial_capital:,.0f} |
    Analysis Period: {engine.config.start_date} to {engine.config.end_date} |
    Benchmark: {engine.diagnostics.benchmark_used}
  </p>
</div>

<div class="container">
  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-label">Best Strategy</div>
      <div class="kpi-value">{best}</div>
      <div class="kpi-sub">Sharpe: {best_sharpe}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Final Portfolio Value</div>
      <div class="kpi-value">${best_final_value:,.0f}</div>
      <div class="kpi-sub">Return: {best_return:.2%}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Initial Capital</div>
      <div class="kpi-value">${engine.config.initial_capital:,.0f}</div>
      <div class="kpi-sub">Risk-Free Rate: {engine.config.risk_free_rate:.2%}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Strategies</div>
      <div class="kpi-value">{len(engine.metrics_df)}</div>
      <div class="kpi-sub">Dropped assets: {len(engine.diagnostics.dropped_assets)}</div>
    </div>
  </div>

  <div class="chart-card"><div class="chart-header"><h3>Info Hub</h3></div><div class="chart-body">{chart_html['info_hub']}</div></div>
  <div class="chart-card"><div class="chart-header"><h3>Executive Dashboard</h3></div><div class="chart-body">{chart_html['dashboard']}</div></div>
  <div class="chart-card"><div class="chart-header"><h3>Portfolio vs Benchmark Equity Curve</h3></div><div class="chart-body">{chart_html['equity']}</div></div>
  <div class="chart-card"><div class="chart-header"><h3>Drawdown Analysis</h3></div><div class="chart-body">{chart_html['drawdown']}</div></div>
  <div class="chart-card"><div class="chart-header"><h3>Optimization</h3></div><div class="chart-body">{chart_html['optimization']}</div></div>
  <div class="chart-card"><div class="chart-header"><h3>Benchmark-Relative Frontier</h3></div><div class="chart-body">{chart_html['frontier_relative']}</div></div>
  <div class="chart-card"><div class="chart-header"><h3>Tracking Error</h3></div><div class="chart-body">{chart_html['tracking_error']}</div></div>
  <div class="chart-card"><div class="chart-header"><h3>Benchmark vs Tracking Error Optimal</h3></div><div class="chart-body">{chart_html['benchmark_vs_te']}</div></div>
  <div class="chart-card"><div class="chart-header"><h3>Historical Stress Testing</h3></div><div class="chart-body">{chart_html['stress']}</div></div>
  <div class="chart-card"><div class="chart-header"><h3>VaR / CVaR</h3></div><div class="chart-body">{chart_html['var_abs']}</div></div>
  <div class="chart-card"><div class="chart-header"><h3>Relative VaR</h3></div><div class="chart-body">{chart_html['var_rel']}</div></div>
  <div class="chart-card"><div class="chart-header"><h3>Allocation</h3></div><div class="chart-body">{chart_html['allocation']}</div></div>
  <div class="chart-card"><div class="chart-header"><h3>Monte Carlo Terminal Distribution</h3></div><div class="chart-body">{chart_html['mc_terminal']}</div></div>
  <div class="chart-card"><div class="chart-header"><h3>Monte Carlo Paths</h3></div><div class="chart-body">{chart_html['mc_paths']}</div></div>

  <div class="chart-card"><div class="chart-header"><h3>Performance Metrics</h3></div><div class="chart-body">{_table_html(engine.metrics_df)}</div></div>
  <div class="chart-card"><div class="chart-header"><h3>Strategy Details</h3></div><div class="chart-body">{_table_html(engine.strategy_df)}</div></div>
  <div class="chart-card"><div class="chart-header"><h3>Data Quality</h3></div><div class="chart-body">{_table_html(engine.data.data_quality)}</div></div>

</div>

<div class="footer">
  <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</div>
</body>
</html>
"""

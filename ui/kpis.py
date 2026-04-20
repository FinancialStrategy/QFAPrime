from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd
import streamlit as st


def _fmt_pct(x: float) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"{x:.2%}"


def _fmt_num(x: float) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"{x:,.2f}"


def _fmt_money(x: float) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"${x:,.0f}"


def render_kpi_band_portfolio_status(best_strategy: str, metrics: Dict[str, Any], initial_capital: float) -> None:
    st.markdown("#### Portfolio Status")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Strategy", best_strategy)
    c2.metric("Initial Capital", _fmt_money(initial_capital))
    c3.metric("Final Value", _fmt_money(metrics.get("final_portfolio_value")))
    c4.metric("Total P&L", _fmt_money(metrics.get("total_profit_loss")))


def render_kpi_band_absolute(metrics: Dict[str, Any]) -> None:
    st.markdown("#### Absolute Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Annual Return", _fmt_pct(metrics.get("annual_return")))
    c2.metric("Volatility", _fmt_pct(metrics.get("volatility")))
    c3.metric("Sharpe", _fmt_num(metrics.get("sharpe_ratio")))
    c4.metric("Sortino", _fmt_num(metrics.get("sortino_ratio")))


def render_kpi_band_relative(metrics: Dict[str, Any]) -> None:
    st.markdown("#### Relative Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Alpha", _fmt_pct(metrics.get("alpha")))
    c2.metric("Beta", _fmt_num(metrics.get("beta")))
    c3.metric("Tracking Error", _fmt_pct(metrics.get("tracking_error")))
    c4.metric("Information Ratio", _fmt_num(metrics.get("information_ratio")))


def render_kpi_band_tail(metrics: Dict[str, Any]) -> None:
    st.markdown("#### Tail / Protection")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Max Drawdown", _fmt_pct(metrics.get("max_drawdown")))
    c2.metric("VaR 95", _fmt_pct(metrics.get("var_95")))
    c3.metric("CVaR 95", _fmt_pct(metrics.get("cvar_95")))
    c4.metric("Relative VaR 95", _fmt_pct(metrics.get("relative_var_95")))


def render_full_kpi_panel(best_strategy: str, metrics: Dict[str, Any], initial_capital: float) -> None:
    render_kpi_band_portfolio_status(best_strategy, metrics, initial_capital)
    st.markdown("")
    render_kpi_band_absolute(metrics)
    st.markdown("")
    render_kpi_band_relative(metrics)
    st.markdown("")
    render_kpi_band_tail(metrics)

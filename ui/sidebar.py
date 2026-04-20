from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from core.config import ProfessionalConfig
from core.universes import get_available_regions, get_universe_metadata


def render_sidebar() -> ProfessionalConfig:
    st.sidebar.header("Platform Controls")

    selected_universe = st.sidebar.selectbox(
        "Universe Type",
        ["Institutional Multi-Asset", "Global Major Stock Indices"],
        index=0,
    )

    available_regions = get_available_regions(selected_universe)
    selected_region = st.sidebar.selectbox(
        "Region Filter",
        available_regions,
        index=0,
    )

    benchmark = st.sidebar.text_input("Benchmark", value="SPY")
    lookback_years = st.sidebar.slider("Lookback Years", min_value=3, max_value=15, value=6, step=1)
    risk_free_rate = st.sidebar.slider("Risk-Free Rate", min_value=0.00, max_value=0.10, value=0.0425, step=0.0025)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Portfolio Constraints")

    min_weight = st.sidebar.slider("Min Weight", min_value=0.00, max_value=0.10, value=0.00, step=0.01)
    max_weight = st.sidebar.slider("Max Weight", min_value=0.05, max_value=0.50, value=0.20, step=0.01)
    max_category_weight = st.sidebar.slider("Max Category Weight", min_value=0.10, max_value=0.80, value=0.35, step=0.01)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Settings")

    covariance_method = st.sidebar.selectbox(
        "Covariance Method",
        ["ledoit_wolf", "sample", "shrinkage"],
        index=0,
    )

    expected_return_method = st.sidebar.selectbox(
        "Expected Return Method",
        ["ema_historical", "historical_mean", "capm"],
        index=0,
    )

    turnover_penalty = st.sidebar.slider("Turnover Penalty", min_value=0.000, max_value=0.050, value=0.005, step=0.001)
    transaction_cost_bps = st.sidebar.slider("Transaction Cost (bps)", min_value=0.0, max_value=50.0, value=10.0, step=1.0)
    tracking_error_target = st.sidebar.slider("Tracking Error Target", min_value=0.01, max_value=0.20, value=0.06, step=0.01)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Capital")

    initial_capital = st.sidebar.number_input(
        "Initial Capital",
        min_value=100_000.0,
        max_value=10_000_000_000.0,
        value=10_000_000.0,
        step=100_000.0,
        format="%.2f",
    )

    config = ProfessionalConfig(
        selected_universe=selected_universe,
        selected_region=selected_region,
        benchmark=benchmark,
        lookback_years=lookback_years,
        risk_free_rate=risk_free_rate,
        min_weight=min_weight,
        max_weight=max_weight,
        max_category_weight=max_category_weight,
        covariance_method=covariance_method,
        expected_return_method=expected_return_method,
        turnover_penalty=turnover_penalty,
        transaction_cost_bps=transaction_cost_bps,
        tracking_error_target=tracking_error_target,
        initial_capital=initial_capital,
    )

    return config


def render_black_litterman_controls(config: ProfessionalConfig) -> Dict[str, Any]:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Black-Litterman Controls")

    enable_bl_views = st.sidebar.checkbox("Enable Custom BL Views", value=False)
    view_mode = st.sidebar.selectbox("View Scope", ["region", "ticker"], index=0)

    views_payload: List[Dict[str, Any]] = []

    if enable_bl_views:
        max_views = 5
        num_views = st.sidebar.slider("Number of Views", min_value=1, max_value=max_views, value=2, step=1)

        if view_mode == "region":
            candidates = sorted(
                list(
                    {
                        row["region_type"]
                        for row in get_universe_metadata(config.selected_universe, "All")
                    }
                )
            )
        else:
            candidates = sorted(
                [row["ticker"] for row in get_universe_metadata(config.selected_universe, config.selected_region)]
            )

        for i in range(num_views):
            st.sidebar.markdown(f"**View {i+1}**")
            target = st.sidebar.selectbox(
                f"Target {i+1}",
                candidates,
                index=min(i, len(candidates) - 1) if candidates else 0,
                key=f"bl_target_{i}",
            )
            expected_return = st.sidebar.slider(
                f"Expected Annual Return {i+1}",
                min_value=-0.10,
                max_value=0.20,
                value=0.06,
                step=0.01,
                key=f"bl_return_{i}",
            )
            confidence = st.sidebar.slider(
                f"Confidence {i+1}",
                min_value=0.05,
                max_value=1.00,
                value=0.60,
                step=0.05,
                key=f"bl_confidence_{i}",
            )

            views_payload.append(
                {
                    "target": target,
                    "expected_return": expected_return,
                    "confidence": confidence,
                }
            )

    return {
        "enabled": enable_bl_views,
        "view_mode": view_mode,
        "views_payload": views_payload,
    }


def render_run_controls() -> bool:
    return st.sidebar.button("Run Institutional Analysis", type="primary", use_container_width=True)

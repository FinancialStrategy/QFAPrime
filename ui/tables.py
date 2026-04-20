from __future__ import annotations

import pandas as pd
import streamlit as st


def show_dataframe(title: str, df: pd.DataFrame, use_container_width: bool = True) -> None:
    st.markdown(f"#### {title}")
    if df is None or df.empty:
        st.info("No data available.")
        return
    st.dataframe(df, use_container_width=use_container_width)


def show_metrics_table(metrics_df: pd.DataFrame) -> None:
    if metrics_df is None or metrics_df.empty:
        st.info("No performance metrics available.")
        return

    display_cols = [
        "annual_return",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "max_drawdown",
        "tracking_error",
        "information_ratio",
        "var_95",
        "cvar_95",
        "relative_var_95",
        "relative_cvar_95",
        "total_return_pct",
        "final_portfolio_value",
        "total_profit_loss",
    ]
    cols = [c for c in display_cols if c in metrics_df.columns]
    show_dataframe("Performance Metrics", metrics_df[cols])


def show_strategy_table(strategy_df: pd.DataFrame) -> None:
    show_dataframe("Strategy Details", strategy_df)


def show_data_quality_table(data_quality_df: pd.DataFrame) -> None:
    show_dataframe("Data Quality", data_quality_df)


def show_asset_metadata_table(asset_metadata_df: pd.DataFrame) -> None:
    show_dataframe("Investment Universe Metadata", asset_metadata_df)


def show_risk_contribution_table(risk_contrib_df: pd.DataFrame) -> None:
    show_dataframe("Risk Contribution", risk_contrib_df)


def show_stress_table(stress_df: pd.DataFrame) -> None:
    show_dataframe("Historical Stress Testing Table", stress_df)

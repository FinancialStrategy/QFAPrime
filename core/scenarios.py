from __future__ import annotations

from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd


SCENARIO_LIBRARY: Dict[str, Dict[str, Tuple[str, str]]] = {
    "Crisis": {
        "Global Financial Crisis Lehman Phase": ("2008-09-01", "2009-03-09"),
        "COVID Crash": ("2020-02-19", "2020-03-23"),
        "Eurozone Stress Window": ("2011-07-01", "2011-10-04"),
    },
    "Inflation": {
        "2022 Inflation Shock": ("2022-01-03", "2022-10-14"),
        "US Rate Shock Window": ("2022-08-01", "2022-10-31"),
    },
    "Banking_Stress": {
        "2023 Banking Stress": ("2023-03-08", "2023-03-31"),
        "Regional Bank Volatility Burst": ("2023-03-01", "2023-03-24"),
    },
    "Sharp_Rally": {
        "COVID Rebound": ("2020-03-24", "2020-08-31"),
        "2024 Q1 Rally": ("2024-01-02", "2024-03-28"),
        "Sharp One-Month Rebound": ("2023-11-01", "2023-11-30"),
    },
    "Sharp_Selloff": {
        "Sharp One-Month Selloff": ("2022-04-01", "2022-04-30"),
        "Autumn 2018 Risk-Off": ("2018-10-01", "2018-12-24"),
    },
}

HYPOTHETICAL_SHOCKS = {
    "Equity Shock -10%": -0.10,
    "Equity Shock -15%": -0.15,
    "Equity Shock -20%": -0.20,
    "Equity Shock -25%": -0.25,
    "Equity Shock -30%": -0.30,
    "Sharp Rally +10%": 0.10,
    "Sharp Rally +15%": 0.15,
}


def flatten_scenario_library() -> Dict[str, Dict[str, str]]:
    rows = {}
    for family, scenarios in SCENARIO_LIBRARY.items():
        for scenario_name, (start_date, end_date) in scenarios.items():
            rows[scenario_name] = {
                "family": family,
                "start_date": start_date,
                "end_date": end_date,
            }
    return rows


def run_historical_stress_tests(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    scenario_library: Dict[str, Dict[str, Tuple[str, str]]] = SCENARIO_LIBRARY,
) -> pd.DataFrame:
    rows = []
    pr = portfolio_returns.copy()
    br = benchmark_returns.copy()

    for family, family_scenarios in scenario_library.items():
        for name, (s, e) in family_scenarios.items():
            mask = (pr.index >= pd.Timestamp(s)) & (pr.index <= pd.Timestamp(e))
            p = pr.loc[mask]
            b = br.reindex(p.index).dropna()
            p = p.reindex(b.index)

            if len(p) < 5:
                continue

            p_total = float((1 + p).prod() - 1)
            b_total = float((1 + b).prod() - 1)

            p_vol = float(p.std() * np.sqrt(252)) if len(p) > 1 else np.nan
            b_vol = float(b.std() * np.sqrt(252)) if len(b) > 1 else np.nan

            p_path = (1 + p).cumprod()
            b_path = (1 + b).cumprod()

            p_dd = float((p_path / p_path.cummax() - 1).min())
            b_dd = float((b_path / b_path.cummax() - 1).min())

            rows.append({
                "family": family,
                "scenario": name,
                "start_date": s,
                "end_date": e,
                "portfolio_return": p_total,
                "benchmark_return": b_total,
                "relative_return": p_total - b_total,
                "portfolio_volatility": p_vol,
                "benchmark_volatility": b_vol,
                "portfolio_max_drawdown": p_dd,
                "benchmark_max_drawdown": b_dd,
                "duration_days": len(p),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["severity_score"] = (
        out["portfolio_return"].abs().fillna(0.0) * 0.40
        + out["portfolio_max_drawdown"].abs().fillna(0.0) * 0.35
        + out["portfolio_volatility"].fillna(0.0) * 0.25
    )
    out = out.sort_values(["severity_score", "portfolio_return"], ascending=[False, True]).reset_index(drop=True)
    return out


def rank_scenario_severity(stress_df: pd.DataFrame) -> pd.DataFrame:
    if stress_df is None or stress_df.empty:
        return pd.DataFrame()

    ranked = stress_df.copy().sort_values("severity_score", ascending=False).reset_index(drop=True)
    ranked["severity_rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def summarize_scenario_families(stress_df: pd.DataFrame) -> pd.DataFrame:
    if stress_df is None or stress_df.empty:
        return pd.DataFrame()

    summary = (
        stress_df.groupby("family", as_index=False)
        .agg(
            scenario_count=("scenario", "count"),
            avg_portfolio_return=("portfolio_return", "mean"),
            worst_portfolio_return=("portfolio_return", "min"),
            avg_relative_return=("relative_return", "mean"),
            worst_drawdown=("portfolio_max_drawdown", "min"),
            avg_severity_score=("severity_score", "mean"),
        )
        .sort_values("avg_severity_score", ascending=False)
    )
    return summary


def extract_scenario_path(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    aligned.columns = ["portfolio", "benchmark"]

    mask = (aligned.index >= pd.Timestamp(start_date)) & (aligned.index <= pd.Timestamp(end_date))
    x = aligned.loc[mask].copy()

    if x.empty:
        return pd.DataFrame()

    x["portfolio_cum"] = (1 + x["portfolio"]).cumprod() - 1
    x["benchmark_cum"] = (1 + x["benchmark"]).cumprod() - 1
    x["relative_cum"] = x["portfolio_cum"] - x["benchmark_cum"]
    return x


def detect_sharp_fluctuation_windows(
    returns: pd.Series,
    window: int = 21,
    top_n: int = 5,
) -> pd.DataFrame:
    x = returns.dropna().copy()
    if len(x) < window + 5:
        return pd.DataFrame()

    rolling_total = (1 + x).rolling(window).apply(np.prod, raw=True) - 1
    rolling_vol = x.rolling(window).std() * np.sqrt(252)

    largest_drops = rolling_total.nsmallest(top_n)
    largest_rallies = rolling_total.nlargest(top_n)

    rows = []
    for date, value in largest_drops.items():
        rows.append({
            "event_type": "sharp_drop",
            "end_date": date,
            "window_days": window,
            "window_return": float(value),
            "window_volatility": float(rolling_vol.loc[date]) if date in rolling_vol.index else np.nan,
        })

    for date, value in largest_rallies.items():
        rows.append({
            "event_type": "sharp_rally",
            "end_date": date,
            "window_days": window,
            "window_return": float(value),
            "window_volatility": float(rolling_vol.loc[date]) if date in rolling_vol.index else np.nan,
        })

    out = pd.DataFrame(rows).sort_values(["event_type", "window_return"])
    return out


def run_hypothetical_shocks(weights: dict, shocks: Dict[str, float] = HYPOTHETICAL_SHOCKS) -> pd.DataFrame:
    rows = []
    gross_weight = sum(abs(v) for v in weights.values())

    for name, shock in shocks.items():
        rows.append({
            "scenario": name,
            "shock": shock,
            "approx_portfolio_impact": gross_weight * shock,
        })

    return pd.DataFrame(rows)

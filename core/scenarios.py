from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd


DEFAULT_HISTORICAL_SCENARIOS: Dict[str, Tuple[str, str]] = {
    "COVID Crash": ("2020-02-19", "2020-03-23"),
    "2022 Inflation Shock": ("2022-01-03", "2022-10-14"),
    "2023 Banking Stress": ("2023-03-08", "2023-03-31"),
    "2024 Q1 Rally": ("2024-01-02", "2024-03-28"),
}

HYPOTHETICAL_SHOCKS = {
    "Equity Shock -10%": -0.10,
    "Equity Shock -15%": -0.15,
    "Equity Shock -20%": -0.20,
    "Equity Shock -25%": -0.25,
    "Equity Shock -30%": -0.30,
}


def run_historical_stress_tests(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    scenarios: Dict[str, Tuple[str, str]] = DEFAULT_HISTORICAL_SCENARIOS,
) -> pd.DataFrame:
    rows = []
    pr = portfolio_returns.copy()
    br = benchmark_returns.copy()

    for name, (s, e) in scenarios.items():
        mask = (pr.index >= pd.Timestamp(s)) & (pr.index <= pd.Timestamp(e))
        p = pr.loc[mask]
        b = br.reindex(p.index).dropna()
        p = p.reindex(b.index)

        if len(p) < 5:
            continue

        p_total = float((1 + p).prod() - 1)
        b_total = float((1 + b).prod() - 1)
        rows.append({
            "scenario": name,
            "portfolio_return": p_total,
            "benchmark_return": b_total,
            "relative_return": p_total - b_total,
            "duration_days": len(p),
        })

    return pd.DataFrame(rows)


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

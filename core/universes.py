from __future__ import annotations

from typing import Dict, List

# -------------------------------------------------------------------
# Final universe registry for QFA Prime
# -------------------------------------------------------------------
# Notes:
# - Every universe must contain at least 2 assets.
# - Keep universes compact to reduce Yahoo Finance throttling on Render.
# - Use liquid Yahoo Finance tickers only.
# -------------------------------------------------------------------

UNIVERSE_REGISTRY: Dict[str, List[str]] = {
    "institutional_multi_asset": [
        "SPY",      # US Equities
        "QQQ",      # Nasdaq
        "GLD",      # Gold
        "TLT",      # Long US Treasuries
        "IEF",      # Intermediate US Treasuries
        "DBC",      # Broad Commodities
        "VNQ",      # REITs
    ],
    "global_equities": [
        "SPY",      # US
        "VEA",      # Developed ex-US
        "VWO",      # Emerging Markets
        "EWJ",      # Japan
        "EWU",      # UK
    ],
    "defensive_allocation": [
        "TLT",
        "IEF",
        "GLD",
        "TIP",
        "VNQ",
    ],
    "inflation_hedge": [
        "GLD",
        "DBC",
        "TIP",
        "XLE",
    ],
    "growth_allocation": [
        "QQQ",
        "SPY",
        "VUG",
        "SOXX",
    ],
    "balanced_60_40_plus": [
        "SPY",
        "IEF",
        "TLT",
        "GLD",
        "VNQ",
    ],
    "major_indices_proxy": [
        "SPY",
        "QQQ",
        "IWM",
        "DIA",
    ],
    "precious_metals": [
        "GLD",
        "SLV",
        "PPLT",
    ],
    "real_assets": [
        "GLD",
        "DBC",
        "VNQ",
        "TIP",
    ],
    "risk_on_risk_off": [
        "QQQ",
        "SPY",
        "TLT",
        "GLD",
        "DBC",
    ],
}


def list_universes() -> List[str]:
    return list(UNIVERSE_REGISTRY.keys())


def get_universe_tickers(universe_name: str) -> List[str]:
    raw = UNIVERSE_REGISTRY.get(universe_name, [])
    tickers = [str(x).strip().upper() for x in raw if str(x).strip()]
    tickers = list(dict.fromkeys(tickers))
    return tickers

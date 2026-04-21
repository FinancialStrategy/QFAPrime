from __future__ import annotations

from typing import Dict, List

UNIVERSE_REGISTRY: Dict[str, List[str]] = {
    "institutional_multi_asset": [
        # US Equity Core
        "SPY", "QQQ", "IWM", "DIA", "VTI",

        # International Equities
        "VEA", "VWO", "EWJ", "EWU", "EEM",

        # Fixed Income
        "TLT", "IEF", "BND", "LQD", "TIP", "HYG",

        # Real Assets / Alternatives
        "GLD", "SLV", "DBC", "VNQ", "XLE",

        # Defensive / Style / Sector Spread
        "XLV", "XLF", "XLI", "XLP", "XLK"
    ],

    "global_equities": [
        "SPY", "QQQ", "VTI", "VEA", "VWO", "EWJ", "EWU", "EEM", "DIA", "IWM"
    ],

    "defensive_allocation": [
        "TLT", "IEF", "BND", "TIP", "LQD", "GLD", "VNQ", "XLP", "XLV"
    ],

    "inflation_hedge": [
        "GLD", "SLV", "DBC", "TIP", "XLE", "VNQ"
    ],

    "growth_allocation": [
        "QQQ", "SPY", "VUG", "SOXX", "XLK", "IWM"
    ],

    "balanced_60_40_plus": [
        "SPY", "VTI", "IEF", "TLT", "BND", "GLD", "VNQ", "TIP"
    ],

    "major_indices_proxy": [
        "SPY", "QQQ", "IWM", "DIA", "VEA", "VWO"
    ],

    "precious_metals": [
        "GLD", "SLV", "PPLT"
    ],

    "real_assets": [
        "GLD", "DBC", "VNQ", "TIP", "XLE"
    ],

    "risk_on_risk_off": [
        "QQQ", "SPY", "IWM", "TLT", "IEF", "GLD", "DBC", "XLP"
    ],

    "sector_rotation": [
        "XLK", "XLF", "XLV", "XLE", "XLI", "XLP", "XLY", "XLU"
    ],

    "institutional_broad_25": [
        "SPY", "QQQ", "IWM", "DIA", "VTI",
        "VEA", "VWO", "EWJ", "EWU", "EEM",
        "TLT", "IEF", "BND", "LQD", "TIP",
        "GLD", "SLV", "DBC", "VNQ", "XLE",
        "XLK", "XLV", "XLF", "XLI", "XLP"
    ],
}


def list_universes() -> List[str]:
    return list(UNIVERSE_REGISTRY.keys())


def get_universe_tickers(universe_name: str) -> List[str]:
    raw = UNIVERSE_REGISTRY.get(universe_name, [])
    tickers = [str(x).strip().upper() for x in raw if str(x).strip()]
    return list(dict.fromkeys(tickers))

from __future__ import annotations

from typing import Dict, List

UNIVERSE_REGISTRY: Dict[str, List[str]] = {
    "institutional_multi_asset": [
        "SPY",
        "QQQ",
        "GLD",
        "TLT",
        "IEF",
        "DBC",
        "VNQ",
    ],
    "global_equities": [
        "SPY",
        "VEA",
        "VWO",
        "EWJ",
        "EWU",
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

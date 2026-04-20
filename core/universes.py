from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


INSTITUTIONAL_MULTI_ASSET_UNIVERSE: Dict[str, Dict[str, str]] = {
    "US_Equities": {
        "SPDR S&P 500 ETF": "SPY",
        "Invesco QQQ Trust": "QQQ",
        "iShares Russell 2000 ETF": "IWM",
        "Vanguard Total Stock Market ETF": "VTI",
        "SPDR Dow Jones Industrial Average ETF": "DIA",
    },
    "International_Equities": {
        "Vanguard FTSE Developed Markets ETF": "VEA",
        "Vanguard FTSE Emerging Markets ETF": "VWO",
        "iShares MSCI Japan ETF": "EWJ",
        "iShares MSCI Eurozone ETF": "EZU",
    },
    "Fixed_Income": {
        "Vanguard Total Bond Market ETF": "BND",
        "iShares 20+ Year Treasury ETF": "TLT",
        "iShares 7-10 Year Treasury ETF": "IEF",
        "iShares iBoxx $ Investment Grade Corporate Bond ETF": "LQD",
        "iShares National Muni Bond ETF": "MUB",
        "iShares iBoxx $ High Yield Corporate Bond ETF": "HYG",
    },
    "Real_Assets": {
        "SPDR Gold Shares": "GLD",
        "iShares Silver Trust": "SLV",
        "Vanguard Real Estate ETF": "VNQ",
        "iShares S&P GSCI Commodity-Indexed Trust": "GSG",
        "Invesco DB Base Metals Fund": "DBB",
    },
    "Sectors": {
        "Technology Select Sector SPDR Fund": "XLK",
        "Financial Select Sector SPDR Fund": "XLF",
        "Health Care Select Sector SPDR Fund": "XLV",
        "Energy Select Sector SPDR Fund": "XLE",
        "Industrial Select Sector SPDR Fund": "XLI",
        "Consumer Staples Select Sector SPDR Fund": "XLP",
        "Consumer Discretionary Select Sector SPDR Fund": "XLY",
        "Utilities Select Sector SPDR Fund": "XLU",
    },
}


GLOBAL_MAJOR_INDICES_30: Dict[str, Dict[str, str]] = {
    "North_America": {
        "S&P 500": "^GSPC",
        "Nasdaq 100": "^NDX",
        "Dow Jones Industrial Average": "^DJI",
        "Russell 2000": "^RUT",
        "S&P/TSX Composite": "^GSPTSE",
        "Mexico IPC": "^MXX",
    },
    "Europe": {
        "BIST 100": "XU100.IS",
        "Euro Stoxx 50": "^STOXX50E",
        "FTSE 100": "^FTSE",
        "DAX": "^GDAXI",
        "CAC 40": "^FCHI",
        "AEX": "^AEX",
        "SMI": "^SSMI",
        "IBEX 35": "^IBEX",
        "FTSE MIB": "FTSEMIB.MI",
    },
    "Asia_Pacific": {
        "Nikkei 225": "^N225",
        "TOPIX": "^TOPX",
        "Hang Seng": "^HSI",
        "Shanghai Composite": "000001.SS",
        "CSI 300": "000300.SS",
        "Shenzhen Component": "399001.SZ",
        "KOSPI": "^KS11",
        "ASX 200": "^AXJO",
        "Nifty 50": "^NSEI",
        "Straits Times": "^STI",
    },
    "Emerging_Markets": {
        "Bovespa": "^BVSP",
        "MSCI Emerging Markets Proxy": "EEM",
        "MSCI Frontier Markets Proxy": "FM",
        "MSCI Turkey Proxy": "TUR",
        "South Africa Proxy": "EZA",
    },
}


UNIVERSE_REGISTRY: Dict[str, Dict[str, Dict[str, str]]] = {
    "institutional_multi_asset": INSTITUTIONAL_MULTI_ASSET_UNIVERSE,
    "global_major_indices_30": GLOBAL_MAJOR_INDICES_30,
    "global_multi_asset": INSTITUTIONAL_MULTI_ASSET_UNIVERSE,
    "major_indices": GLOBAL_MAJOR_INDICES_30,
}


def _infer_asset_class(bucket_name: str, display_name: str, ticker: str) -> str:
    bucket = str(bucket_name).lower()
    ticker = str(ticker).upper()

    if bucket == "us_equities":
        return "US Equities"
    if bucket == "international_equities":
        return "International Equities"
    if bucket == "fixed_income":
        return "Fixed Income"
    if bucket == "real_assets":
        return "Real Assets"
    if bucket == "sectors":
        return "Sectors"
    if bucket in {"north_america", "europe", "asia_pacific", "emerging_markets"}:
        return "Equity Indices"
    if ticker.endswith("-USD"):
        return "Crypto"
    if ticker.endswith("=X"):
        return "FX"
    return "Other"


def _infer_region(bucket_name: str) -> str:
    mapping = {
        "US_Equities": "US",
        "International_Equities": "International",
        "Fixed_Income": "US",
        "Real_Assets": "Global",
        "Sectors": "US",
        "North_America": "North America",
        "Europe": "Europe",
        "Asia_Pacific": "Asia Pacific",
        "Emerging_Markets": "Emerging Markets",
    }
    return mapping.get(bucket_name, "Unknown")


def get_available_universes() -> List[str]:
    return list(UNIVERSE_REGISTRY.keys())


def get_universe_definition(selected_universe: str) -> Dict[str, Dict[str, str]]:
    return UNIVERSE_REGISTRY.get(selected_universe, INSTITUTIONAL_MULTI_ASSET_UNIVERSE)


def flatten_universe_dict(universe_definition: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
    flat: Dict[str, Dict[str, Any]] = {}

    for bucket_name, instrument_map in universe_definition.items():
        if not isinstance(instrument_map, dict):
            continue

        for display_name, ticker in instrument_map.items():
            ticker_str = str(ticker).strip()
            display_name_str = str(display_name).strip()

            if not ticker_str:
                continue

            flat[ticker_str] = {
                "ticker": ticker_str,
                "name": display_name_str,
                "bucket": bucket_name,
                "asset_class": _infer_asset_class(bucket_name, display_name_str, ticker_str),
                "region_type": _infer_region(bucket_name),
                "category": bucket_name,
            }

    return flat


def get_universe_metadata(selected_universe: str) -> pd.DataFrame:
    universe_definition = get_universe_definition(selected_universe)
    flat = flatten_universe_dict(universe_definition)

    if not flat:
        return pd.DataFrame(
            columns=["ticker", "name", "bucket", "asset_class", "region_type", "category"]
        )

    return pd.DataFrame(list(flat.values())).sort_values(
        ["bucket", "name"],
        ascending=[True, True],
    ).reset_index(drop=True)


def get_available_regions(selected_universe: str) -> List[str]:
    meta = get_universe_metadata(selected_universe)
    if meta.empty or "region_type" not in meta.columns:
        return ["All"]

    regions = sorted([r for r in meta["region_type"].dropna().astype(str).unique().tolist() if r])
    return ["All"] + regions

from __future__ import annotations

from typing import Dict, List


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
    return rows

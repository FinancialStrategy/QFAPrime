import random
import time
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
import yfinance as yf


def _normalize_close_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize yfinance output to a clean close-price DataFrame.
    Supports both single-ticker and multi-ticker downloads.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        level0 = [str(c[0]).lower() for c in df.columns]
        if "adj close" in level0:
            out = df.xs("Adj Close", axis=1, level=0, drop_level=True).copy()
        elif "close" in level0:
            out = df.xs("Close", axis=1, level=0, drop_level=True).copy()
        else:
            return pd.DataFrame()
    else:
        cols_lower = {str(c).lower(): c for c in df.columns}
        if "adj close" in cols_lower:
            out = df[[cols_lower["adj close"]]].copy()
        elif "close" in cols_lower:
            out = df[[cols_lower["close"]]].copy()
        else:
            return pd.DataFrame()

    if isinstance(out, pd.Series):
        out = out.to_frame()

    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def _chunk_list(items: List[str], chunk_size: int) -> List[List[str]]:
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def _single_ticker_download(
    ticker: str,
    start: str,
    end: str,
    auto_adjust: bool = False,
    timeout: int = 30,
    max_retries: int = 4,
    pause_seconds: float = 2.0,
) -> pd.DataFrame:
    """
    Download one ticker with retry/backoff.
    Returns a single-column DataFrame named by ticker.
    """
    for attempt in range(max_retries):
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=auto_adjust,
                threads=False,
                timeout=timeout,
            )

            px = _normalize_close_frame(raw)
            if not px.empty:
                if px.shape[1] == 1:
                    px.columns = [ticker]
                return px

        except Exception as exc:
            print(
                f"[single_ticker_download] ticker={ticker} "
                f"attempt={attempt + 1}/{max_retries} error={exc}"
            )

        sleep_time = pause_seconds * (attempt + 1) + random.uniform(0.5, 1.25)
        time.sleep(sleep_time)

    return pd.DataFrame()


def _batch_download(
    tickers: List[str],
    start: str,
    end: str,
    auto_adjust: bool = False,
    timeout: int = 30,
    max_retries: int = 3,
    pause_seconds: float = 2.0,
) -> pd.DataFrame:
    """
    Download a small batch of tickers with retry/backoff.
    """
    joined = " ".join(tickers)

    for attempt in range(max_retries):
        try:
            raw = yf.download(
                joined,
                start=start,
                end=end,
                progress=False,
                auto_adjust=auto_adjust,
                threads=False,
                group_by="column",
                timeout=timeout,
            )

            px = _normalize_close_frame(raw)
            if not px.empty:
                existing = [c for c in tickers if c in px.columns]
                if existing:
                    return px[existing].copy()
                return px

        except Exception as exc:
            print(
                f"[batch_download] tickers={tickers} "
                f"attempt={attempt + 1}/{max_retries} error={exc}"
            )

        sleep_time = pause_seconds * (attempt + 1) + random.uniform(0.5, 1.25)
        time.sleep(sleep_time)

    return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def load_price_data(
    tickers: List[str],
    start: str,
    end: str,
    auto_adjust: bool = False,
    batch_size: int = 3,
    min_non_na_ratio: float = 0.70,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Robust Yahoo Finance loader for Render/Streamlit deployments.

    Returns:
        prices: cleaned close-price DataFrame
        meta: {
            "requested": [...],
            "downloaded": [...],
            "failed": [...]
        }
    """
    tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
    tickers = list(dict.fromkeys(tickers))

    if not tickers:
        return pd.DataFrame(), {"requested": [], "downloaded": [], "failed": []}

    all_frames: List[pd.DataFrame] = []
    downloaded = set()

    # Step 1: small-batch download
    for chunk in _chunk_list(tickers, batch_size):
        df_chunk = _batch_download(
            tickers=chunk,
            start=start,
            end=end,
            auto_adjust=auto_adjust,
        )

        if not df_chunk.empty:
            available_cols = [c for c in chunk if c in df_chunk.columns]
            if available_cols:
                all_frames.append(df_chunk[available_cols].copy())
                downloaded.update(available_cols)

        time.sleep(random.uniform(1.0, 2.2))

    # Step 2: fallback one-by-one for missed tickers
    missing = [t for t in tickers if t not in downloaded]
    for ticker in missing:
        df_one = _single_ticker_download(
            ticker=ticker,
            start=start,
            end=end,
            auto_adjust=auto_adjust,
        )
        if not df_one.empty:
            all_frames.append(df_one)
            downloaded.add(ticker)

        time.sleep(random.uniform(1.0, 2.2))

    if not all_frames:
        return pd.DataFrame(), {
            "requested": tickers,
            "downloaded": [],
            "failed": tickers,
        }

    prices = pd.concat(all_frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated(keep="last")]

    final_cols = [t for t in tickers if t in prices.columns]
    prices = prices[final_cols].copy()

    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]

    valid_cols = []
    for col in prices.columns:
        ratio = prices[col].notna().mean()
        if ratio >= min_non_na_ratio:
            valid_cols.append(col)

    prices = prices[valid_cols].copy()
    prices = prices.ffill(limit=3)
    prices = prices.dropna(how="all")

    downloaded_final = list(prices.columns)
    failed = [t for t in tickers if t not in downloaded_final]

    meta = {
        "requested": tickers,
        "downloaded": downloaded_final,
        "failed": failed,
    }
    return prices, meta


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    if prices is None or prices.empty:
        return pd.DataFrame()

    returns = prices.pct_change()
    returns = returns.replace([float("inf"), float("-inf")], pd.NA)
    returns = returns.dropna(how="all")
    return returns

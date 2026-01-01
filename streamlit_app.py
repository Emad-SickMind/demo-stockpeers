# -*- coding: utf-8 -*-
# Copyright 2024-2025 Streamlit Inc.
# Licensed under the Apache License, Version 2.0

import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# ----------------------------
# 0) Top market-cap cryptos (Yahoo Finance tickers)
# ----------------------------
# (Common top market-cap set: BTC, ETH, USDT, BNB, XRP, USDC, SOL, TRX, DOGE, ADA)
CRYPTO_TOP10 = [
    "BTC-USD",
    "ETH-USD",
    "USDT-USD",
    "BNB-USD",
    "XRP-USD",
    "USDC-USD",
    "SOL-USD",
    "TRX-USD",
    "DOGE-USD",
    "ADA-USD",
]


# ----------------------------
# 1) Safe download (Retry + Cache)
# ----------------------------
@st.cache_data(ttl=3600, show_spinner=False)  # 1 hour cache
def safe_download(tickers: list[str], period: str) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance with retry to reduce rate-limit issues.
    Returns a DataFrame as yfinance.download output.
    """
    tickers_str = " ".join(tickers)  # yfinance accepts space separated tickers
    last_err = None

    for i in range(3):  # 3 tries
        try:
            df = yf.download(
                tickers=tickers_str,
                period=period,
                group_by="ticker",
                auto_adjust=True,
                threads=False,  # less aggressive -> fewer requests
                progress=False,
            )

            if df is None or df.empty:
                raise ValueError("Empty data returned from Yahoo Finance.")

            return df

        except Exception as e:
            last_err = e
            time.sleep(2 * (i + 1))  # 2s, 4s, 6s

    raise last_err


def extract_close(download_df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """
    Convert yfinance download output into a DataFrame of Close prices:
    index: Date
    columns: tickers
    """
    # If single ticker, yfinance returns columns like: Open High Low Close Volume
    if not isinstance(download_df.columns, pd.MultiIndex):
        if "Close" not in download_df.columns:
            raise ValueError("Downloaded data has no 'Close' column.")
        close = download_df[["Close"]].copy()
        close.columns = [tickers[0]]
        close.index.name = "Date"
        return close

    # Multi-ticker case with group_by="ticker":
    # columns: (TICKER, field) e.g. ('AAPL','Close')
    close_cols = {}
    level0 = download_df.columns.get_level_values(0)
    for t in tickers:
        if t in level0:
            if "Close" in download_df[t].columns:
                close_cols[t] = download_df[t]["Close"]

    if not close_cols:
        raise ValueError("No Close data found for selected tickers.")

    close = pd.DataFrame(close_cols)
    close.index.name = "Date"
    return close


# ----------------------------
# 2) Page config
# ----------------------------
st.set_page_config(
    page_title="Stock/Crypto peer analysis dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

"""
# :material/query_stats: Stock/Crypto peer analysis

Easily compare stocks and top market-cap crypto assets.
"""

""  # Add some space.

cols = st.columns([1, 3])
# Will declare right cell later to avoid showing it when no data.

STOCKS = [
    "AAPL", "ABBV", "ACN", "ADBE", "ADP", "AMD", "AMGN", "AMT", "AMZN", "APD", "AVGO",
    "AXP", "BA", "BK", "BKNG", "BMY", "BRK.B", "BSX", "C", "CAT", "CI", "CL", "CMCSA",
    "COST", "CRM", "CSCO", "CVX", "DE", "DHR", "DIS", "DUK", "ELV", "EOG", "EQR", "FDX",
    "GD", "GE", "GILD", "GOOG", "GOOGL", "HD", "HON", "HUM", "IBM", "ICE", "INTC", "ISRG",
    "JNJ", "JPM", "KO", "LIN", "LLY", "LMT", "LOW", "MA", "MCD", "MDLZ", "META", "MMC",
    "MO", "MRK", "MSFT", "NEE", "NFLX", "NKE", "NOW", "NVDA", "ORCL", "PEP", "PFE", "PG",
    "PLD", "PM", "PSA", "REGN", "RTX", "SBUX", "SCHW", "SLB", "SO", "SPGI", "T", "TJX",
    "TMO", "TSLA", "TXN", "UNH", "UNP", "UPS", "V", "VZ", "WFC", "WM", "WMT", "XOM",
]

# برای اینکه Rate-limit کمتر بخوری، پیش‌فرض رو کم‌تر گذاشتیم:
DEFAULT_ASSETS = ["AAPL", "MSFT", "NVDA", "BTC-USD", "ETH-USD"]  # هم سهام هم کریپتو (کم)


def assets_to_str(assets: list[str]) -> str:
    return ",".join(assets)


if "tickers_input" not in st.session_state:
    st.session_state.tickers_input = st.query_params.get(
        "stocks", assets_to_str(DEFAULT_ASSETS)
    ).split(",")


top_left_cell = cols[0].container(
    border=True, height="stretch", vertical_alignment="center"
)

ALL_OPTIONS = sorted(set(STOCKS) | set(CRYPTO_TOP10) | set(st.session_state.tickers_input))

with top_left_cell:
    tickers = st.multiselect(
        "Assets (stocks + crypto)",
        options=ALL_OPTIONS,
        default=st.session_state.tickers_input,
        placeholder="Choose assets to compare. Example: NVDA, BTC-USD",
        accept_new_options=True,
    )

# Time horizon selector
horizon_map = {
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "5 Years": "5y",
    "10 Years": "10y",
    "20 Years": "20y",
}

with top_left_cell:
    horizon = st.pills(
        "Time horizon",
        options=list(horizon_map.keys()),
        default="1 Month",  # برای کاهش Rate-limit پیش‌فرض رو کوتاه کردیم
    )

tickers = [t.upper() for t in tickers]

# Update query param (kept name "stocks" for backward compatibility)
if tickers:
    st.query_params["stocks"] = assets_to_str(tickers)
else:
    st.query_params.pop("stocks", None)

if not tickers:
    top_left_cell.info("Pick some assets to compare", icon=":material/info:")
    st.stop()

right_cell = cols[1].container(
    border=True, height="stretch", vertical_alignment="center"
)


# ----------------------------
# 3) Load close prices (uses safe_download)
# ----------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def load_close_prices(tickers: list[str], period: str) -> pd.DataFrame:
    raw = safe_download(tickers, period)
    close = extract_close(raw, tickers)
    if close is None or close.empty:
        raise RuntimeError("No close price data returned.")
    return close


# Load the data (with friendly handling)
try:
    data = load_close_prices(tickers, horizon_map[horizon])

except yf.exceptions.YFRateLimitError:
    st.warning("Yahoo Finance الان Rate limit کرده (Too Many Requests).")
    st.info("برای دمو: تعداد نمادها رو کم کن (مثلاً 1-2 تا) و بازه رو کوتاه بگذار (1 Month).")
    st.stop()

except Exception as e:
    st.error("Error while fetching data.")
    st.caption(f"Details: {e}")
    st.stop()


# Check empty columns (all NaN)
empty_columns = data.columns[data.isna().all()].tolist()
if empty_columns:
    st.error(f"Error loading data for the tickers: {', '.join(empty_columns)}.")
    st.stop()

# Normalize prices (start at 1)
normalized = data.div(data.iloc[0])

# Best/Worst based on latest normalized value (more robust)
latest = normalized.iloc[-1]
best_ticker = latest.idxmax()
worst_ticker = latest.idxmin()
best_val = float(latest.max())
worst_val = float(latest.min())

best_delta_pct = (best_val - 1.0) * 100.0
worst_delta_pct = (worst_val - 1.0) * 100.0

bottom_left_cell = cols[0].container(
    border=True, height="stretch", vertical_alignment="center"
)

with bottom_left_cell:
    mcols = st.columns(2)
    mcols[0].metric(
        "Best asset",
        best_ticker,
        delta=f"{best_delta_pct:.1f}%",
        width="content",
    )
    mcols[1].metric(
        "Worst asset",
        worst_ticker,
        delta=f"{worst_delta_pct:.1f}%",
        width="content",
    )

# Plot normalized prices
with right_cell:
    st.altair_chart(
        alt.Chart(
            normalized.reset_index().melt(
                id_vars=["Date"], var_name="Asset", value_name="Normalized price"
            )
        )
        .mark_line()
        .encode(
            alt.X("Date:T"),
            alt.Y("Normalized price:Q").scale(zero=False),
            alt.Color("Asset:N"),
        )
        .properties(height=400),
        use_container_width=True,
    )

""  # spacer
""  # spacer

# Plot individual asset vs peer average
"""
## Individual assets vs peer average

For the analysis below, the "peer average" when analyzing asset X always
excludes X itself.
"""

if len(tickers) <= 1:
    st.warning("Pick 2 or more assets to compare them")
    st.stop()

NUM_COLS = 4
cols2 = st.columns(NUM_COLS)

for i, ticker in enumerate(tickers):
    peers = normalized.drop(columns=[ticker])
    peer_avg = peers.mean(axis=1)

    plot_data = pd.DataFrame(
        {
            "Date": normalized.index,
            ticker: normalized[ticker],
            "Peer average": peer_avg,
        }
    ).melt(id_vars=["Date"], var_name="Series", value_name="Price")

    chart = (
        alt.Chart(plot_data)
        .mark_line()
        .encode(
            alt.X("Date:T"),
            alt.Y("Price:Q").scale(zero=False),
            alt.Color(
                "Series:N",
                scale=alt.Scale(domain=[ticker, "Peer average"], range=["red", "gray"]),
                legend=alt.Legend(orient="bottom"),
            ),
            alt.Tooltip(["Date", "Series", "Price"]),
        )
        .properties(title=f"{ticker} vs peer average", height=300)
    )

    cell = cols2[(i * 2) % NUM_COLS].container(border=True)
    cell.write("")
    cell.altair_chart(chart, use_container_width=True)

    # Delta chart
    plot_data2 = pd.DataFrame(
        {
            "Date": normalized.index,
            "Delta": normalized[ticker] - peer_avg,
        }
    )

    chart2 = (
        alt.Chart(plot_data2)
        .mark_area()
        .encode(
            alt.X("Date:T"),
            alt.Y("Delta:Q").scale(zero=False),
        )
        .properties(title=f"{ticker} minus peer average", height=300)
    )

    cell = cols2[(i * 2 + 1) % NUM_COLS].container(border=True)
    cell.write("")
    cell.altair_chart(chart2, use_container_width=True)

""  # spacer
""  # spacer

"""
## Raw data
"""

data

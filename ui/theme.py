from __future__ import annotations

import streamlit as st


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        .main > div {
            padding-top: 1rem;
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 1.2rem;
            padding-left: 1.4rem;
            padding-right: 1.4rem;
        }
        h1, h2, h3, h4 {
            color: #22303d;
            letter-spacing: 0.1px;
        }
        .stMetric {
            background-color: #ffffff;
            border: 1px solid #dde3e8;
            padding: 0.7rem 0.9rem;
            border-radius: 0.65rem;
            box-shadow: 0 1px 6px rgba(15, 23, 42, 0.03);
        }
        .stMetric label {
            font-size: 0.72rem !important;
        }
        .stMetric div[data-testid="stMetricValue"] {
            font-size: 1.18rem !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.3rem;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #eef2f5;
            border-radius: 999px;
            padding-left: 0.8rem;
            padding-right: 0.8rem;
            height: 2rem;
            font-size: 0.78rem;
        }
        .stTabs [aria-selected="true"] {
            background-color: #41576c !important;
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

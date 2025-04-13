# tariff_app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
from typing import Optional, Dict, Any, Callable, List

# --- Configuration ---
st.set_page_config(
    layout="wide", page_title="QUANTIFI // Tariff & Derivatives", initial_sidebar_state="expanded",
    menu_items={'About': "Advanced Tariff Impact & Derivatives Analyzer. For educational purposes only."}
)

# --- Constants ---
MIN_PRICE_LEVEL: float = 1e-6

# --- Import Core Modules ---
from sidebar_config import render_sidebar
from utils import (
    get_stock_price, get_option_expiries,
    SentimentIntensityAnalyzer, NewsApiClient,
    RISK_FREE_RATE_SERIES,
    DOLTHUB_OWNER, DOLTHUB_REPO,
    DEFAULT_TICKER, DEFAULT_S_FALLBACK
)
from tabs.tab_live_analysis import render_tab_live_analysis
from tabs.tab_vol_smile import render_tab_vol_smile
from tabs.tab_sensitivity import render_tab_sensitivity
from tabs.tab_stress_test import render_tab_stress_test
from tabs.tab_backtest import render_tab_backtest
from tabs.tab_3d_surface import render_tab_3d_surface
from tabs.tab_news import render_tab_news
from tabs.tab_explain import render_tab_explain

# --- Secrets Management ---
NEWS_API_KEY: Optional[str] = None
# FIX: Removed FRED_API_KEY handling
# FRED_API_KEY: Optional[str] = None
NEWS_API_ENABLED: bool = False
try:
    # FRED_API_KEY = st.secrets.get("FRED_API_KEY") # Removed
    if "NEWS_API_KEY" in st.secrets:
        NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
        NEWS_API_ENABLED = bool(NEWS_API_KEY)
    else: st.sidebar.warning("NewsAPI Key not found in secrets.", icon="⚠️")
except FileNotFoundError: st.error("Secrets file not found.", icon="🔥")
except Exception as e: st.error(f"Secrets loading error: {e}", icon="🔥")
if not NEWS_API_ENABLED: st.sidebar.warning("NewsAPI disabled.", icon="📰")
# FIX: Removed warning for FRED API Key
# if not FRED_API_KEY: st.sidebar.warning("FRED API Key missing.", icon="🔑")

# --- Global Initializations ---
sentiment_analyzer = SentimentIntensityAnalyzer()
newsapi_client: Optional[NewsApiClient] = None
if NEWS_API_ENABLED and NEWS_API_KEY:
    try: newsapi_client = NewsApiClient(api_key=NEWS_API_KEY)
    except Exception as e: st.error(f"NewsAPI client init failed: {e}"); NEWS_API_ENABLED = False

# --- Initialize Session State ---
if 'sigma_base' not in st.session_state: st.session_state.sigma_base = 0.20
if 'current_S' not in st.session_state: st.session_state.current_S = None
if 'current_r' not in st.session_state: st.session_state.current_r = 0.05
if 'news_df' not in st.session_state: st.session_state.news_df = pd.DataFrame()
if 'news_status' not in st.session_state: st.session_state.news_status = "News not fetched yet."
if 'news_keywords_processed' not in st.session_state: st.session_state.news_keywords_processed = ""
if 'smile_df' not in st.session_state: st.session_state.smile_df = pd.DataFrame()
if 'backtest_results_df' not in st.session_state: st.session_state.backtest_results_df = pd.DataFrame()
if 'historical_rates' not in st.session_state: st.session_state.historical_rates = None
if 'bt_strike_input' not in st.session_state: st.session_state.bt_strike_input = None
if 'bt_expiry_input' not in st.session_state: st.session_state.bt_expiry_input = None
if 'recent_options_df' not in st.session_state: st.session_state.recent_options_df = pd.DataFrame()
if 'selected_K' not in st.session_state: st.session_state.selected_K = None
if 'selected_T_opt' not in st.session_state: st.session_state.selected_T_opt = None
if 'selected_expiry_str' not in st.session_state: st.session_state.selected_expiry_str = None
if 'option_chain_data_dict' not in st.session_state: st.session_state.option_chain_data_dict = None


def main_app():
    st.title("QUANTIFI // Tariff Impact & Derivatives Analyzer")
    st.markdown("""*Advanced simulation & analysis incorporating tariffs, jumps, GARCH, market data, and news sentiment.*
**Disclaimer:** Educational tool. Not financial advice. Data limitations apply.""", unsafe_allow_html=True)

    # --- Render Sidebar & Get Config ---
    # FIX: Removed fred_api_key from arguments
    ticker_symbol, pricing_model_name, pricing_model_func, tau, lambda_sens, r, sigma_base, jump_params, req_jumps = render_sidebar(
        newsapi_client=newsapi_client,
        sentiment_analyzer=sentiment_analyzer,
        # fred_api_key=FRED_API_KEY, # Removed
        news_api_enabled=NEWS_API_ENABLED
    )
    model_args_common = {'lambda_sensitivity': lambda_sens, **jump_params}

    # --- Fetch Core Market Data ---
    st.sidebar.header("📈 Market Data")
    if st.sidebar.button("Refresh Market Data", key="refresh_market"):
        get_stock_price.clear(); get_option_expiries.clear()
        try: from utils import get_option_chain_data; get_option_chain_data.clear()
        except ImportError: pass
        st.sidebar.info("Refreshing data...")

    with st.spinner(f"Fetching latest price for {ticker_symbol}..."):
        fetched_S = get_stock_price(ticker_symbol)
    if fetched_S is not None: st.session_state.current_S = fetched_S
    elif st.session_state.current_S is None: st.session_state.current_S = DEFAULT_S_FALLBACK
    current_S = st.session_state.current_S
    if current_S is not None: st.sidebar.success(f"Current {ticker_symbol}: ${current_S:,.2f}")
    else: st.sidebar.error(f"{ticker_symbol} price unavailable.")

    with st.spinner(f"Fetching option expiries for {ticker_symbol}..."):
        option_expiries = get_option_expiries(ticker_symbol)
    if option_expiries: st.sidebar.caption(f"Found {len(option_expiries)} expiries.")
    else: st.sidebar.caption("No expiries found.")

    # --- Define Tabs ---
    tab_names = ["📊 Live Analysis", "💹 Vol Smile", "📈 Sensitivity", "💥 Stress Test", "⏳ Backtest", "🧊 3D Surface", "📰 News", "ℹ️ Explain"]
    tabs = st.tabs(tab_names)

    # --- Render Tabs ---
    render_tab_live_analysis(tabs[0], ticker_symbol, pricing_model_name, pricing_model_func, current_S, option_expiries, r, sigma_base, tau, model_args_common)
    render_tab_vol_smile(tabs[1], ticker_symbol, current_S, r, sigma_base)
    render_tab_sensitivity(tabs[2], pricing_model_name, pricing_model_func, current_S, r, sigma_base, tau, model_args_common, req_jumps)
    render_tab_stress_test(tabs[3], pricing_model_name, pricing_model_func, current_S, r, sigma_base, tau, model_args_common, req_jumps)
    # FIX: Removed fred_api_key from arguments
    render_tab_backtest(tabs[4], ticker_symbol, pricing_model_name, pricing_model_func, current_S, r, sigma_base, tau, lambda_sens, model_args_common, req_jumps) # Removed fred_api_key
    render_tab_3d_surface(tabs[5], pricing_model_name, pricing_model_func, current_S, r, sigma_base, tau, model_args_common, req_jumps)
    render_tab_news(tabs[6], NEWS_API_ENABLED)
    render_tab_explain(tabs[7], DOLTHUB_OWNER, DOLTHUB_REPO, RISK_FREE_RATE_SERIES)

# --- Main Execution ---
if __name__ == '__main__':
    try:
        main_app()
    except Exception as e:
        st.error(f"A critical application error occurred: {type(e).__name__} - {e}")
        st.exception(e)
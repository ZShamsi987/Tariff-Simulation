# sidebar_config.py
import streamlit as st
from datetime import date, timedelta
from typing import Tuple, Optional, Callable, Dict, Any
import pandas as pd

# Import necessary functions from utils
from utils import (
    get_fred_rate,
    fit_garch_and_forecast,
    fetch_news_and_sentiment,
    RISK_FREE_RATE_SERIES,
    MIN_HIST_DATA_POINTS,
    MIN_VOL_LEVEL,
    MAX_VOL_LEVEL,
    DEFAULT_TICKER
)

def render_sidebar(
    newsapi_client: Optional[Any],
    sentiment_analyzer: Any,
    # FIX: Removed fred_api_key parameter
    # fred_api_key: Optional[str],
    news_api_enabled: bool
    ) -> Tuple[str, str, Callable, float, float, float, float, Dict[str, float], bool]:
    """Renders the sidebar configuration and returns selected parameters."""

    st.sidebar.header("⚙️ Configuration")

    # --- Market & Model Configuration ---
    with st.sidebar.expander("Market & Model", expanded=True):
        ticker_symbol = st.text_input("Ticker", DEFAULT_TICKER, key="sb_ticker", help="Enter the underlying stock ticker symbol (e.g., AAPL, SPY).").upper().strip()
        pricing_model_name = st.selectbox(
            "Pricing Model", ["Mod. Black-Scholes", "Merton Jump Diffusion"], key="sb_model", index=0,
            help="Select the core option pricing model."
        )
        from utils import modified_black_scholes_with_tariff, merton_jump_diffusion
        if "Merton" in pricing_model_name:
            pricing_model_func = merton_jump_diffusion; req_jumps = True
        else:
            pricing_model_func = modified_black_scholes_with_tariff; req_jumps = False

    # --- Tariff Configuration ---
    with st.sidebar.expander("Tariff & Risk"):
        tau = st.slider("Tariff Rate τ", 0.0, 1.0, 0.10, 0.01, format="%.2f", key="sb_tau", help="Assumed tariff level (e.g., 0.10 = 10%).")
        lambda_sens = st.slider("Tariff Sensitivity λ", 0.0, 0.5, 0.10, 0.01, format="%.2f", key="sb_lambda", help="Factor scaling tariff impact on drift (r_drift = r + τ*λ).")

    # --- Interest Rate Configuration ---
    with st.sidebar.expander("Interest Rate r"):
        r_src = st.radio("Rate Source", ["Manual", f"FRED ({RISK_FREE_RATE_SERIES})"], index=1, key="sb_r_src", horizontal=True, help="Use manual rate or fetch from FRED.")
        r_man = st.number_input("Manual Rate r", -0.02, 0.2, value=st.session_state.current_r, step=0.001, format="%.4f", key="sb_r_man", help="Manually set annualized risk-free rate.", disabled=(r_src.startswith("FRED")))
        r = st.session_state.current_r
        if r_src.startswith("FRED"):
            with st.spinner(f"Fetching {RISK_FREE_RATE_SERIES} from FRED..."):
                # FIX: Removed api_key argument
                fetched_r = get_fred_rate(series_id=RISK_FREE_RATE_SERIES) # Removed api_key
                if isinstance(fetched_r, float):
                    r = fetched_r; st.session_state.current_r = r
                    st.success(f"Using FRED {RISK_FREE_RATE_SERIES}: {r:.4%}")
                else: st.warning(f"FRED fetch failed. Using previous rate: {r:.4%}")
        else:
            r = r_man; st.session_state.current_r = r
            st.info(f"Using Manual r: {r:.4%}")
        st.caption(f"Current Rate (r): {r:.4%}")

    # --- Volatility Configuration ---
    with st.sidebar.expander("Volatility σ"):
        vol_src = st.radio("Base Volatility Source", ["Manual", "GARCH(1,1) Forecast"], index=1, key="sb_vol_src", horizontal=True, help="Use manual vol or estimate using GARCH.")
        sig_man = st.number_input("Manual Volatility σ", 0.01, 2.0, value=st.session_state.sigma_base, step=0.01, format="%.4f", key="sb_sig_man", help="Manually set annualized base volatility.", disabled=(vol_src == "GARCH(1,1) Forecast"))
        sig_base_curr = st.session_state.sigma_base
        if vol_src == "GARCH(1,1) Forecast":
            g_days = st.number_input("GARCH Fit Days", min_value=MIN_HIST_DATA_POINTS, max_value=2500, value=252, step=50, key="sb_g_days", help=f"Number of past trading days for GARCH (min {MIN_HIST_DATA_POINTS}).")
            g_end = date.today(); g_start = g_end - timedelta(days=int(g_days * 1.7 + 30))
            if st.button("Run GARCH Forecast", key="sb_run_g"):
                with st.spinner(f"Fitting GARCH for {ticker_symbol}..."):
                    g_vol, g_stat = fit_garch_and_forecast(ticker_symbol, g_start, g_end, MIN_HIST_DATA_POINTS)
                if g_vol is not None:
                    sig_base_curr = g_vol; st.session_state.sigma_base = sig_base_curr
                    st.success(f"GARCH σ Forecast: {sig_base_curr:.4f}"); st.caption(f"Details: {g_stat}")
                else: st.warning(f"GARCH Failed: {g_stat}"); sig_base_curr = st.session_state.sigma_base
            else: st.info(f"Current Base σ: {st.session_state.sigma_base:.4f}. Click 'Run GARCH Forecast' to update.")
        else:
            st.session_state.sigma_base = sig_man; sig_base_curr = sig_man
            st.info(f"Using Manual σ: {sig_base_curr:.4f}")
        sigma_base = sig_base_curr
        if sigma_base < MIN_VOL_LEVEL: sigma_base = MIN_VOL_LEVEL; st.sidebar.warning(f"Vol clamped to min: {MIN_VOL_LEVEL:.4f}")
        elif sigma_base > MAX_VOL_LEVEL: st.sidebar.warning(f"Vol ({sigma_base:.4f}) is very high (> {MAX_VOL_LEVEL:.1f}).")
        st.session_state.sigma_base = sigma_base
        st.caption(f"Current Base σ: {sigma_base:.4f}")

    # --- Jump Parameter Configuration (Conditional) ---
    jump_params = {'jump_intensity': 0.0, 'jump_mean': 0.0, 'jump_vol': 0.0}
    if req_jumps:
        with st.sidebar.expander("Merton Jump Parameters"):
            jump_params['jump_intensity'] = st.slider("Jump Intensity λ_j (per year)", 0.0, 10.0, 0.7, 0.1, format="%.1f", key="sb_ji", help="Average number of jumps per year.")
            jump_params['jump_mean'] = st.slider("Mean Log Jump Size μ_j", -0.5, 0.5, -0.02, 0.005, format="%.3f", key="sb_jm", help="Average log-return of a jump.")
            jump_params['jump_vol'] = st.slider("Log Jump Size Volatility σ_j", 0.0, 1.0, 0.15, 0.01, format="%.3f", key="sb_jv", help="Std dev of log jump size.")

    # --- News & Sentiment Configuration ---
    st.sidebar.header("📰 News & Sentiment")
    with st.sidebar.expander("News Settings", expanded=False):
        default_kw = f"{ticker_symbol}, tariff, trade" if ticker_symbol else "tariff, trade"
        news_kw_in = st.text_input("Keywords (comma-separated)", default_kw, key="sb_news_kw", help="Keywords for news search.")
        news_kws = [k.strip() for k in news_kw_in.split(',') if k.strip()]
        news_num = st.slider("Max Articles", 5, 50, 15, key="sb_news_num", help="Max articles to fetch.")
        if news_api_enabled:
            if st.button("Fetch News & Sentiment", key="sb_fetch_news"):
                if not news_kws: st.warning("Enter valid keywords.")
                else:
                    with st.spinner(f"Fetching news for: {', '.join(news_kws)}..."):
                        news_df_res, news_stat_res = fetch_news_and_sentiment(
                            newsapi_client, sentiment_analyzer, news_kws, page_size=news_num
                        )
                    st.session_state.news_df = news_df_res
                    st.session_state.news_status = news_stat_res
                    st.session_state.news_keywords_processed = news_kw_in
                    st.info(f"News Status: {news_stat_res}")
        else: st.caption("News fetching disabled (check API key).")

    # --- Display Average Sentiment ---
    if news_api_enabled:
         news_df_state = st.session_state.get('news_df', pd.DataFrame())
         if not news_df_state.empty and 'Sent' in news_df_state.columns:
            avg_sent = news_df_state['Sent'].mean()
            num_articles = len(news_df_state)
            keywords_used = st.session_state.get('news_keywords_processed', 'N/A')
            st.sidebar.metric("Average News Sentiment", f"{avg_sent:.3f}", help=f"Avg VADER score based on {num_articles} articles for '{keywords_used}'.")
         else: st.sidebar.caption(st.session_state.get('news_status', 'News not fetched.'))

    # --- Return all configured parameters ---
    return ticker_symbol, pricing_model_name, pricing_model_func, tau, lambda_sens, r, sigma_base, jump_params, req_jumps
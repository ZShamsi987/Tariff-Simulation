# tabs/tab_live_analysis.py
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, Callable, List

# Import necessary functions from utils
from utils import (
    # Keep yfinance chain data for live bid/ask/IV
    get_option_chain_data,
    # Use Polygon strikes function if needed, but yfinance chain provides them too
    # get_polygon_options_strikes,
    calculate_implied_volatility,
    calculate_greeks,
    display_greeks_gauges,
    MIN_PRICE_LEVEL,
    MIN_TIME_LEVEL,
    DEFAULT_S_FALLBACK
)

def render_tab_live_analysis(
    st_tab: Any,
    ticker_symbol: str,
    pricing_model_name: str,
    pricing_model_func: Callable,
    current_S: Optional[float],
    # FIX: Accept polygon expirations list
    option_expiries: List[str], # This now comes from Polygon
    r: float,
    sigma_base: float,
    tau: float,
    model_args_common: Dict[str, Any]
    ):
    """Renders the content for the Live Analysis tab."""

    with st_tab:
        st.header(f"Live Option Analysis ({pricing_model_name})")
        col1a, col1b = st.columns([1, 2])

        with col1a:
            st.subheader("Option Selection")

            # --- Expiry Selection (Uses Polygon List) ---
            expiry_str: Optional[str] = None
            if not option_expiries:
                st.warning(f"No option expiries found for {ticker_symbol} via Polygon.")
            else:
                # Find default expiry (e.g., ~40 days out)
                target_date = datetime.now() + timedelta(days=40)
                try:
                    # Convert string dates to datetime objects for comparison
                    expiry_dates = [datetime.strptime(e, '%Y-%m-%d') for e in option_expiries]
                    best_match_expiry_dt = min(expiry_dates, key=lambda d: abs(d - target_date))
                    best_match_expiry_str = best_match_expiry_dt.strftime('%Y-%m-%d')
                    default_index = option_expiries.index(best_match_expiry_str)
                except (ValueError, IndexError):
                    default_index = 0 # Fallback
                expiry_str = st.selectbox(
                    "Select Expiry (Polygon)", option_expiries, index=default_index, key="t1_exp_select_poly",
                    help="Select option expiration date (from Polygon)."
                )

            # --- Fetch Live Option Chain (Still using yfinance for Bid/Ask/IV) ---
            # This part remains unchanged for now to keep live data functionality
            option_chain_data_dict: Optional[Dict[str, pd.DataFrame]] = None
            if expiry_str:
                with st.spinner(f"Fetching LIVE option chain via yfinance for {ticker_symbol} / {expiry_str}..."):
                    option_chain_data_dict = get_option_chain_data(ticker_symbol, expiry_str)
                    if not option_chain_data_dict:
                        st.warning(f"Could not fetch live yfinance chain data for {expiry_str}.")
                        option_chain_data_dict = None
            else:
                st.info("Select an expiry date to load option chain.")


            # --- Strike Selection (Uses strikes from yfinance chain) ---
            K: Optional[float] = None
            available_strikes: List[float] = []
            default_strike: Optional[float] = None
            if isinstance(current_S, (int, float)) and current_S > 0:
                step = 1.0 if current_S < 50 else (5.0 if current_S < 500 else 10.0)
                default_strike = round(current_S / step) * step

            if option_chain_data_dict: # Based on yfinance chain
                calls_df = option_chain_data_dict.get('calls', pd.DataFrame())
                puts_df = option_chain_data_dict.get('puts', pd.DataFrame())
                strikes_c = pd.to_numeric(calls_df['strike'], errors='coerce').dropna() if not calls_df.empty else pd.Series(dtype=float)
                strikes_p = pd.to_numeric(puts_df['strike'], errors='coerce').dropna() if not puts_df.empty else pd.Series(dtype=float)
                all_strikes = pd.concat([strikes_c, strikes_p]).unique()
                if len(all_strikes) > 0: available_strikes = sorted(all_strikes)

            if available_strikes:
                if default_strike is None: default_strike = available_strikes[len(available_strikes) // 2]
                closest_strike_val = min(available_strikes, key=lambda x: abs(x - default_strike))
                try: default_strike_index = available_strikes.index(closest_strike_val)
                except ValueError: default_strike_index = 0
                K = st.selectbox("Select Strike (from yfinance chain)", available_strikes, index=default_strike_index, format_func=lambda x: f"{x:,.2f}", key="t1_K_select_yf", help="Select option strike price.")
            else:
                st.info("No strikes in yfinance chain data. Enter manually.")
                input_default_K = float(default_strike) if default_strike is not None else DEFAULT_S_FALLBACK
                K = st.number_input("Enter Strike (K)", value=input_default_K, step=1.0, format="%.2f", key="t1_K_manual_yf", help="Manually enter strike price.")

            # --- Calculate TTM ---
            T_opt: Optional[float] = None; days_to_expiry: int = -1
            if expiry_str:
                try:
                    expiry_date_obj = datetime.strptime(expiry_str, '%Y-%m-%d').date()
                    today_date = date.today(); days_to_expiry = (expiry_date_obj - today_date).days
                    if days_to_expiry >= 0: T_opt = max(MIN_TIME_LEVEL, days_to_expiry / 365.25)
                    else: T_opt = 0.0
                    st.info(f"Time to Maturity (T): {T_opt:.6f} yrs ({days_to_expiry} days)")
                except Exception as e: T_opt = None; st.error(f"TTM Error: {e}")
            else: st.warning("Select expiry to calculate TTM.")

            # --- Store key outputs for other tabs ---
            st.session_state['selected_K'] = K
            st.session_state['selected_T_opt'] = T_opt
            st.session_state['selected_expiry_str'] = expiry_str
            # Store yfinance chain data for Vol Smile tab
            st.session_state['option_chain_data_dict'] = option_chain_data_dict

            # --- Model Calculation Section ---
            # ... (Keep model calculation logic as is) ...
            st.subheader("Model Calculations")
            inputs_valid = all(isinstance(v, (int, float)) and not np.isnan(v) for v in [current_S, K, T_opt, r, sigma_base, tau]) and \
                           current_S > MIN_PRICE_LEVEL and K > MIN_PRICE_LEVEL and T_opt is not None and \
                           isinstance(model_args_common.get('lambda_sensitivity'), (int, float))
            if model_args_common.get('jump_intensity') is not None:
                 jump_inputs_valid = all(isinstance(v, (int, float)) and not np.isnan(v) for k, v in model_args_common.items() if k.startswith('jump_'))
                 inputs_valid = inputs_valid and jump_inputs_valid

            price_call_model, price_put_model = np.nan, np.nan
            greeks_call_dict, greeks_put_dict = {}, {}

            if inputs_valid:
                model_calc_args = {'S': current_S, 'K': K, 'T': T_opt, 'r': r, 'sigma': sigma_base, 'tau': tau, **model_args_common}
                try:
                    with st.spinner("Calculating model prices and Greeks..."):
                        price_call_model = pricing_model_func(**model_calc_args, option_type='call')
                        price_put_model = pricing_model_func(**model_calc_args, option_type='put')
                        greeks_call_dict = calculate_greeks(S=current_S, K=K, T=T_opt, r=r, sigma=sigma_base, tau=tau, option_type='call', model_func=pricing_model_func, model_args=model_args_common)
                        greeks_put_dict = calculate_greeks(S=current_S, K=K, T=T_opt, r=r, sigma=sigma_base, tau=tau, option_type='put', model_func=pricing_model_func, model_args=model_args_common)
                    st.metric(f"Model Call Price (τ={tau:.2f})", f"${price_call_model:.3f}" if pd.notna(price_call_model) else "N/A")
                    st.metric(f"Model Put Price (τ={tau:.2f})", f"${price_put_model:.3f}" if pd.notna(price_put_model) else "N/A")
                except Exception as e:
                    st.error(f"Model calculation error: {type(e).__name__} - {e}")
                    st.metric("Model Call Price", "Error"); st.metric("Model Put Price", "Error")
            else:
                st.warning("Inputs invalid. Cannot calculate model prices/Greeks.")
                st.metric("Model Call Price", "N/A"); st.metric("Model Put Price", "N/A")


        # --- Column 2: Market Data & Greeks ---
        with col1b:
            # This part still relies on the yfinance chain data for live bid/ask/mid/IV
            st.subheader("Market Data (yfinance) & Implied Volatility")
            iv_call_market, iv_put_market = np.nan, np.nan
            market_price_call, market_price_put = np.nan, np.nan
            call_row, put_row = None, None

            if option_chain_data_dict and K is not None:
                calls_df = option_chain_data_dict.get('calls', pd.DataFrame())
                puts_df = option_chain_data_dict.get('puts', pd.DataFrame())
                if not calls_df.empty: call_rows = calls_df[np.isclose(calls_df['strike'], K)]; call_row = call_rows.iloc[0] if not call_rows.empty else None
                if not puts_df.empty: put_rows = puts_df[np.isclose(puts_df['strike'], K)]; put_row = put_rows.iloc[0] if not put_rows.empty else None

            st.write("**Market Call (yfinance):**")
            if call_row is not None:
                mkt_c_last=call_row.get('lastPrice',np.nan); mkt_c_bid=call_row.get('bid',np.nan); mkt_c_ask=call_row.get('ask',np.nan)
                mkt_c_vol=call_row.get('volume',np.nan); mkt_c_oi=call_row.get('openInterest',np.nan); mkt_c_iv_yf=call_row.get('impliedVolatility',np.nan)
                market_price_call = call_row.get('mid', np.nan);
                if pd.isna(market_price_call): market_price_call = mkt_c_last
                st.markdown(f"  Last: `{mkt_c_last:,.2f}` | Bid: `{mkt_c_bid:,.2f}` | Ask: `{mkt_c_ask:,.2f}`")
                st.markdown(f"  Mid: `{market_price_call:,.2f}`" if pd.notna(market_price_call) else "  Mid: `N/A`")
                st.markdown(f"  IV (Src): `{mkt_c_iv_yf:.2%}`" if pd.notna(mkt_c_iv_yf) else "  IV (Src): `N/A`")
                st.caption(f"  Vol: {mkt_c_vol:,.0f} | OI: {mkt_c_oi:,.0f}")
            else: st.markdown("  No market data available.")

            st.write("**Market Put (yfinance):**")
            if put_row is not None:
                mkt_p_last=put_row.get('lastPrice',np.nan); mkt_p_bid=put_row.get('bid',np.nan); mkt_p_ask=put_row.get('ask',np.nan)
                mkt_p_vol=put_row.get('volume',np.nan); mkt_p_oi=put_row.get('openInterest',np.nan); mkt_p_iv_yf=put_row.get('impliedVolatility',np.nan)
                market_price_put = put_row.get('mid', np.nan)
                if pd.isna(market_price_put): market_price_put = mkt_p_last
                st.markdown(f"  Last: `{mkt_p_last:,.2f}` | Bid: `{mkt_p_bid:,.2f}` | Ask: `{mkt_p_ask:,.2f}`")
                st.markdown(f"  Mid: `{market_price_put:,.2f}`" if pd.notna(market_price_put) else "  Mid: `N/A`")
                st.markdown(f"  IV (Src): `{mkt_p_iv_yf:.2%}`" if pd.notna(mkt_p_iv_yf) else "  IV (Src): `N/A`")
                st.caption(f"  Vol: {mkt_p_vol:,.0f} | OI: {mkt_p_oi:,.0f}")
            else: st.markdown("  No market data available.")

            # --- Implied Volatility Calculation (based on yfinance market price) ---
            iv_inputs_ok = all(isinstance(v, (int, float)) and not np.isnan(v) for v in [current_S, K, T_opt, r]) and \
                           current_S > MIN_PRICE_LEVEL and K > MIN_PRICE_LEVEL and T_opt is not None and T_opt > MIN_TIME_LEVEL
            if iv_inputs_ok:
                iv_calc_args = {'S': current_S, 'K': K, 'T': T_opt, 'r': r}
                if pd.notna(market_price_call) and market_price_call > MIN_PRICE_LEVEL:
                    with st.spinner("Calculating IV Call..."): iv_call_market = calculate_implied_volatility(market_price_call, **iv_calc_args, option_type='call')
                if pd.notna(market_price_put) and market_price_put > MIN_PRICE_LEVEL:
                     with st.spinner("Calculating IV Put..."): iv_put_market = calculate_implied_volatility(market_price_put, **iv_calc_args, option_type='put')
                c1iv, c2iv = st.columns(2)
                c1iv.metric("Implied Vol (Call, Calc. BSM)", f"{iv_call_market:.2%}" if pd.notna(iv_call_market) else "N/A", help="IV calculated from market price using standard BSM.")
                c2iv.metric("Implied Vol (Put, Calc. BSM)", f"{iv_put_market:.2%}" if pd.notna(iv_put_market) else "N/A", help="IV calculated from market price using standard BSM.")
            elif T_opt is not None and T_opt <= MIN_TIME_LEVEL: st.warning("Cannot calculate IV for expired options (T ≈ 0).")
            else: st.warning("Cannot calculate IV due to invalid base inputs (S, K, T, r).")

            # --- Display Greeks ---
            st.markdown("---"); st.subheader("Model Greeks (Call)")
            if inputs_valid: display_greeks_gauges(greeks_call_dict, 'call')
            else: st.write("Greeks: N/A (Inputs invalid)")
            st.subheader("Model Greeks (Put)")
            if inputs_valid: display_greeks_gauges(greeks_put_dict, 'put')
            else: st.write("Greeks: N/A (Inputs invalid)")
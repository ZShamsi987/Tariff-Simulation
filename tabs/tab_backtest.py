# tabs/tab_backtest.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta, datetime
import time
from typing import Optional, Dict, Any, Callable

# Import necessary functions from utils
from utils import (
    get_polygon_option_eod,
    generate_polygon_ticker,
    get_historical_data,
    get_historical_fred_rates,
    MIN_PRICE_LEVEL,
    MIN_TIME_LEVEL,
    MIN_VOL_LEVEL,
    RISK_FREE_RATE_SERIES,
    POLYGON_API_KEY, # Check if key exists
    DEFAULT_S_FALLBACK
)

# REMOVED DoltHub callback
# def update_backtest_selections(): ...

def render_tab_backtest(
    st_tab: Any,
    ticker_symbol: str,
    pricing_model_name: str,
    pricing_model_func: Callable,
    current_S: Optional[float],
    r: float,
    sigma_base: float,
    tau: float,
    lambda_sens: float,
    model_args_common: Dict[str, Any],
    req_jumps: bool
    ):
    """Renders the content for the Backtest tab using Polygon.io EOD data."""
    with st_tab:
        st.header(f"Historical Backtest vs Market ({pricing_model_name})")
        st.markdown(f"Compare historical model prices against Polygon.io **End-of-Day (EOD)** market data (`close` price).")
        # FIX: Adjusted caption
        st.caption("Methodology notes in 'Explain' tab. Fetches EOD data day-by-day. Longer ranges increase runtime (~15-25s per day queried due to API limits).")

        polygon_ready = bool(POLYGON_API_KEY)
        if not polygon_ready:
             st.error("Polygon API Key not configured in utils.py. Backtest unavailable.")
             return

        col5a, col5b = st.columns([1, 2])
        with col5a:
            st.subheader("Backtest Configuration")
            # Removed DoltHub helper section

            st.markdown("**Backtest Parameters**")
            default_end_date_bt = date.today() - timedelta(days=1)
            default_start_date_bt = default_end_date_bt - timedelta(days=60)
            hist_start_date = st.date_input("Backtest Start Date", default_start_date_bt, max_value=default_end_date_bt, key="bt_start_poly")
            hist_end_date = st.date_input("Backtest End Date", default_end_date_bt, min_value=hist_start_date, max_value=default_end_date_bt, key="bt_end_poly")

            # Default K and Expiry logic remains the same
            hist_K_default = st.session_state.get('selected_K')
            if hist_K_default is None: hist_K_default = current_S if isinstance(current_S,(int,float)) and current_S>0 else DEFAULT_S_FALLBACK
            else: hist_K_default = float(hist_K_default) # Ensure float if from state

            hist_expiry_default_str = st.session_state.get('selected_expiry_str')
            hist_expiry_date_default_val = hist_end_date + timedelta(days=60) # Default fallback
            if hist_expiry_default_str:
                try:
                    temp_date = datetime.strptime(hist_expiry_default_str, '%Y-%m-%d').date()
                    if temp_date > hist_end_date: hist_expiry_date_default_val = temp_date
                except ValueError: pass # Ignore if format is wrong

            hist_K = st.number_input("Option Strike (K)", value=hist_K_default, step=1.0, format="%.2f", key="bt_strike_poly_input") # Use unique key
            hist_expiry_date = st.date_input("Option Expiry Date", value=hist_expiry_date_default_val, min_value=hist_end_date + timedelta(days=1), key="bt_expiry_poly_input") # Use unique key
            hist_expiry_str = hist_expiry_date.strftime('%Y-%m-%d')

            st.markdown("---"); st.markdown("**Model Parameters for Backtest Period:**")
            hist_tau = st.slider("Hist. Tariff τ (Constant)", 0.0, 1.0, tau, 0.01, format="%.2f", key="bt_tau_poly_hist")
            hist_lambda_sens = st.slider("Hist. Tariff Sens. λ (Constant)", 0.0, 0.5, lambda_sens, 0.01, format="%.2f", key="bt_lambda_poly_hist")
            hist_sigma_fallback = sigma_base
            st.info(f"Using Volatility σ = **{hist_sigma_fallback:.4f}** (from Sidebar) for all calculations.")
            hist_sigma_ok = hist_sigma_fallback is not None and isinstance(hist_sigma_fallback, float) and hist_sigma_fallback >= 0
            if not hist_sigma_ok: st.error("Sidebar Volatility σ invalid.")
            hist_jump_params = {k: v for k, v in model_args_common.items() if k.startswith('jump_')} if req_jumps else {}
            if req_jumps: st.info(f"Using constant sidebar jump params: {hist_jump_params}")

            run_backtest = st.button("Run Backtest Analysis", key="bt_run_poly_button", disabled=(not hist_expiry_date or hist_start_date >= hist_end_date or not hist_sigma_ok or not polygon_ready))

        if run_backtest:
            with col5b:
                st.subheader("Backtest Execution & Results")
                # --- Fetch prerequisite data ---
                with st.spinner("Fetching Stock Data & Historical Rates..."):
                    hist_stock_data = get_historical_data(ticker_symbol, hist_start_date, hist_end_date)
                    hist_rates_data = st.session_state.get('historical_rates')
                    rates_need_fetch = True
                    if hist_rates_data is not None:
                         if not isinstance(hist_rates_data.index, pd.DatetimeIndex): hist_rates_data.index = pd.to_datetime(hist_rates_data.index)
                         if hist_rates_data.index.min().date() <= hist_start_date and hist_rates_data.index.max().date() >= hist_end_date:
                              hist_rates_data = hist_rates_data.loc[pd.Timestamp(hist_start_date):pd.Timestamp(hist_end_date)]
                              rates_need_fetch = False
                    if rates_need_fetch:
                         hist_rates_data = get_historical_fred_rates(series_id=RISK_FREE_RATE_SERIES, start_date=hist_start_date, end_date=hist_end_date)
                         st.session_state.historical_rates = hist_rates_data

                # --- Validate prerequisite data ---
                prereq_ok = True
                if hist_stock_data is None or hist_stock_data.empty: st.error(f"No hist stock data for {ticker_symbol}. Cannot run backtest."); prereq_ok = False
                if hist_rates_data is None or hist_rates_data.empty:
                     st.warning(f"No hist FRED rates. Using current rate {r:.3%}. Reduces accuracy.")
                     if hist_stock_data is not None and not hist_stock_data.empty: hist_rates_data = pd.Series(r, index=hist_stock_data.index)
                     else: st.error("Cannot create fallback rate series."); prereq_ok = False

                # --- Proceed if stock and rates OK ---
                if prereq_ok:
                    backtest_results = []
                    trading_days = hist_stock_data.index
                    num_days = len(trading_days)
                    st.write(f"Processing {num_days} trading days from {hist_start_date} to {hist_end_date}...")
                    bt_progress = st.progress(0)
                    days_with_option_data = 0
                    api_calls_counter = 0 # Track calls for rate limiting
                    hist_model_args_common = {'lambda_sensitivity': hist_lambda_sens, **hist_jump_params}

                    # --- Loop through each TRADING DAY ---
                    for idx, quote_date_dt in enumerate(trading_days):
                        # Rate limit check before making calls for the day
                        # Need 2 calls per day (call + put)
                        if api_calls_counter > 0 and api_calls_counter % 4 == 0 : # Sleep every 2 days (4 calls) approx
                             print(f"Pausing for rate limit... ({api_calls_counter} calls)") # Debug print
                             time.sleep(12.5) # Wait slightly > 12 seconds

                        current_quote_date = quote_date_dt.date()
                        stock_row = hist_stock_data.loc[quote_date_dt]
                        hist_S = stock_row['Close']
                        if pd.isna(hist_S) or hist_S <= MIN_PRICE_LEVEL: continue

                        hist_TTM_days = (hist_expiry_date - current_quote_date).days
                        if hist_TTM_days < 0: continue
                        hist_TTM = max(MIN_TIME_LEVEL, hist_TTM_days / 365.25)

                        hist_r_day = hist_rates_data.get(quote_date_dt, r);
                        if pd.isna(hist_r_day): hist_r_day = r

                        # Fetch Polygon EOD Data day by day
                        poly_call_ticker = generate_polygon_ticker(ticker_symbol, hist_expiry_date, hist_K, 'call')
                        call_eod_data = get_polygon_option_eod(poly_call_ticker, current_quote_date)
                        api_calls_counter += 1

                        # Small delay between the two calls for the same day
                        time.sleep(0.3)

                        poly_put_ticker = generate_polygon_ticker(ticker_symbol, hist_expiry_date, hist_K, 'put')
                        put_eod_data = get_polygon_option_eod(poly_put_ticker, current_quote_date)
                        api_calls_counter += 1

                        market_call_close = call_eod_data.get('close', np.nan) if call_eod_data else np.nan
                        market_put_close = put_eod_data.get('close', np.nan) if put_eod_data else np.nan
                        iv_call, iv_put = np.nan, np.nan # No IV from Polygon EOD
                        sigma_used = hist_sigma_fallback

                        if call_eod_data or put_eod_data: days_with_option_data += 1

                        base_args = {'S':hist_S, 'K':hist_K, 'T':hist_TTM, 'r':hist_r_day, 'sigma':sigma_used, **hist_model_args_common}
                        args_tau = {**base_args, 'tau': hist_tau}; args_no_tau = {**base_args, 'tau': 0.0}
                        mc_t, mp_t, mc_nt, mp_nt = np.nan, np.nan, np.nan, np.nan
                        try:
                            mc_t=pricing_model_func(**args_tau, option_type='call'); mp_t=pricing_model_func(**args_tau, option_type='put')
                            mc_nt=pricing_model_func(**args_no_tau, option_type='call'); mp_nt=pricing_model_func(**args_no_tau, option_type='put')
                        except Exception: pass

                        backtest_results.append({"Date": quote_date_dt, "StockPrice": hist_S, "TTM": hist_TTM, "Rate": hist_r_day, "SigmaUsed": sigma_used, "HistIVCall": iv_call, "HistIVPut": iv_put, "MarketCall": market_call_close, "MarketPut": market_put_close, "ModelCall_Tariff": mc_t, "ModelPut_Tariff": mp_t, "ModelCall_NoTariff": mc_nt, "ModelPut_NoTariff": mp_nt})
                        bt_progress.progress((idx + 1) / num_days)

                    # --- End Loop ---
                    bt_progress.empty()
                    st.info(f"Processed {num_days} trading days. Found Polygon EOD options data for {days_with_option_data} days.")

                    # --- Process & Display Results ---
                    if not backtest_results: st.warning("No valid points generated during backtest loop.")
                    else:
                        # ... (Result processing, plotting, error calculation - UNCHANGED, using 'MarketCall'/'MarketPut') ...
                        results_df = pd.DataFrame(backtest_results).set_index("Date"); st.session_state.backtest_results_df = results_df
                        results_df_calls = results_df.dropna(subset=['MarketCall'])
                        if results_df_calls.empty: st.warning("No Polygon EOD call price data found in the period for comparison.")
                        else:
                            st.subheader("Backtest Results - Call Option"); fig_bt_call = go.Figure(); fig_bt_call.add_trace(go.Scatter(x=results_df_calls.index, y=results_df_calls['MarketCall'], name='Market Call Price (EOD Close)', line=dict(color='black', width=1.5))); fig_bt_call.add_trace(go.Scatter(x=results_df_calls.index, y=results_df_calls['ModelCall_Tariff'], name=f'Model Call (τ={hist_tau:.2f})', line=dict(color='blue', width=1.5))); fig_bt_call.add_trace(go.Scatter(x=results_df_calls.index, y=results_df_calls['ModelCall_NoTariff'], name='Model Call (τ=0)', line=dict(color='lightblue', dash='dash', width=1.5))); fig_bt_call.update_layout(title=f"Call Backtest: {ticker_symbol} K={hist_K}, Exp={hist_expiry_str}", yaxis_title="Price ($)", yaxis_tickformat="$,.2f", legend=dict(orientation="h", yanchor="bottom", y=1.02), hovermode="x unified"); st.plotly_chart(fig_bt_call, use_container_width=True)
                            results_df_calls['Error_Tariff'] = results_df_calls['ModelCall_Tariff'] - results_df_calls['MarketCall']; results_df_calls['Error_NoTariff'] = results_df_calls['ModelCall_NoTariff'] - results_df_calls['MarketCall']
                            errors_t=results_df_calls['Error_Tariff'].dropna(); errors_nt=results_df_calls['Error_NoTariff'].dropna()
                            if not errors_t.empty: mae_t=errors_t.abs().mean(); rmse_t=np.sqrt((errors_t**2).mean()); mae_nt=errors_nt.abs().mean(); rmse_nt=np.sqrt((errors_nt**2).mean()); st.subheader("Pricing Error Analysis (Calls: Model - Market)"); err_col1, err_col2 = st.columns(2); err_col1.metric(f"MAE (τ={hist_tau:.2f})", f"${mae_t:.4f}", f"RMSE: ${rmse_t:.4f}"); err_col2.metric("MAE (τ=0)", f"${mae_nt:.4f}", f"RMSE: ${rmse_nt:.4f}"); fig_err_call = go.Figure(); fig_err_call.add_trace(go.Scatter(x=errors_t.index, y=errors_t, name=f'Error (τ={hist_tau:.2f})', line=dict(color='red'))); fig_err_call.add_trace(go.Scatter(x=errors_nt.index, y=errors_nt, name='Error (τ=0)', line=dict(color='pink', dash='dash'))); fig_err_call.add_hline(y=0, line=dict(color='black', width=1)); fig_err_call.update_layout(title="Call Pricing Error Over Time", yaxis_title="Error ($)", yaxis_tickformat="$,.3f", legend=dict(orientation="h", yanchor="bottom", y=1.02), hovermode="x unified"); st.plotly_chart(fig_err_call, use_container_width=True)
                            else: st.warning("Could not calculate call errors.")
                            with st.expander("View Call Backtest Data"): cols=['StockPrice','TTM','Rate','SigmaUsed','MarketCall','ModelCall_Tariff','ModelCall_NoTariff','Error_Tariff','Error_NoTariff']; fmt={'StockPrice':'${:,.2f}','TTM':'{:.4f}','Rate':'{:.4%}','SigmaUsed':'{:.4%}','MarketCall':'${:,.3f}','ModelCall_Tariff':'${:,.3f}','ModelCall_NoTariff':'${:,.3f}','Error_Tariff':'{:+,.3f}','Error_NoTariff':'{:+,.3f}'}; st.dataframe(results_df_calls[cols].style.format(fmt, na_rep="N/A"))
                        st.markdown("---")
                        results_df_puts = results_df.dropna(subset=['MarketPut'])
                        if results_df_puts.empty: st.warning("No Polygon EOD put price data found in the period for comparison.")
                        else:
                            st.subheader("Backtest Results - Put Option"); fig_bt_put = go.Figure(); fig_bt_put.add_trace(go.Scatter(x=results_df_puts.index, y=results_df_puts['MarketPut'], name='Market Put Price (EOD Close)', line=dict(color='black', width=1.5))); fig_bt_put.add_trace(go.Scatter(x=results_df_puts.index, y=results_df_puts['ModelPut_Tariff'], name=f'Model Put (τ={hist_tau:.2f})', line=dict(color='green', width=1.5))); fig_bt_put.add_trace(go.Scatter(x=results_df_puts.index, y=results_df_puts['ModelPut_NoTariff'], name='Model Put (τ=0)', line=dict(color='lightgreen', dash='dash', width=1.5))); fig_bt_put.update_layout(title=f"Put Backtest: {ticker_symbol} K={hist_K}, Exp={hist_expiry_str}", yaxis_title="Price ($)", yaxis_tickformat="$,.2f", legend=dict(orientation="h", yanchor="bottom", y=1.02), hovermode="x unified"); st.plotly_chart(fig_bt_put, use_container_width=True)
                            results_df_puts['Error_Tariff'] = results_df_puts['ModelPut_Tariff'] - results_df_puts['MarketPut']; results_df_puts['Error_NoTariff'] = results_df_puts['ModelPut_NoTariff'] - results_df_puts['MarketPut']
                            errors_t=results_df_puts['Error_Tariff'].dropna(); errors_nt=results_df_puts['Error_NoTariff'].dropna()
                            if not errors_t.empty: mae_t=errors_t.abs().mean(); rmse_t=np.sqrt((errors_t**2).mean()); mae_nt=errors_nt.abs().mean(); rmse_nt=np.sqrt((errors_nt**2).mean()); st.subheader("Pricing Error Analysis (Puts: Model - Market)"); err_col1p, err_col2p = st.columns(2); err_col1p.metric(f"MAE (τ={hist_tau:.2f})", f"${mae_t:.4f}", f"RMSE: ${rmse_t:.4f}"); err_col2p.metric("MAE (τ=0)", f"${mae_nt:.4f}", f"RMSE: ${rmse_nt:.4f}")
                            else: st.warning("Could not calculate put errors.")
                            with st.expander("View Put Backtest Data"): cols=['StockPrice','TTM','Rate','SigmaUsed','MarketPut','ModelPut_Tariff','ModelPut_NoTariff','Error_Tariff','Error_NoTariff']; fmt={'StockPrice':'${:,.2f}','TTM':'{:.4f}','Rate':'{:.4%}','SigmaUsed':'{:.4%}','MarketPut':'${:,.3f}','ModelPut_Tariff':'${:,.3f}','ModelPut_NoTariff':'${:,.3f}','Error_Tariff':'{:+,.3f}','Error_NoTariff':'{:+,.3f}'}; st.dataframe(results_df_puts[cols].style.format(fmt, na_rep="N/A"))

        elif run_backtest:
             with col5b: st.error("Backtest prerequisites failed. Check stock/rate data availability.")
        else:
             with col5b: st.info("Configure backtest parameters and click 'Run Backtest Analysis'.")
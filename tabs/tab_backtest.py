# tabs/tab_backtest.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta, datetime
from typing import Optional, Dict, Any, Callable

# Import necessary functions from utils
from utils import (
    fetch_historical_options_via_api,
    fetch_recent_distinct_options_api,
    get_historical_data,
    get_historical_fred_rates,
    MIN_PRICE_LEVEL,
    MIN_TIME_LEVEL,
    MIN_VOL_LEVEL,
    RISK_FREE_RATE_SERIES,
    DOLTHUB_API_KEY,
    DEFAULT_S_FALLBACK
)

# Helper function to update session state for autofill
def update_backtest_selections():
    selected_option = st.session_state.get("recent_option_selector")
    if selected_option and selected_option != "Select Recent Option...":
        try:
            parts = selected_option.split(" ")
            expiry_str = parts[0]; strike_str = parts[1].split("=")[1]
            st.session_state['bt_strike_input'] = float(strike_str)
            st.session_state['bt_expiry_input'] = datetime.strptime(expiry_str, '%Y-%m-%d').date()
        except Exception as e:
            st.warning(f"Could not parse selected option: {selected_option}. Error: {e}")

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
    # FIX: Removed fred_api_key parameter
    # fred_api_key: Optional[str],
    req_jumps: bool
    ):
    """Renders the content for the Backtest tab."""
    with st_tab:
        st.header(f"Historical Backtest vs Market ({pricing_model_name})")
        st.markdown(f"Compare historical model prices against market data from DoltHub API.")
        st.caption("Methodology notes in 'Explain' tab.")

        dolthub_ready = bool(DOLTHUB_API_KEY)
        if not dolthub_ready:
             st.error("DoltHub API Key not configured. Backtest unavailable.")
             return

        col5a, col5b = st.columns([1, 2])
        with col5a:
            st.subheader("Backtest Configuration")

            st.markdown("**Select Recent Option (Optional Helper)**")
            st.caption("Fetches recent distinct options to help select valid K/Expiry.")
            if st.button("Load Recent Options", key="load_recent_opts"):
                with st.spinner(f"Fetching recent options for {ticker_symbol} from DoltHub..."):
                    st.session_state.recent_options_df = fetch_recent_distinct_options_api(ticker_symbol, limit=50)
            recent_options_df = st.session_state.get("recent_options_df", pd.DataFrame())
            options_list = ["Select Recent Option..."]
            if not recent_options_df.empty:
                options_list.extend(recent_options_df.apply(lambda row: f"{row['expiration'].strftime('%Y-%m-%d')} K={row['strike']:.2f} {row['type'].upper()}", axis=1).tolist())
            st.selectbox("Choose a recent option:", options=options_list, key="recent_option_selector", on_change=update_backtest_selections, help="Select an option here to autofill Strike and Expiry below.")
            st.markdown("---")

            st.markdown("**Main Backtest Parameters**")
            default_end_date_bt = date.today() - timedelta(days=1); default_start_date_bt = default_end_date_bt - timedelta(days=90)
            hist_start_date = st.date_input("Backtest Start Date", default_start_date_bt, max_value=default_end_date_bt, key="bt_start")
            hist_end_date = st.date_input("Backtest End Date", default_end_date_bt, min_value=hist_start_date, max_value=default_end_date_bt, key="bt_end")
            if 'bt_strike_input' not in st.session_state or st.session_state.bt_strike_input is None:
                 hist_K_default = current_S if isinstance(current_S,(int,float)) and current_S>0 else DEFAULT_S_FALLBACK
                 st.session_state.bt_strike_input = float(hist_K_default)
            if 'bt_expiry_input' not in st.session_state or st.session_state.bt_expiry_input is None:
                 st.session_state.bt_expiry_input = hist_end_date + timedelta(days=60)
            hist_K = st.number_input("Option Strike (K)", value=st.session_state.bt_strike_input, step=1.0, format="%.2f", key="bt_strike")
            hist_expiry_date = st.date_input("Option Expiry Date", value=st.session_state.bt_expiry_input, min_value=hist_end_date + timedelta(days=1), key="bt_expiry")
            hist_expiry_str = hist_expiry_date.strftime('%Y-%m-%d')

            st.markdown("---"); st.markdown("**Model Parameters for Backtest Period:**")
            hist_tau = st.slider("Hist. Tariff τ (Constant)", 0.0, 1.0, tau, 0.01, format="%.2f", key="bt_tau_hist")
            hist_lambda_sens = st.slider("Hist. Tariff Sens. λ (Constant)", 0.0, 0.5, lambda_sens, 0.01, format="%.2f", key="bt_lambda_hist")
            hist_sigma_fallback = sigma_base
            st.info(f"Volatility Fallback σ: **{hist_sigma_fallback:.4f}**")
            hist_sigma_ok = hist_sigma_fallback is not None and isinstance(hist_sigma_fallback, float) and hist_sigma_fallback >= 0
            if not hist_sigma_ok: st.error("Fallback σ invalid.")
            hist_jump_params = {k: v for k, v in model_args_common.items() if k.startswith('jump_')} if req_jumps else {}
            if req_jumps: st.info(f"Using constant sidebar jump params: {hist_jump_params}")

            run_backtest = st.button("Run Backtest Analysis", key="bt_run_button", disabled=(not hist_expiry_date or hist_start_date >= hist_end_date or not hist_sigma_ok or not dolthub_ready))

        if run_backtest:
            with col5b:
                st.subheader("Backtest Execution & Results")
                with st.spinner("Fetching historical data (Stock, Options API, Rates)..."):
                    hist_options_calls = fetch_historical_options_via_api(ticker=ticker_symbol, start_date=hist_start_date, end_date=hist_end_date, strike=hist_K, expiry=hist_expiry_str, option_type='call')
                    hist_options_puts = fetch_historical_options_via_api(ticker=ticker_symbol, start_date=hist_start_date, end_date=hist_end_date, strike=hist_K, expiry=hist_expiry_str, option_type='put')
                    hist_stock_data = get_historical_data(ticker_symbol, hist_start_date, hist_end_date)
                    hist_rates_data = st.session_state.get('historical_rates')
                    rates_need_fetch = True
                    if hist_rates_data is not None:
                         if hist_rates_data.index.min().date() <= hist_start_date and hist_rates_data.index.max().date() >= hist_end_date:
                              hist_rates_data = hist_rates_data.loc[hist_start_date:hist_end_date]
                              st.info("Using cached historical rates.")
                              rates_need_fetch = False
                    if rates_need_fetch:
                         st.info("Fetching historical rates from FRED...")
                         # FIX: Removed api_key argument
                         hist_rates_data = get_historical_fred_rates(series_id=RISK_FREE_RATE_SERIES, start_date=hist_start_date, end_date=hist_end_date) # Removed api_key
                         st.session_state.historical_rates = hist_rates_data

                data_ok = True
                if (hist_options_calls.empty and hist_options_puts.empty): st.error(f"No hist options data from DoltHub API for K={hist_K}, Exp={hist_expiry_str}."); data_ok = False
                if hist_stock_data is None or hist_stock_data.empty: st.error(f"No hist stock data for {ticker_symbol}."); data_ok = False
                if hist_rates_data is None or hist_rates_data.empty:
                     st.warning(f"No hist FRED rates. Using current rate {r:.3%}. Reduces accuracy.")
                     if hist_stock_data is not None and not hist_stock_data.empty: hist_rates_data = pd.Series(r, index=hist_stock_data.index)
                     else: st.error("Cannot create fallback rate series."); data_ok = False

                if data_ok:
                    # ... (Rest of the backtest calculation and display logic remains the same as previous version) ...
                    st.success(f"Data fetched: {len(hist_stock_data)} stock days, {len(hist_options_calls)} calls, {len(hist_options_puts)} puts.")
                    backtest_results = []; bt_progress = st.progress(0); num_days = len(hist_stock_data)
                    hist_model_args_common = {'lambda_sensitivity': hist_lambda_sens, **hist_jump_params}
                    for idx, (quote_date_dt, stock_row) in enumerate(hist_stock_data.iterrows()):
                        current_quote_date = quote_date_dt.date(); hist_S = stock_row['Close'];
                        if pd.isna(hist_S) or hist_S <= MIN_PRICE_LEVEL: continue
                        hist_TTM_days = (hist_expiry_date - current_quote_date).days;
                        if hist_TTM_days < 0: continue
                        hist_TTM = max(MIN_TIME_LEVEL, hist_TTM_days / 365.25)
                        hist_r_day = hist_rates_data.get(quote_date_dt, r);
                        if pd.isna(hist_r_day): hist_r_day = r
                        market_call, market_put = np.nan, np.nan; iv_call, iv_put = np.nan, np.nan
                        call_rec = hist_options_calls[hist_options_calls.index.date == current_quote_date]
                        if not call_rec.empty: cr=call_rec.iloc[0]; market_call=cr.get('mid', cr.get('close',np.nan)); iv_call=cr.get('implied_volatility')
                        put_rec = hist_options_puts[hist_options_puts.index.date == current_quote_date]
                        if not put_rec.empty: pr=put_rec.iloc[0]; market_put=pr.get('mid', pr.get('close',np.nan)); iv_put=pr.get('implied_volatility')
                        if pd.notna(market_call) and market_call < 0: market_call = np.nan
                        if pd.notna(market_put) and market_put < 0: market_put = np.nan
                        sigma_used = hist_sigma_fallback; valid_ivs = [iv for iv in [iv_call, iv_put] if pd.notna(iv)];
                        if valid_ivs: sigma_used = np.mean(valid_ivs)
                        base_args = {'S':hist_S, 'K':hist_K, 'T':hist_TTM, 'r':hist_r_day, 'sigma':sigma_used, **hist_model_args_common}
                        args_tau = {**base_args, 'tau': hist_tau}; args_no_tau = {**base_args, 'tau': 0.0}
                        mc_t, mp_t, mc_nt, mp_nt = np.nan, np.nan, np.nan, np.nan
                        try:
                            mc_t=pricing_model_func(**args_tau, option_type='call'); mp_t=pricing_model_func(**args_tau, option_type='put')
                            mc_nt=pricing_model_func(**args_no_tau, option_type='call'); mp_nt=pricing_model_func(**args_no_tau, option_type='put')
                        except Exception: pass
                        backtest_results.append({"Date": quote_date_dt, "StockPrice": hist_S, "TTM": hist_TTM, "Rate": hist_r_day, "SigmaUsed": sigma_used, "HistIVCall": iv_call, "HistIVPut": iv_put, "MarketCall": market_call, "MarketPut": market_put, "ModelCall_Tariff": mc_t, "ModelPut_Tariff": mp_t, "ModelCall_NoTariff": mc_nt, "ModelPut_NoTariff": mp_nt})
                        bt_progress.progress((idx + 1) / num_days)
                    bt_progress.empty()
                    if not backtest_results: st.warning("No valid points generated during backtest loop.")
                    else:
                        results_df = pd.DataFrame(backtest_results).set_index("Date"); st.session_state.backtest_results_df = results_df
                        results_df_calls = results_df.dropna(subset=['MarketCall'])
                        if results_df_calls.empty: st.warning("No market call price data.")
                        else:
                            st.subheader("Backtest Results - Call Option"); fig_bt_call = go.Figure(); fig_bt_call.add_trace(go.Scatter(x=results_df_calls.index, y=results_df_calls['MarketCall'], name='Market Call Price', line=dict(color='black', width=1.5))); fig_bt_call.add_trace(go.Scatter(x=results_df_calls.index, y=results_df_calls['ModelCall_Tariff'], name=f'Model Call (τ={hist_tau:.2f})', line=dict(color='blue', width=1.5))); fig_bt_call.add_trace(go.Scatter(x=results_df_calls.index, y=results_df_calls['ModelCall_NoTariff'], name='Model Call (τ=0)', line=dict(color='lightblue', dash='dash', width=1.5))); fig_bt_call.update_layout(title=f"Call Backtest: {ticker_symbol} K={hist_K}, Exp={hist_expiry_str}", yaxis_title="Price ($)", yaxis_tickformat="$,.2f", legend=dict(orientation="h", yanchor="bottom", y=1.02), hovermode="x unified"); st.plotly_chart(fig_bt_call, use_container_width=True)
                            results_df_calls['Error_Tariff'] = results_df_calls['ModelCall_Tariff'] - results_df_calls['MarketCall']; results_df_calls['Error_NoTariff'] = results_df_calls['ModelCall_NoTariff'] - results_df_calls['MarketCall']
                            errors_t=results_df_calls['Error_Tariff'].dropna(); errors_nt=results_df_calls['Error_NoTariff'].dropna()
                            if not errors_t.empty: mae_t=errors_t.abs().mean(); rmse_t=np.sqrt((errors_t**2).mean()); mae_nt=errors_nt.abs().mean(); rmse_nt=np.sqrt((errors_nt**2).mean()); st.subheader("Pricing Error Analysis (Calls: Model - Market)"); err_col1, err_col2 = st.columns(2); err_col1.metric(f"MAE (τ={hist_tau:.2f})", f"${mae_t:.4f}", f"RMSE: ${rmse_t:.4f}"); err_col2.metric("MAE (τ=0)", f"${mae_nt:.4f}", f"RMSE: ${rmse_nt:.4f}"); fig_err_call = go.Figure(); fig_err_call.add_trace(go.Scatter(x=errors_t.index, y=errors_t, name=f'Error (τ={hist_tau:.2f})', line=dict(color='red'))); fig_err_call.add_trace(go.Scatter(x=errors_nt.index, y=errors_nt, name='Error (τ=0)', line=dict(color='pink', dash='dash'))); fig_err_call.add_hline(y=0, line=dict(color='black', width=1)); fig_err_call.update_layout(title="Call Pricing Error Over Time", yaxis_title="Error ($)", yaxis_tickformat="$,.3f", legend=dict(orientation="h", yanchor="bottom", y=1.02), hovermode="x unified"); st.plotly_chart(fig_err_call, use_container_width=True)
                            else: st.warning("Could not calculate call errors.")
                            with st.expander("View Call Backtest Data"): cols=['StockPrice','TTM','Rate','SigmaUsed','HistIVCall','MarketCall','ModelCall_Tariff','ModelCall_NoTariff','Error_Tariff','Error_NoTariff']; fmt={'StockPrice':'${:,.2f}','TTM':'{:.4f}','Rate':'{:.4%}','SigmaUsed':'{:.4%}','HistIVCall':'{:.4%}','MarketCall':'${:,.3f}','ModelCall_Tariff':'${:,.3f}','ModelCall_NoTariff':'${:,.3f}','Error_Tariff':'{:+,.3f}','Error_NoTariff':'{:+,.3f}'}; st.dataframe(results_df_calls[cols].style.format(fmt, na_rep="N/A"))
                        st.markdown("---")
                        results_df_puts = results_df.dropna(subset=['MarketPut'])
                        if results_df_puts.empty: st.warning("No market put price data.")
                        else:
                            st.subheader("Backtest Results - Put Option"); fig_bt_put = go.Figure(); fig_bt_put.add_trace(go.Scatter(x=results_df_puts.index, y=results_df_puts['MarketPut'], name='Market Put Price', line=dict(color='black', width=1.5))); fig_bt_put.add_trace(go.Scatter(x=results_df_puts.index, y=results_df_puts['ModelPut_Tariff'], name=f'Model Put (τ={hist_tau:.2f})', line=dict(color='green', width=1.5))); fig_bt_put.add_trace(go.Scatter(x=results_df_puts.index, y=results_df_puts['ModelPut_NoTariff'], name='Model Put (τ=0)', line=dict(color='lightgreen', dash='dash', width=1.5))); fig_bt_put.update_layout(title=f"Put Backtest: {ticker_symbol} K={hist_K}, Exp={hist_expiry_str}", yaxis_title="Price ($)", yaxis_tickformat="$,.2f", legend=dict(orientation="h", yanchor="bottom", y=1.02), hovermode="x unified"); st.plotly_chart(fig_bt_put, use_container_width=True)
                            results_df_puts['Error_Tariff'] = results_df_puts['ModelPut_Tariff'] - results_df_puts['MarketPut']; results_df_puts['Error_NoTariff'] = results_df_puts['ModelPut_NoTariff'] - results_df_puts['MarketPut']
                            errors_t=results_df_puts['Error_Tariff'].dropna(); errors_nt=results_df_puts['Error_NoTariff'].dropna()
                            if not errors_t.empty: mae_t=errors_t.abs().mean(); rmse_t=np.sqrt((errors_t**2).mean()); mae_nt=errors_nt.abs().mean(); rmse_nt=np.sqrt((errors_nt**2).mean()); st.subheader("Pricing Error Analysis (Puts: Model - Market)"); err_col1p, err_col2p = st.columns(2); err_col1p.metric(f"MAE (τ={hist_tau:.2f})", f"${mae_t:.4f}", f"RMSE: ${rmse_t:.4f}"); err_col2p.metric("MAE (τ=0)", f"${mae_nt:.4f}", f"RMSE: ${rmse_nt:.4f}")
                            else: st.warning("Could not calculate put errors.")
                            with st.expander("View Put Backtest Data"): cols=['StockPrice','TTM','Rate','SigmaUsed','HistIVPut','MarketPut','ModelPut_Tariff','ModelPut_NoTariff','Error_Tariff','Error_NoTariff']; fmt={'StockPrice':'${:,.2f}','TTM':'{:.4f}','Rate':'{:.4%}','SigmaUsed':'{:.4%}','HistIVPut':'{:.4%}','MarketPut':'${:,.3f}','ModelPut_Tariff':'${:,.3f}','ModelPut_NoTariff':'${:,.3f}','Error_Tariff':'{:+,.3f}','Error_NoTariff':'{:+,.3f}'}; st.dataframe(results_df_puts[cols].style.format(fmt, na_rep="N/A"))

        elif run_backtest:
             with col5b: st.error("Backtest could not run. Check configuration and API readiness.")
        else:
             with col5b: st.info("Configure backtest parameters and click 'Run Backtest Analysis'.")
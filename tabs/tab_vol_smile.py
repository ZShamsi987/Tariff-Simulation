# tabs/tab_vol_smile.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
# FIX: Import Any
from typing import Optional, Dict, Any

# Import necessary functions from utils
from utils import (
    calculate_implied_volatility,
    MIN_PRICE_LEVEL,
    MIN_TIME_LEVEL
)

def render_tab_vol_smile(
    st_tab: Any, # FIX: Keep type hint
    ticker_symbol: str,
    current_S: Optional[float],
    r: float,
    sigma_base: float,
    # Get data from session state, set by live analysis tab
    ):
    """Renders the content for the Volatility Smile tab."""
    with st_tab:
        st.header("Volatility Smile / Skew")
        st.markdown("Displays Implied Volatility (calculated using standard BSM) against Strike Price for the selected expiry.")

        # Retrieve necessary data from session state
        option_chain_data_dict = st.session_state.get('option_chain_data_dict', None)
        T_opt = st.session_state.get('selected_T_opt', None)
        expiry_str = st.session_state.get('selected_expiry_str', None)

        # Check if necessary data is available and valid
        smile_inputs_ok = (option_chain_data_dict is not None) and \
                          (T_opt is not None and T_opt > MIN_TIME_LEVEL) and \
                          (isinstance(current_S, (int, float)) and current_S > MIN_PRICE_LEVEL) and \
                          (isinstance(r, (int, float))) and \
                          (expiry_str is not None)

        if not smile_inputs_ok:
            st.warning("Please select a valid Ticker and future Expiry on the 'Live Analysis' tab first. Ensure core inputs (S, T, r) are valid.")
        else:
            calls_df_sm = option_chain_data_dict.get('calls', pd.DataFrame())
            puts_df_sm = option_chain_data_dict.get('puts', pd.DataFrame())

            strikes_c_sm = pd.to_numeric(calls_df_sm['strike'], errors='coerce').dropna() if not calls_df_sm.empty else pd.Series(dtype=float)
            strikes_p_sm = pd.to_numeric(puts_df_sm['strike'], errors='coerce').dropna() if not puts_df_sm.empty else pd.Series(dtype=float)
            all_strikes_sm = pd.concat([strikes_c_sm, strikes_p_sm]).unique()

            if len(all_strikes_sm) == 0:
                 st.warning(f"No valid strikes found in the option chain for expiry {expiry_str}.")
            else:
                # ... (Rest of the function remains exactly the same as previous version) ...
                unique_strikes_sm = sorted(all_strikes_sm)
                atm_strike_sm = min(unique_strikes_sm, key=lambda x: abs(x - current_S))
                try: atm_index_sm = unique_strikes_sm.index(atm_strike_sm)
                except ValueError: atm_index_sm = len(unique_strikes_sm) // 2
                max_range = min(30, len(unique_strikes_sm) // 2)
                num_strikes_half = st.slider("Number of Strikes +/- ATM to Plot", 1, max_range, min(15, max_range), 1, key="sm_range", help="Select how many strikes around ATM to include.")
                min_idx = max(0, atm_index_sm - num_strikes_half)
                max_idx = min(len(unique_strikes_sm), atm_index_sm + num_strikes_half + 1)
                strikes_to_plot = unique_strikes_sm[min_idx:max_idx]
                if st.button("Calculate & Plot Volatility Smile", key="plot_smile_button"):
                    smile_data = []
                    iv_args_base = {'S': current_S, 'T': T_opt, 'r': r}
                    progress_bar = st.progress(0); total_strikes = len(strikes_to_plot)
                    for i, k_smile in enumerate(strikes_to_plot):
                        iv_args_k = {**iv_args_base, 'K': k_smile}
                        call_row_sm = calls_df_sm[np.isclose(calls_df_sm['strike'], k_smile)].iloc[0] if not calls_df_sm[np.isclose(calls_df_sm['strike'], k_smile)].empty else None
                        put_row_sm = puts_df_sm[np.isclose(puts_df_sm['strike'], k_smile)].iloc[0] if not puts_df_sm[np.isclose(puts_df_sm['strike'], k_smile)].empty else None
                        iv_c_calc, iv_p_calc = np.nan, np.nan; iv_c_yf, iv_p_yf = np.nan, np.nan
                        mkt_c_price, mkt_p_price = np.nan, np.nan
                        if call_row_sm is not None:
                            mkt_c_price = call_row_sm.get('mid', call_row_sm.get('lastPrice')); iv_c_yf = call_row_sm.get('impliedVolatility')
                            if pd.notna(mkt_c_price) and mkt_c_price > MIN_PRICE_LEVEL: iv_c_calc = calculate_implied_volatility(mkt_c_price, **iv_args_k, option_type='call')
                        if put_row_sm is not None:
                            mkt_p_price = put_row_sm.get('mid', put_row_sm.get('lastPrice')); iv_p_yf = put_row_sm.get('impliedVolatility')
                            if pd.notna(mkt_p_price) and mkt_p_price > MIN_PRICE_LEVEL: iv_p_calc = calculate_implied_volatility(mkt_p_price, **iv_args_k, option_type='put')
                        if pd.notna(iv_c_calc) or pd.notna(iv_p_calc) or pd.notna(iv_c_yf) or pd.notna(iv_p_yf):
                            smile_data.append({'K': k_smile, 'MktCall': mkt_c_price, 'IV_Call_Calc': iv_c_calc, 'IV_Call_Src': iv_c_yf, 'MktPut': mkt_p_price, 'IV_Put_Calc': iv_p_calc, 'IV_Put_Src': iv_p_yf})
                        progress_bar.progress((i + 1) / total_strikes)
                    if smile_data: st.session_state.smile_df = pd.DataFrame(smile_data).set_index('K')
                    else: st.session_state.smile_df = pd.DataFrame(); st.info("No valid IV data calculated/found.")
                    progress_bar.empty()
                df_plot = st.session_state.get('smile_df', pd.DataFrame())
                if not df_plot.empty:
                    fig_smile = go.Figure()
                    fig_smile.add_trace(go.Scatter(x=df_plot.index, y=df_plot['IV_Call_Calc'], mode='lines+markers', name='IV Call (Calc. BSM)', line=dict(color='blue', width=2), marker=dict(size=6)))
                    fig_smile.add_trace(go.Scatter(x=df_plot.index, y=df_plot['IV_Put_Calc'], mode='lines+markers', name='IV Put (Calc. BSM)', line=dict(color='red', width=2), marker=dict(size=6)))
                    if 'IV_Call_Src' in df_plot.columns and df_plot['IV_Call_Src'].notna().any():
                         fig_smile.add_trace(go.Scatter(x=df_plot.index, y=df_plot['IV_Call_Src'], mode='markers', name='IV Call (Source)', marker=dict(symbol='circle-open', size=8, color='lightblue', line=dict(width=1))))
                    if 'IV_Put_Src' in df_plot.columns and df_plot['IV_Put_Src'].notna().any():
                         fig_smile.add_trace(go.Scatter(x=df_plot.index, y=df_plot['IV_Put_Src'], mode='markers', name='IV Put (Source)', marker=dict(symbol='cross-open', size=8, color='lightcoral', line=dict(width=1))))
                    if sigma_base is not None: fig_smile.add_hline(y=sigma_base, line=dict(dash='dash', width=1.5, color='grey'), annotation_text=f"Base σ ({sigma_base:.1%})", annotation_position="bottom right")
                    if current_S is not None: fig_smile.add_vline(x=current_S, line=dict(dash='dot', width=1.5, color='black'), annotation_text=f"Current S (${current_S:.2f})", annotation_position="top left")
                    fig_smile.update_layout(title=f"{ticker_symbol} Volatility Smile/Skew | Expiry: {expiry_str} | T = {T_opt:.3f} yrs", xaxis_title="Strike Price (K)", yaxis_title="Implied Volatility (Annualized)", yaxis_tickformat=".1%", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), hovermode="x unified")
                    st.plotly_chart(fig_smile, use_container_width=True)
                    with st.expander("View Plotted Smile Data"):
                        st.dataframe(df_plot.style.format({'MktCall':'${:,.2f}', 'MktPut':'${:,.2f}', 'IV_Call_Calc':'{:.2%}', 'IV_Put_Calc':'{:.2%}', 'IV_Call_Src':'{:.2%}', 'IV_Put_Src':'{:.2%}'}, na_rep="N/A"))
                elif 'smile_df' in st.session_state:
                    st.info("Click 'Calculate & Plot Volatility Smile' to generate the graph.")
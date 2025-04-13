# tabs/tab_sensitivity.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
# FIX: Import Any
from typing import Optional, Dict, Any, Callable

# Import necessary functions from utils
from utils import (
    MIN_PRICE_LEVEL,
    MIN_TIME_LEVEL,
    DEFAULT_S_FALLBACK
)

def render_tab_sensitivity(
    st_tab: Any, # FIX: Keep type hint
    pricing_model_name: str,
    pricing_model_func: Callable,
    current_S: Optional[float],
    r: float,
    sigma_base: float,
    tau: float,
    model_args_common: Dict[str, Any],
    req_jumps: bool
    ):
    """Renders the content for the Sensitivity Analysis tab."""
    with st_tab:
        st.header(f"Sensitivity Analysis ({pricing_model_name})")
        st.markdown("Analyze how the theoretical option price changes when one input parameter varies, holding others constant.")

        # ... (Rest of the function remains exactly the same as previous version) ...
        base_K_sens = st.session_state.get('selected_K', None)
        base_T_sens = st.session_state.get('selected_T_opt', None)
        if base_K_sens is None or not isinstance(base_K_sens, (int, float)) or base_K_sens <= MIN_PRICE_LEVEL:
             if isinstance(current_S, (int, float)) and current_S > 0:
                 step = 1.0 if current_S < 50 else (5.0 if current_S < 500 else 10.0)
                 base_K_sens = round(current_S / step) * step
             else: base_K_sens = DEFAULT_S_FALLBACK
        if base_T_sens is None or not isinstance(base_T_sens, (int, float)) or base_T_sens < 0:
             base_T_sens = 0.25
        lambda_sens = model_args_common.get('lambda_sensitivity', 0.0)
        sens_inputs_ok = all(isinstance(v, (int, float)) and not np.isnan(v) for v in [current_S, base_K_sens, base_T_sens, r, sigma_base, tau, lambda_sens]) and \
                         current_S > MIN_PRICE_LEVEL and base_K_sens > MIN_PRICE_LEVEL and base_T_sens >= 0 and sigma_base >= 0
        if req_jumps:
             jump_params = {k: v for k, v in model_args_common.items() if k.startswith('jump_')}
             jump_inputs_valid_sens = all(isinstance(v, (int, float)) and not np.isnan(v) for k, v in jump_params.items())
             sens_inputs_ok = sens_inputs_ok and jump_inputs_valid_sens
        else: jump_params = {}
        if not sens_inputs_ok:
            st.warning("Base parameters invalid. Check 'Live Analysis' and sidebar. Sensitivity analysis may fail.")
        else:
            st.caption(f"Base Parameters: S=${current_S:.2f}, K=${base_K_sens:.2f}, T={base_T_sens:.3f} yrs, r={r:.3%}, σ={sigma_base:.3%}, τ={tau:.2f}, λ={lambda_sens:.2f}")
            if req_jumps: st.caption(f"Jump Params: λ_j={jump_params.get('jump_intensity',0):.1f}, μ_j={jump_params.get('jump_mean',0):.3f}, σ_j={jump_params.get('jump_vol',0):.3f}")
        col3a, col3b = st.columns([1, 3])
        with col3a:
            st.subheader("Vary Parameter")
            params_available_sens = {"Stock Price (S)":'S', "Tariff Rate (τ)":'tau', "Volatility (σ)":'sigma', "Time to Expiry (T)":'T', "Interest Rate (r)":'r', "Tariff Sensitivity (λ)":'lambda_sensitivity'}
            if req_jumps: params_available_sens.update({"Jump Intensity (λ_j)":'jump_intensity', "Mean Jump Size (μ_j)":'jump_mean', "Jump Volatility (σ_j)":'jump_vol'})
            selected_param_name = st.selectbox("Select Parameter to Vary:", list(params_available_sens.keys()), key="sens_param_select")
            vary_key = params_available_sens[selected_param_name]
            base_values_sens = {'S': current_S, 'K': base_K_sens, 'T': base_T_sens, 'r': r, 'sigma': sigma_base, 'tau': tau, **model_args_common }
            current_param_value = base_values_sens.get(vary_key)
            sens_range_pct = st.slider(f"Variation Range (%)", 1, 100, 30, 1, key="sens_range_pct", help=f"Range around base ({current_param_value:.3f}).") / 100.0
            sens_steps = st.slider("Number of Steps", 11, 101, 21, 2, key="sens_steps")
            param_values_range = []
            if isinstance(current_param_value, (int, float)):
                 delta = abs(current_param_value * sens_range_pct); delta = max(delta, 1e-6)
                 if vary_key in ['S','K']: min_val=max(MIN_PRICE_LEVEL, current_param_value-delta); max_val=current_param_value+delta
                 elif vary_key == 'T': min_val=max(MIN_TIME_LEVEL, current_param_value-delta); max_val=current_param_value+delta
                 elif vary_key in ['sigma','lambda_sensitivity','jump_intensity','jump_vol']: min_val=max(0.0, current_param_value-delta); max_val=current_param_value+delta
                 elif vary_key == 'tau': min_val=max(0.0, current_param_value-delta); max_val=min(1.0, current_param_value+delta)
                 else: min_val=current_param_value-delta; max_val=current_param_value+delta
                 if min_val >= max_val: st.warning(f"Invalid range for {selected_param_name}.")
                 else: param_values_range = np.linspace(min_val, max_val, sens_steps)
            else: st.error(f"Base value for {selected_param_name} non-numeric.")
            run_sensitivity = st.button("Run Sensitivity Analysis", key="run_sensitivity_button", disabled=(not sens_inputs_ok or len(param_values_range)==0))
        with col3b:
            st.subheader("Sensitivity Plot")
            if run_sensitivity and sens_inputs_ok and len(param_values_range) > 0:
                 call_prices_sens, put_prices_sens = [], []
                 progress_bar_sens = st.progress(0)
                 for i, p_val in enumerate(param_values_range):
                     iter_args = base_values_sens.copy(); iter_args[vary_key] = p_val
                     iter_args['K'] = base_K_sens; iter_args['T'] = base_T_sens; iter_args['S'] = current_S
                     iter_args[vary_key] = p_val
                     args_call = {**iter_args, 'option_type': 'call'}; args_put = {**iter_args, 'option_type': 'put'}
                     try:
                         call_prices_sens.append(pricing_model_func(**args_call))
                         put_prices_sens.append(pricing_model_func(**args_put))
                     except Exception: call_prices_sens.append(np.nan); put_prices_sens.append(np.nan)
                     progress_bar_sens.progress((i + 1) / sens_steps)
                 progress_bar_sens.empty()
                 if call_prices_sens and put_prices_sens:
                     fig_sens = go.Figure()
                     fig_sens.add_trace(go.Scatter(x=param_values_range, y=call_prices_sens, mode='lines+markers', name='Call Price', line=dict(color='blue')))
                     fig_sens.add_trace(go.Scatter(x=param_values_range, y=put_prices_sens, mode='lines+markers', name='Put Price', line=dict(color='red')))
                     if isinstance(current_param_value, (int, float)): fig_sens.add_vline(x=current_param_value, line=dict(dash='dash', color='grey', width=1.5), annotation_text=f"Base ({current_param_value:.3f})", annotation_position="bottom right")
                     fig_sens.update_layout(title=f"Option Price Sensitivity to {selected_param_name}", xaxis_title=selected_param_name, yaxis_title="Option Price ($)", yaxis_tickformat="$,.2f", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), hovermode="x unified")
                     st.plotly_chart(fig_sens, use_container_width=True)
                     sens_df = pd.DataFrame({selected_param_name: param_values_range, 'Call Price': call_prices_sens, 'Put Price': put_prices_sens})
                     with st.expander("View Sensitivity Data"): st.dataframe(sens_df.style.format({selected_param_name: '{:.4f}', 'Call Price': '${:,.3f}', 'Put Price': '${:,.3f}'}, na_rep="N/A"))
                 else: st.warning("No sensitivity data calculated.")
            elif run_sensitivity: st.error("Cannot run Sensitivity Analysis: Invalid base parameters or range.")
            else: st.info("Select parameter, adjust range/steps, and click 'Run Sensitivity Analysis'.")
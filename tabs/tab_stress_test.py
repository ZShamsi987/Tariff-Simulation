# tabs/tab_stress_test.py
import streamlit as st
import numpy as np
import pandas as pd
# FIX: Import Any
from typing import Optional, Dict, Any, Callable

# Import necessary functions from utils
from utils import (
    MIN_PRICE_LEVEL,
    DEFAULT_S_FALLBACK
)

def format_price_delta(shock_price, base_price):
    """Helper to format price changes for st.metric delta."""
    shock_price = float(shock_price) if pd.notna(shock_price) else np.nan
    base_price = float(base_price) if pd.notna(base_price) else np.nan
    if pd.notna(shock_price) and pd.notna(base_price):
        delta = shock_price - base_price; delta_pct_str = ""
        if not np.isclose(base_price, 0, atol=1e-6):
            delta_pct = (delta / base_price) * 100; delta_pct_str = f" ({delta_pct:+.1f}%)"
        return f"{delta:+.3f}{delta_pct_str}"
    return "N/A"

def render_tab_stress_test(
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
    """Renders the content for the Stress Test tab."""
    with st_tab:
        st.header(f"Stress Testing ({pricing_model_name})")
        st.markdown("Evaluate the impact of simultaneous shocks to multiple parameters on the option price.")

        # ... (Rest of the function remains exactly the same as previous version) ...
        base_K_stress = st.session_state.get('selected_K', None)
        base_T_stress = st.session_state.get('selected_T_opt', None)
        if base_K_stress is None or not isinstance(base_K_stress, (int, float)) or base_K_stress <= MIN_PRICE_LEVEL:
             if isinstance(current_S, (int, float)) and current_S > 0:
                 step = 1.0 if current_S < 50 else (5.0 if current_S < 500 else 10.0)
                 base_K_stress = round(current_S / step) * step
             else: base_K_stress = DEFAULT_S_FALLBACK
        if base_T_stress is None or not isinstance(base_T_stress, (int, float)) or base_T_stress < 0:
             base_T_stress = 0.25
        lambda_sens = model_args_common.get('lambda_sensitivity', 0.0)
        stress_inputs_ok = all(isinstance(v, (int, float)) and not np.isnan(v) for v in [current_S, base_K_stress, base_T_stress, r, sigma_base, tau, lambda_sens]) and \
                           current_S > MIN_PRICE_LEVEL and base_K_stress > MIN_PRICE_LEVEL and base_T_stress >= 0 and sigma_base >= 0
        if req_jumps:
             jump_params = {k: v for k, v in model_args_common.items() if k.startswith('jump_')}
             jump_inputs_valid_stress = all(isinstance(v, (int, float)) and not np.isnan(v) for k, v in jump_params.items())
             stress_inputs_ok = stress_inputs_ok and jump_inputs_valid_stress
        else: jump_params = {}
        if not stress_inputs_ok: st.warning("Base parameters invalid. Stress test results may be inaccurate.")
        col4a, col4b = st.columns(2)
        with col4a:
            st.subheader("Define Stress Scenario")
            s_shock_pct = st.slider("Stock Price (S) Shock (%)", -50.0, 50.0, 0.0, 1.0, format="%.1f%%", key="st_s_pct") / 100.0
            sig_shock_pct = st.slider("Volatility (σ) Shock (%)", -80.0, 200.0, 0.0, 5.0, format="%.1f%%", key="st_sig_pct") / 100.0
            tau_shock_abs = st.slider("Tariff (τ) Shock (Abs)", min_value=max(-tau, -1.0), max_value=min(1.0 - tau, 1.0), value=0.0, step=0.01, format="%+.3f", key="st_tau_abs")
            r_shock_abs = st.slider("Interest Rate (r) Shock (Abs)", -0.05, 0.05, 0.0, 0.001, format="%+.4f", key="st_r_abs")
            lambda_shock_abs = st.slider("Tariff Sensitivity (λ) Shock (Abs)", -0.2, 0.2, 0.0, 0.01, format="%+.3f", key="st_lambda_abs")
            shocked_jump_params = jump_params.copy()
            if req_jumps:
                st.markdown("---"); st.markdown("**Jump Parameter Shocks:**")
                ji_base = jump_params.get('jump_intensity', 0); jm_base = jump_params.get('jump_mean', 0); jv_base = jump_params.get('jump_vol', 0)
                ji_shock_pct = st.slider("Jump Intensity (λ_j) Shock (%)", -80.0, 200.0, 0.0, 5.0, format="%.1f%%", key="st_ji_pct") / 100.0
                jm_shock_abs = st.slider("Mean Jump Size (μ_j) Shock (Abs)", -0.1, 0.1, 0.0, 0.005, format="%+.3f", key="st_jm_abs")
                jv_shock_pct = st.slider("Jump Volatility (σ_j) Shock (%)", -80.0, 200.0, 0.0, 5.0, format="%.1f%%", key="st_jv_pct") / 100.0
                shocked_jump_params['jump_intensity'] = max(0.0, ji_base * (1 + ji_shock_pct))
                shocked_jump_params['jump_mean'] = jm_base + jm_shock_abs
                shocked_jump_params['jump_vol'] = max(0.0, jv_base * (1 + jv_shock_pct))
        price_call_base, price_put_base = np.nan, np.nan
        price_call_shock, price_put_shock = np.nan, np.nan
        if stress_inputs_ok:
            base_args_stress = {'S': current_S, 'K': base_K_stress, 'T': base_T_stress, 'r': r, 'sigma': sigma_base, 'tau': tau, **model_args_common}
            shocked_S = max(MIN_PRICE_LEVEL, current_S * (1 + s_shock_pct))
            shocked_sigma = max(0.0, sigma_base * (1 + sig_shock_pct))
            shocked_tau = max(0.0, min(1.0, tau + tau_shock_abs))
            shocked_r = r + r_shock_abs
            shocked_lambda_sens = max(0.0, lambda_sens + lambda_shock_abs)
            shocked_model_args_common = {'lambda_sensitivity': shocked_lambda_sens, **shocked_jump_params}
            shock_args_stress = {'S': shocked_S, 'K': base_K_stress, 'T': base_T_stress, 'r': shocked_r, 'sigma': shocked_sigma, 'tau': shocked_tau, **shocked_model_args_common}
            with st.spinner("Calculating base and stressed prices..."):
                 try:
                    price_call_base = pricing_model_func(**base_args_stress, option_type='call')
                    price_put_base = pricing_model_func(**base_args_stress, option_type='put')
                    price_call_shock = pricing_model_func(**shock_args_stress, option_type='call')
                    price_put_shock = pricing_model_func(**shock_args_stress, option_type='put')
                 except Exception as e: st.error(f"Error during stress calc: {e}")
        with col4b:
            st.subheader("Stress Test Results")
            st.metric("Base Call Price", f"${price_call_base:,.3f}" if pd.notna(price_call_base) else "N/A")
            st.metric("Stressed Call Price", f"${price_call_shock:,.3f}" if pd.notna(price_call_shock) else "N/A", delta=format_price_delta(price_call_shock, price_call_base), delta_color="normal")
            st.markdown("---")
            st.metric("Base Put Price", f"${price_put_base:,.3f}" if pd.notna(price_put_base) else "N/A")
            st.metric("Stressed Put Price", f"${price_put_shock:,.3f}" if pd.notna(price_put_shock) else "N/A", delta=format_price_delta(price_put_shock, price_put_base), delta_color="normal")
            st.markdown("---"); st.markdown("**Applied Shocks Summary:**")
            shock_desc_lines = []
            if not np.isclose(s_shock_pct, 0): shock_desc_lines.append(f"S: {s_shock_pct:+.1%}")
            if not np.isclose(sig_shock_pct, 0): shock_desc_lines.append(f"σ: {sig_shock_pct:+.1%}")
            if not np.isclose(tau_shock_abs, 0): shock_desc_lines.append(f"τ: {tau_shock_abs:+.3f}")
            if not np.isclose(r_shock_abs, 0): shock_desc_lines.append(f"r: {r_shock_abs:+.4f}")
            if not np.isclose(lambda_shock_abs, 0): shock_desc_lines.append(f"λ: {lambda_shock_abs:+.3f}")
            if req_jumps:
                 if not np.isclose(ji_shock_pct, 0): shock_desc_lines.append(f"λ_j: {ji_shock_pct:+.1%}")
                 if not np.isclose(jm_shock_abs, 0): shock_desc_lines.append(f"μ_j: {jm_shock_abs:+.3f}")
                 if not np.isclose(jv_shock_pct, 0): shock_desc_lines.append(f"σ_j: {jv_shock_pct:+.1%}")
            if shock_desc_lines: st.code(", ".join(shock_desc_lines), language=None)
            else: st.caption("No shocks applied.")
        if not stress_inputs_ok: st.error("Cannot run stress test: Invalid base parameters provided.")
# tabs/tab_3d_surface.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Optional, Dict, Any, Callable

# Import necessary functions from utils
from utils import (
    MIN_PRICE_LEVEL,
    MIN_TIME_LEVEL,
    DEFAULT_S_FALLBACK
)

def render_tab_3d_surface(
    st_tab: Any, 
    pricing_model_name: str,
    pricing_model_func: Callable,
    current_S: Optional[float],
    r: float,
    sigma_base: float,
    tau: float,
    model_args_common: Dict[str, Any],
    req_jumps: bool
    ):
    """Renders the content for the 3D Surface tab."""
    with st_tab:
        st.header(f"3D Option Price Surface ({pricing_model_name})")
        st.markdown("Visualize the theoretical option price across a range of Strike Prices (K) and Times to Maturity (T).")

        base_S_3d = current_S; base_r_3d = r; base_sigma_3d = sigma_base
        base_tau_3d = tau; base_model_args_3d = model_args_common
        base_S_is_valid = isinstance(base_S_3d, (int, float)) and base_S_3d > MIN_PRICE_LEVEL
        inputs_ok_3d = all(isinstance(v,(int,float)) and not np.isnan(v) for v in [base_r_3d, base_sigma_3d, base_tau_3d]) and \
                       base_S_is_valid and base_sigma_3d >= 0 and isinstance(base_model_args_3d.get('lambda_sensitivity'), (int, float))
        if req_jumps:
             jump_params_3d = {k: v for k,v in base_model_args_3d.items() if k.startswith('jump_')}
             inputs_ok_3d = inputs_ok_3d and all(isinstance(v, (int, float)) and not np.isnan(v) for k, v in jump_params_3d.items())
        col6a, col6b = st.columns([1, 3])
        with col6a:
             st.subheader("Surface Parameters")
             plot_type_3d = st.radio("Plot Price:", ["Call", "Put"], key="3d_plot_type", horizontal=True)
             option_type_3d = 'call' if 'Call' in plot_type_3d else 'put'
             k_range_pct_3d = st.slider("Strike Range (%)", 5, 50, 25, 5, key="3d_k_range", help="Range around S.") / 100.0
             k_steps_3d = st.slider("Strike Steps", 10, 50, 25, key="3d_k_steps")
             t_max_years_3d = st.slider("Max T (Years)", 0.1, 2.0, 1.0, 0.1, key="3d_t_max")
             t_steps_3d = st.slider("Time Steps", 10, 50, 25, key="3d_t_steps")
             strikes_3d, times_3d = [], []
             s_for_range = base_S_3d if base_S_is_valid else DEFAULT_S_FALLBACK
             if s_for_range > MIN_PRICE_LEVEL:
                 k_min = max(MIN_PRICE_LEVEL, s_for_range * (1 - k_range_pct_3d))
                 k_max = s_for_range * (1 + k_range_pct_3d)
                 if k_min < k_max: strikes_3d = np.linspace(k_min, k_max, k_steps_3d)
                 else: st.warning("Invalid strike range.")
                 t_min = MIN_TIME_LEVEL * 10
                 if t_min < t_max_years_3d: times_3d = np.linspace(t_min, t_max_years_3d, t_steps_3d)
                 else: st.warning("Invalid time range.")
             else: st.warning("Base S invalid.")
             run_3d_plot = st.button("Generate 3D Surface", key="3d_run_button", disabled=(not inputs_ok_3d or len(strikes_3d)==0 or len(times_3d)==0))
        with col6b:
            st.subheader("Plot")
            if run_3d_plot:
                if not inputs_ok_3d or len(strikes_3d) == 0 or len(times_3d) == 0: st.error("Cannot generate: Invalid base parameters or range.")
                else:
                    K_mesh, T_mesh = np.meshgrid(strikes_3d, times_3d)
                    Z_prices = np.full_like(K_mesh, fill_value=np.nan)
                    progress_bar_3d = st.progress(0); total_calcs = K_mesh.size; calcs_done = 0
                    with st.spinner(f"Calculating {plot_type_3d} surface..."):
                         for i in range(T_mesh.shape[0]):
                             for j in range(K_mesh.shape[1]):
                                 k_val=K_mesh[i,j]; t_val=T_mesh[i,j]
                                 args_3d = {'S':base_S_3d, 'K':k_val, 'T':t_val, 'r':base_r_3d, 'sigma':base_sigma_3d, 'tau':base_tau_3d, **base_model_args_3d}
                                 try: Z_prices[i, j] = pricing_model_func(**args_3d, option_type=option_type_3d)
                                 except Exception: Z_prices[i, j] = np.nan
                                 calcs_done += 1; progress_bar_3d.progress(calcs_done / total_calcs)
                    progress_bar_3d.empty()
                    if np.isnan(Z_prices).all(): st.warning("Could not calculate any valid prices.")
                    else:
                        Z_plot = np.nan_to_num(Z_prices, nan=0.0)
                        fig_3d = go.Figure(data=[go.Surface(z=Z_plot, x=K_mesh, y=T_mesh, colorscale='Viridis', cmin=0, cmax=np.percentile(Z_plot[Z_plot > 1e-6], 98) if np.any(Z_plot > 1e-6) else 1, colorbar=dict(title='Price ($)', tickformat="$,.2f"), contours={"z":{"show":True, "highlight":True, "highlightcolor":"limegreen", "project":{"z":True}, "usecolormap":True}})])
                        fig_3d.update_layout(title=f'3D Surface: {plot_type_3d} Price ({pricing_model_name}) | S={base_S_3d:.2f}, σ={base_sigma_3d:.1%}, τ={base_tau_3d:.1%}', scene=dict(xaxis_title='Strike (K)', yaxis_title='Time (T, yrs)', zaxis_title='Option Price ($)', zaxis_tickformat="$,.2f", camera=dict(eye=dict(x=1.8, y=-1.8, z=0.8))), margin=dict(l=10, r=10, b=10, t=60))
                        st.plotly_chart(fig_3d, use_container_width=True)
            else: st.info("Configure parameters and click 'Generate 3D Surface'.")
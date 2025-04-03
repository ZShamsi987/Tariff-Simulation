import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd # Added for better data handling if needed later

# --- Mathematical Models ---

def economic_model_simulation(T0, tau, mu, sigma, dt, steps, paths):
    """
    Simulates the evolution of trade volume under tariff impact using a geometric Brownian motion model.

    Args:
        T0 (float): Initial trade volume.
        tau (float): Tariff rate (0 to 1).
        mu (float): Baseline drift rate of trade volume.
        sigma (float): Baseline volatility of trade volume.
        dt (float): Time step size.
        steps (int): Number of time steps in the simulation.
        paths (int): Number of simulation paths to generate.

    Returns:
        np.ndarray: A 2D numpy array where rows are simulation paths and columns are time steps.
                    Shape: (paths, steps + 1)
    """
    simulations = np.zeros((paths, steps + 1))
    simulations[:, 0] = T0

    # Adjust drift based on tariff impact (simple linear reduction model)
    # A higher tariff reduces the expected growth rate.
    mu_adj = mu * (1 - tau)

    # Adjust volatility based on tariff impact (optional, e.g., tariffs increase uncertainty)
    # sigma_adj = sigma * (1 + tau * 0.5) # Example: volatility increases slightly with tariff
    sigma_adj = sigma # Keeping sigma constant for this version as per initial plan

    for i in range(1, steps + 1):
        # Generate random shocks for all paths at this step
        dW = np.random.randn(paths) * np.sqrt(dt)

        # Update trade volume using the SDE discretization (Euler-Maruyama)
        # dTt = mu_adj * Tt * dt + sigma_adj * Tt * dWt
        # T_{t+dt} = T_t + mu_adj * T_t * dt + sigma_adj * T_t * dWt
        # T_{t+dt} = T_t * (1 + mu_adj * dt + sigma_adj * dW)
        simulations[:, i] = simulations[:, i-1] * (1 + mu_adj * dt + sigma_adj * dW)

        # Ensure trade volume doesn't go negative (optional floor)
        simulations[:, i] = np.maximum(simulations[:, i], 0)

    return simulations

def modified_black_scholes(S, K, T, r, sigma, tau, option_type='call'):
    """
    Calculates the price of a European option using a modified Black-Scholes formula
    that incorporates a tariff risk premium.

    Args:
        S (float): Current underlying asset price (e.g., related to trade volume or a related index).
        K (float): Strike price of the option.
        T (float): Time to maturity (in years).
        r (float): Risk-free interest rate.
        sigma (float): Volatility of the underlying asset.
        tau (float): Tariff rate, used to calculate the risk premium.
        option_type (str): Type of option, 'call' or 'put'.

    Returns:
        float: The estimated price of the option.
    """
    # Define the tariff risk premium lambda(tau)
    # This is a simple model; could be more complex (e.g., non-linear, dependent on other factors)
    # Example: premium increases linearly with tariff rate
    lambda_tau = tau * 0.1 # This factor (0.1) represents sensitivity to tariff risk

    # Adjust the risk-free rate (or drift term in risk-neutral measure)
    r_adj = r + lambda_tau # Higher tariff implies higher required return/risk

    # Standard Black-Scholes calculations with adjusted rate
    if T <= 0: # Handle expired option
        if option_type == 'call':
            return max(0, S - K)
        else: # put
            return max(0, K - S)
    if sigma <= 0: # Handle zero volatility
        if option_type == 'call':
            return max(0, S * np.exp(lambda_tau * T) - K * np.exp(-r * T))
        else: # put
             return max(0, K * np.exp(-r * T) - S * np.exp(lambda_tau * T))


    d1 = (np.log(S / K) + (r_adj + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2) # Discount strike at original r
        # Alternative: Discount strike at r_adj if premium affects overall discounting
        # price = S * norm.cdf(d1) - K * np.exp(-r_adj * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1) # Discount strike at original r
        # Alternative:
        # price = K * np.exp(-r_adj * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price

# --- Streamlit User Interface ---

def main():
    st.set_page_config(layout="wide") # Use wide layout for better dashboard feel
    st.title("📈 Tariff Impact & Derivatives Risk Simulator")
    st.markdown("""
    This application simulates the impact of tariffs on global trade volumes using a stochastic model
    and evaluates the price of options using a modified Black-Scholes formula incorporating tariff risk.
    Adjust the parameters in the sidebar to see the effects.
    """)

    # --- Sidebar for Input Parameters ---
    st.sidebar.header("Simulation Parameters")
    st.sidebar.markdown("**Economic Model**")
    T0 = st.sidebar.number_input("Initial Trade Volume (T₀)", value=100.0, min_value=0.1, step=10.0, format="%.1f",
                                 help="Starting value for trade volume in the simulation.")
    tau = st.sidebar.slider("Tariff Rate (τ)", min_value=0.0, max_value=1.0, value=0.1, step=0.01, format="%.2f",
                            help="The imposed tariff rate (0% to 100%). Higher tariffs negatively impact trade drift.")
    mu = st.sidebar.slider("Baseline Drift (μ)", min_value=-0.2, max_value=0.5, value=0.05, step=0.01, format="%.2f",
                           help="The expected annual growth rate of trade volume *without* tariffs.")
    sigma_econ = st.sidebar.slider("Trade Volatility (σ_econ)", min_value=0.01, max_value=1.0, value=0.2, step=0.01, format="%.2f",
                                help="The annual volatility (standard deviation) of trade volume changes.")
    T_sim = st.sidebar.number_input("Simulation Horizon (Years)", value=1.0, min_value=0.1, max_value=10.0, step=0.5, format="%.1f",
                                 help="Total time period for the economic simulation.")
    steps = st.sidebar.number_input("Time Steps (per year)", value=100, min_value=10, max_value=1000, step=10,
                                  help="Number of discrete steps within one year for simulation accuracy.")
    paths = st.sidebar.number_input("Simulation Paths", value=1000, min_value=100, max_value=10000, step=100,
                                  help="Number of independent simulation runs for statistical analysis.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Derivatives Pricing**")
    S = st.sidebar.number_input("Current Underlying Price (S)", value=100.0, min_value=0.1, step=1.0, format="%.1f",
                                help="The current price of the asset underlying the option (e.g., a trade index).")
    K = st.sidebar.number_input("Strike Price (K)", value=100.0, min_value=0.1, step=1.0, format="%.1f",
                                help="The price at which the option can be exercised.")
    T_opt = st.sidebar.number_input("Time to Maturity (T_opt, years)", value=1.0, min_value=0.01, max_value=10.0, step=0.1, format="%.2f",
                                 help="The remaining lifespan of the option.")
    r = st.sidebar.slider("Risk-Free Rate (r)", min_value=0.0, max_value=0.2, value=0.03, step=0.005, format="%.3f",
                          help="The annualized risk-free interest rate.")
    sigma_opt = st.sidebar.slider("Option Volatility (σ_opt)", min_value=0.01, max_value=1.0, value=0.25, step=0.01, format="%.2f",
                                help="The implied or historical volatility of the underlying asset for option pricing.")


    # --- Calculations ---
    dt = 1.0 / steps # Time step size based on steps per year
    total_steps = int(T_sim * steps) # Total number of steps for the simulation horizon

    # Run Economic Simulation
    simulations = economic_model_simulation(T0, tau, mu, sigma_econ, dt, total_steps, paths)
    time_axis = np.linspace(0, T_sim, total_steps + 1)

    # Calculate Option Prices
    price_call = modified_black_scholes(S, K, T_opt, r, sigma_opt, tau, option_type='call')
    price_put = modified_black_scholes(S, K, T_opt, r, sigma_opt, tau, option_type='put')

    # Calculate standard BS prices for comparison (tau=0)
    price_call_std = modified_black_scholes(S, K, T_opt, r, sigma_opt, 0.0, option_type='call')
    price_put_std = modified_black_scholes(S, K, T_opt, r, sigma_opt, 0.0, option_type='put')


    # --- Display Results ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Economic Model Simulation")
        st.markdown(f"Simulating trade volume over **{T_sim} years** with **{paths} paths**.")

        fig_econ, ax_econ = plt.subplots(figsize=(10, 6))
        # Plot a subset of paths for clarity
        num_paths_to_plot = min(paths, 50)
        for i in range(num_paths_to_plot):
            ax_econ.plot(time_axis, simulations[i, :], lw=0.5, alpha=0.5)

        # Plot the mean path
        mean_path = np.mean(simulations, axis=0)
        ax_econ.plot(time_axis, mean_path, lw=2, color='red', label=f'Mean Path (τ={tau:.2f})')

        # Plot theoretical mean path without tariff for comparison
        theoretical_mean_no_tariff = T0 * np.exp(mu * time_axis)
        ax_econ.plot(time_axis, theoretical_mean_no_tariff, lw=1.5, color='black', linestyle='--', label=f'Mean Path (τ=0.00)')


        ax_econ.set_xlabel("Time (Years)")
        ax_econ.set_ylabel("Trade Volume")
        ax_econ.set_title(f"Simulated Trade Volume Paths (Tariff τ = {tau:.2f})")
        ax_econ.grid(True, linestyle='--', alpha=0.6)
        ax_econ.legend()
        st.pyplot(fig_econ)

        # Display final distribution histogram
        st.subheader("Distribution at Simulation End")
        final_volumes = simulations[:, -1]
        fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
        ax_hist.hist(final_volumes, bins=50, density=True, alpha=0.7, label='Final Volume Distribution')
        mean_final = np.mean(final_volumes)
        std_final = np.std(final_volumes)
        ax_hist.axvline(mean_final, color='red', linestyle='-', lw=2, label=f'Mean: {mean_final:.2f}')
        ax_hist.set_xlabel("Final Trade Volume")
        ax_hist.set_ylabel("Density")
        ax_hist.set_title(f"Distribution of Trade Volume at T = {T_sim:.1f} Years")
        ax_hist.legend()
        ax_hist.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig_hist)
        st.markdown(f"**Statistics at T={T_sim:.1f}:** Mean = {mean_final:.2f}, Std Dev = {std_final:.2f}")


    with col2:
        st.subheader("Modified Option Pricing")
        st.markdown(f"Calculating option prices with tariff risk premium (τ={tau:.2f}).")

        # Display calculated prices
        st.metric(label="Modified Call Option Price", value=f"{price_call:.4f}", delta=f"{price_call - price_call_std:.4f} vs Std BS",
                  help="Call price including tariff risk premium. Delta shows the difference compared to standard Black-Scholes.")
        st.metric(label="Modified Put Option Price", value=f"{price_put:.4f}", delta=f"{price_put - price_put_std:.4f} vs Std BS",
                   help="Put price including tariff risk premium. Delta shows the difference compared to standard Black-Scholes.")

        st.markdown("---")
        st.markdown("**Standard Black-Scholes (τ=0) for Comparison:**")
        st.write(f"Standard Call Price: {price_call_std:.4f}")
        st.write(f"Standard Put Price: {price_put_std:.4f}")

        # Placeholder for Sensitivity Analysis / PDE Solver Results
        st.subheader("Further Analysis (Placeholders)")
        st.markdown("""
        * **Sensitivity Analysis:** How do option prices change with varying tariff rates or volatility? (Could add plots here).
        * **PDE Solver:** Implement and compare results from a numerical PDE solver for the modified Black-Scholes equation.
        * **Risk Metrics:** Calculate Value-at-Risk (VaR) or Expected Shortfall (ES) from the simulated trade volume distributions.
        """)

    # --- Documentation/Explanation ---
    st.markdown("---")
    st.subheader("Model Explanations")
    with st.expander("Economic Model (Geometric Brownian Motion with Tariff Adjustment)"):
        st.markdown(r"""
        The trade volume $T_t$ is modeled using a Stochastic Differential Equation (SDE):
        $$ d T_t = \mu_{adj}(T_t, \tau) T_t dt + \sigma_{econ}(T_t, \tau) T_t dW_t $$
        Where:
        - $T_t$: Trade volume at time $t$.
        - $\mu_{adj}$: Adjusted drift rate, incorporating the tariff impact. Here, $\mu_{adj} = \mu (1 - \tau)$.
        - $\sigma_{econ}$: Volatility of trade volume. Assumed constant here, but could depend on $\tau$.
        - $\tau$: Tariff rate.
        - $W_t$: A standard Wiener process (Brownian motion) representing random shocks.
        - $dt$: Infinitesimal time step.

        The simulation uses the Euler-Maruyama method to discretize this SDE:
        $$ T_{t+\Delta t} \approx T_t (1 + \mu_{adj} \Delta t + \sigma_{econ} \sqrt{\Delta t} Z) $$
        where $Z$ is a standard normal random variable.
        """)

    with st.expander("Modified Black-Scholes Model"):
        st.markdown(r"""
        The standard Black-Scholes model assumes constant risk-free rate $r$ and volatility $\sigma$. To incorporate tariff risk, we introduce a risk premium $\lambda(\tau)$ that depends on the tariff rate $\tau$. This premium adjusts the drift term in the risk-neutral pricing framework.

        A simple model for the premium is $\lambda(\tau) = c \cdot \tau$, where $c$ is a sensitivity factor (here, $c=0.1$).

        The adjusted risk-neutral drift becomes $r_{adj} = r + \lambda(\tau)$. The Black-Scholes formula is then applied using this adjusted rate $r_{adj}$ in the $d_1$ and $d_2$ calculations:
        $$ d_1 = \frac{\ln(S/K) + (r_{adj} + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}} $$
        $$ d_2 = d_1 - \sigma\sqrt{T} $$
        The call and put price formulas remain structurally similar, but use these modified $d_1$ and $d_2$. Note: The discounting factor $e^{-rT}$ typically still uses the original risk-free rate $r$, as $\lambda(\tau)$ represents an adjustment to the expected *growth* under the risk-neutral measure, not necessarily the time value of money itself. (This interpretation can vary).

        The corresponding Partial Differential Equation (PDE) would be:
        $$ \frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + (r + \lambda(\tau))S \frac{\partial V}{\partial S} - rV = 0 $$
        This PDE could be solved numerically (e.g., using finite differences) for more complex scenarios where analytical solutions are not available.
        """)


if __name__ == '__main__':
    main()

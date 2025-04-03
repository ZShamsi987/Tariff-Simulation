import streamlit as st
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt # No longer needed directly if using Plotly exclusively
import plotly.graph_objects as go
from scipy.stats import norm
import yfinance as yf
from datetime import datetime, timedelta
import pickle # Used for checking serializability if needed for debugging

# --- Configuration & Constants ---
DEFAULT_TICKER = "SPY" # Default ticker (e.g., S&P 500 ETF)

# --- Mathematical Models ---

def economic_model_simulation(T0, tau, mu, sigma_econ, dt, steps, paths):
    """
    Simulates the evolution of a generic economic indicator (e.g., trade volume proxy)
    under tariff impact using a geometric Brownian motion model.
    This simulation primarily informs the *potential impact* level, represented by tau.

    Args:
        T0 (float): Initial indicator value.
        tau (float): Tariff rate (0 to 1). Used conceptually to justify lambda_tau.
        mu (float): Baseline drift rate of the indicator.
        sigma_econ (float): Baseline volatility of the indicator.
        dt (float): Time step size.
        steps (int): Number of time steps in the simulation.
        paths (int): Number of simulation paths to generate.

    Returns:
        np.ndarray: A 2D numpy array of simulation paths. Shape: (paths, steps + 1)
        np.ndarray: The time axis for the simulation.
    """
    simulations = np.zeros((paths, steps + 1))
    simulations[:, 0] = T0
    mu_adj = mu * (1 - tau) # Adjusted drift based on tariff
    sigma_adj = sigma_econ # Keeping volatility constant for simplicity

    for i in range(1, steps + 1):
        dW = np.random.randn(paths) * np.sqrt(dt)
        simulations[:, i] = simulations[:, i-1] * (1 + mu_adj * dt + sigma_adj * dW)
        simulations[:, i] = np.maximum(simulations[:, i], 0) # Floor at zero

    time_axis = np.linspace(0, dt * steps, steps + 1)
    return simulations, time_axis

def modified_black_scholes(S, K, T, r, sigma, tau, option_type='call', lambda_sensitivity=0.1):
    """
    Calculates the price of a European option using a modified Black-Scholes formula
    that incorporates a tariff risk premium lambda(tau).

    Args:
        S (float): Current underlying asset price.
        K (float): Strike price.
        T (float): Time to maturity (years).
        r (float): Risk-free interest rate.
        sigma (float): Volatility of the underlying asset.
        tau (float): Tariff rate (used to calculate risk premium).
        option_type (str): 'call' or 'put'.
        lambda_sensitivity (float): Factor determining how much tariff affects the risk premium.

    Returns:
        float: The estimated price of the option. Returns np.nan if inputs are invalid.
    """
    # Input validation
    if T < 0 or sigma < 0 or S <= 0 or K <= 0:
        st.warning(f"Invalid input to BSM: S={S}, K={K}, T={T}, sigma={sigma}")
        return np.nan # Invalid inputs lead to undefined price

    if T == 0: # Option expired
        return max(0, S - K) if option_type == 'call' else max(0, K - S)

    # Using np.isclose for floating point comparison
    if np.isclose(sigma, 0): # Zero volatility edge case
        lambda_tau = tau * lambda_sensitivity
        # Simplified pricing: compare discounted strike with adjusted asset value growth
        # Using risk-free rate 'r' for discounting K as is standard
        if option_type == 'call':
            price = max(0, S * np.exp((r + lambda_tau - r) * T) - K * np.exp(-r*T)) # Simplified form S*exp(lambda*T) - K*exp(-rT) ? No, drift matters.
            price = max(0, S * np.exp(lambda_tau * T) - K * np.exp(-r*T)) # Let's try the previous simpler form again for sigma=0
        else: # put
             price = max(0, K * np.exp(-r*T) - S * np.exp(lambda_tau * T))
        return price

    # Calculate tariff risk premium
    lambda_tau = tau * lambda_sensitivity
    r_adj = r + lambda_tau # Adjusted risk-neutral drift

    # Standard Black-Scholes calculations with adjusted rate
    # Add small epsilon to denominator to avoid division by zero if T is extremely small but not exactly 0
    sigma_sqrt_T = sigma * np.sqrt(T)
    if np.isclose(sigma_sqrt_T, 0):
        st.warning(f"Near-zero sigma*sqrt(T) encountered: sigma={sigma}, T={T}. Returning intrinsic value.")
        return max(0, S - K) if option_type == 'call' else max(0, K - S)

    d1 = (np.log(S / K) + (r_adj + 0.5 * sigma**2) * T) / sigma_sqrt_T
    d2 = d1 - sigma_sqrt_T

    try:
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == 'put':
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            return np.nan # Invalid option type
    except Exception as e:
        st.error(f"Error during BSM calculation: {e}")
        return np.nan # Catch potential math errors

    # Ensure price is not negative
    return max(0, price)


def calculate_vaR_es(final_values, confidence_level=0.95):
    """Calculates Value-at-Risk (VaR) and Expected Shortfall (ES)."""
    if final_values is None or len(final_values) == 0:
        return np.nan, np.nan
    # Ensure final_values is a numpy array for sorting
    final_values = np.asarray(final_values)
    # Remove NaN values if any exist
    final_values = final_values[~np.isnan(final_values)]
    if len(final_values) == 0:
         return np.nan, np.nan

    sorted_values = np.sort(final_values)
    index = int((1 - confidence_level) * len(sorted_values))
    # Ensure index is within bounds
    if index >= len(sorted_values):
        index = len(sorted_values) - 1
    if index < 0:
        index = 0

    var = sorted_values[index]
    # Ensure ES calculation doesn't involve empty slice
    if index > 0:
      es = np.mean(sorted_values[:index])
    else:
      es = sorted_values[0] # If VaR is the lowest value, ES is that value too

    return var, es

# --- Data Fetching ---

@st.cache_data(ttl=600) # Cache data for 10 minutes
def get_stock_price(ticker_symbol):
    """Fetches current stock price ONLY. Returns None on failure."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        # Try multiple ways to get a recent price
        history = ticker.history(period="5d")
        current_price = None
        if not history.empty:
            current_price = history['Close'].iloc[-1]

        # Fallback using info if history failed or price is NaN
        if current_price is None or np.isnan(current_price):
             info = ticker.info
             current_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('ask') or info.get('bid') or info.get('previousClose')

        if current_price is None or np.isnan(current_price):
             st.error(f"Could not reliably fetch current price for {ticker_symbol}.")
             return None

        # Check if the price is serializable (should be float or int)
        # try:
        #     pickle.dumps(current_price)
        # except Exception as e:
        #      st.error(f"Price for {ticker_symbol} is not serializable: {current_price}, Error: {e}")
        #      return None

        return float(current_price) # Ensure it's a float

    except Exception as e:
        st.error(f"Error fetching price data for {ticker_symbol}: {e}")
        return None

# Added caching here - DataFrames are serializable
@st.cache_resource(ttl=600)
def get_option_chain(ticker_symbol, expiry_date):
    """Fetches the option chain for a specific expiry date.
       Requires ticker_symbol string, not the object, to be cache-friendly.
    """
    try:
        ticker_obj = yf.Ticker(ticker_symbol)  # Create object inside function
        return ticker_obj.option_chain(expiry_date)
    except Exception as e:
        st.warning(f"Could not fetch option chain for {ticker_symbol} on {expiry_date}: {e}")
        return None

@st.cache_data(ttl=3600) # Cache historical data longer
def get_historical_data(ticker_symbol, start_date, end_date):
    """Fetches historical stock data."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(start=start_date, end=end_date)
        # Convert timezone-aware index to timezone-naive UTC
        if isinstance(hist.index, pd.DatetimeIndex) and hist.index.tz is not None:
             hist.index = hist.index.tz_convert(None)

        if hist.empty:
            st.warning(f"No historical data found for {ticker_symbol} in the specified range.")
            return None
        return hist # DataFrame should be serializable
    except Exception as e:
        st.error(f"Error fetching historical data for {ticker_symbol}: {e}")
        return None

# --- Streamlit UI ---

def main():
    st.set_page_config(layout="wide", page_title="Tariff Impact Simulator")
    st.title("🌍 Tariff Impact & Derivatives Risk Simulator")
    st.markdown("""
    Analyze the potential impact of trade tariffs on economic indicators and derivatives pricing.
    Fetch near real-time market data, run simulations, perform historical backtests, and explore sensitivity.
    """)

    # --- Sidebar for Core Inputs ---
    st.sidebar.header("Core Parameters")
    ticker_symbol = st.sidebar.text_input("Stock Ticker", value=DEFAULT_TICKER, help="e.g., SPY, AAPL, MSFT").upper()
    tau = st.sidebar.slider("Assumed Tariff Rate (τ)", min_value=0.0, max_value=1.0, value=0.1, step=0.01, format="%.2f",
                            help="Hypothetical tariff level (0% to 100%). Influences the risk premium.")
    lambda_sens = st.sidebar.slider("Tariff Sensitivity (λ factor)", min_value=0.0, max_value=0.5, value=0.1, step=0.01, format="%.2f",
                                   help="Determines how strongly tariffs affect the option pricing risk premium (r_adj = r + τ * λ_factor).")
    r = st.sidebar.slider("Risk-Free Rate (r)", min_value=0.0, max_value=0.2, value=0.05, step=0.005, format="%.3f",
                          help="Annualized risk-free interest rate (e.g., T-bill rate).")
    sigma_opt = st.sidebar.slider("Option Volatility (σ_opt)", min_value=0.01, max_value=1.0, value=0.20, step=0.01, format="%.2f",
                                help="Implied or historical volatility for the specific underlying asset.")

    # --- Fetch Market Data ---
    # Get cached price
    current_S = get_stock_price(ticker_symbol)
    ticker_obj = None # Initialize ticker_obj
    option_expiries = []

    if current_S is not None:
        st.sidebar.metric(f"Current {ticker_symbol} Price", f"${current_S:,.2f}")
        # Price fetched successfully, now create the Ticker object (not cached)
        try:
            ticker_obj = yf.Ticker(ticker_symbol)
            option_expiries = list(ticker_obj.options)
        except Exception as e:
             st.sidebar.warning(f"Could not fetch option expiry dates for {ticker_symbol}: {e}")
             option_expiries = [] # Ensure it's an empty list on failure
    else:
        st.sidebar.warning(f"Could not fetch current price for {ticker_symbol}. Using default $100.00 for calculations.")
        current_S = 100.0 # Use default for calculations if fetch failed

    # --- Main Application Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Live Pricing & Eco Sim",
        "⏳ Historical Backtest",
        "📈 Sensitivity Analysis",
        "📉 Risk Metrics (VaR/ES)",
        "ℹ️ Model Explanations"
    ])

    # Initialize variables that might be needed across tabs if dependencies exist
    K = None
    T_opt = None
    sim_data = None # Ensure sim_data is initialized
    time_axis_econ = None
    T_sim_econ = 1.0 # Default value if not set in tab 1 yet

    # --- Tab 1: Live Pricing & Economic Simulation ---
    with tab1:
        st.header(f"Live Option Pricing for {ticker_symbol}")
        col1a, col1b = st.columns([1, 2]) # Adjusted column width ratio

        with col1a:
            st.subheader("Option Parameters")
            if not option_expiries: # Check if list is empty
                st.warning("No expiry dates available. Enter manually or check ticker.")
                # Provide a default date if none are available
                default_expiry = datetime.today() + timedelta(days=30)
                expiry_select_date = st.date_input("Select Option Expiry Date", value=default_expiry)
                expiry_str = expiry_select_date.strftime('%Y-%m-%d')
            else:
                # Try to find a reasonable default expiry (e.g., ~30-45 days out)
                today = datetime.today()
                future_expiries = [exp for exp in option_expiries if datetime.strptime(exp, '%Y-%m-%d') > today]
                default_expiry_index = 0
                if future_expiries:
                    # Find expiry closest to 30 days from now
                    target_date = today + timedelta(days=30)
                    best_expiry = min(future_expiries, key=lambda d: abs(datetime.strptime(d, '%Y-%m-%d') - target_date))
                    if best_expiry in option_expiries:
                         default_expiry_index = option_expiries.index(best_expiry)

                expiry_str = st.selectbox("Select Option Expiry Date", options=option_expiries, index=default_expiry_index)

            # Fetch option chain for selected expiry (using cached function)
            option_chain_data = None
            if expiry_str:
                 # Pass ticker_symbol (string) to the cached function
                 option_chain_data = get_option_chain(ticker_symbol, expiry_str)

            # Select Strike Price
            available_strikes = []
            default_strike = round(current_S / 5) * 5 # Default guess before fetching chain
            K = default_strike # Initialize K with default

            if option_chain_data is not None and not option_chain_data.calls.empty:
                available_strikes = sorted(option_chain_data.calls['strike'].unique())
                if available_strikes: # Check if strikes were actually found
                    # Suggest a strike near the current price
                    default_strike = min(available_strikes, key=lambda x:abs(x-current_S))
                    # Check if default_strike is valid before finding index
                    if default_strike in available_strikes:
                        default_strike_index = available_strikes.index(default_strike)
                    else:
                        default_strike_index = 0 # Fallback index
                    K = st.selectbox("Select Strike Price (K)", options=available_strikes, index=default_strike_index, key="strike_select")
                else:
                    st.warning("Strikes list is empty in fetched option chain. Enter manually.")
                    K = st.number_input("Strike Price (K)", value=default_strike, step=1.0, key="strike_manual_empty")
            else:
                 # Fallback if no chain data found
                 st.warning("No option chain data found for this expiry. Enter strike manually.")
                 K = st.number_input("Strike Price (K)", value=default_strike, step=1.0, key="strike_manual_nochain")


            # Calculate Time to Maturity
            try:
                expiry_date = datetime.strptime(expiry_str, '%Y-%m-%d')
                T_opt = max(0.00001, (expiry_date - datetime.today()).days / 365.25) # Avoid T=0, use small epsilon
                st.write(f"Time to Maturity (T): {T_opt:.4f} years")
            except ValueError:
                st.error("Invalid date format selected.")
                T_opt = 0.00001 # Assign a small default T

            # Calculate and Display Prices
            st.subheader("Calculated Option Prices")
            # Ensure K and T_opt have valid values before calculation
            if K is not None and T_opt is not None:
                price_call_mod = modified_black_scholes(current_S, K, T_opt, r, sigma_opt, tau, 'call', lambda_sens)
                price_put_mod = modified_black_scholes(current_S, K, T_opt, r, sigma_opt, tau, 'put', lambda_sens)
                price_call_std = modified_black_scholes(current_S, K, T_opt, r, sigma_opt, 0.0, 'call', lambda_sens) # tau=0
                price_put_std = modified_black_scholes(current_S, K, T_opt, r, sigma_opt, 0.0, 'put', lambda_sens) # tau=0

                # Format prices, handle potential NaN
                def format_price(p):
                    return f"${p:.2f}" if p is not None and not np.isnan(p) else "N/A"

                def format_delta(p_mod, p_std):
                     if p_mod is not None and not np.isnan(p_mod) and p_std is not None and not np.isnan(p_std):
                         return f"{p_mod - p_std:+.2f} vs Std BS"
                     return None # Handle cases where calculation failed

                st.metric(f"Modified Call (τ={tau:.2f})", format_price(price_call_mod), format_delta(price_call_mod, price_call_std))
                st.metric(f"Modified Put (τ={tau:.2f})", format_price(price_put_mod), format_delta(price_put_mod, price_put_std))

                # Display Market Prices if available
                st.subheader("Market Comparison (Last Price)")
                if option_chain_data is not None and K is not None:
                    market_call = option_chain_data.calls[option_chain_data.calls['strike'] == K]
                    market_put = option_chain_data.puts[option_chain_data.puts['strike'] == K]
                    market_call_price = "N/A"
                    market_put_price = "N/A"
                    if not market_call.empty:
                        mc = market_call.iloc[0]
                        market_call_price = f"${mc['lastPrice']:.2f} (Bid: {mc['bid']:.2f}, Ask: {mc['ask']:.2f})"
                    if not market_put.empty:
                        mp = market_put.iloc[0]
                        market_put_price = f"${mp['lastPrice']:.2f} (Bid: {mp['bid']:.2f}, Ask: {mp['ask']:.2f})"

                    st.write(f"Market Call: {market_call_price}")
                    st.write(f"Market Put: {market_put_price}")
                else:
                    st.write("Market prices not available (no chain data or strike mismatch).")
            else:
                 st.warning("Strike Price (K) or Time to Maturity (T) not set. Cannot calculate prices.")


        with col1b:
            st.subheader("Conceptual Economic Simulation")
            st.markdown("This simulation shows how a generic economic indicator might evolve under the specified tariff rate. It's used conceptually to justify the tariff risk premium and for VaR/ES calculation.")
            # Economic Sim Params (can be simplified or linked to sidebar)
            T0_econ = st.number_input("Initial Indicator Value", 100.0, key='econ_T0')
            mu_econ = st.slider("Indicator Drift (μ_econ)", -0.1, 0.2, 0.02, step=0.01, key='econ_mu')
            sigma_econ_val = st.slider("Indicator Volatility (σ_econ)", 0.05, 0.5, 0.15, step=0.01, key='econ_sigma')
            T_sim_econ = st.slider("Simulation Horizon (Years)", 0.5, 5.0, 1.0, step=0.5, key='econ_T') # Assign to variable used in Tab 4 title
            paths_econ = st.select_slider("Simulation Paths", options=[100, 250, 500, 1000], value=250, key='econ_paths')
            steps_per_year_econ = 50 # Fixed steps for faster display

            dt_econ = 1.0 / steps_per_year_econ
            total_steps_econ = int(T_sim_econ * steps_per_year_econ)

            # Run simulation (ensure sim_data is updated for other tabs)
            sim_data, time_axis_econ = economic_model_simulation(T0_econ, tau, mu_econ, sigma_econ_val, dt_econ, total_steps_econ, paths_econ)

            # Plot using Plotly
            fig_econ = go.Figure()
            # Plot subset of paths
            num_paths_to_plot = min(paths_econ, 50)
            for i in range(num_paths_to_plot):
                fig_econ.add_trace(go.Scatter(x=time_axis_econ, y=sim_data[i, :], mode='lines', line=dict(width=0.5), opacity=0.3, showlegend=False))
            # Plot mean path
            mean_path = np.mean(sim_data, axis=0)
            fig_econ.add_trace(go.Scatter(x=time_axis_econ, y=mean_path, mode='lines', name=f'Mean Path (τ={tau:.2f})', line=dict(color='red', width=2)))
            # Plot theoretical mean without tariff
            mean_no_tariff = T0_econ * np.exp(mu_econ * time_axis_econ)
            fig_econ.add_trace(go.Scatter(x=time_axis_econ, y=mean_no_tariff, mode='lines', name='Mean Path (τ=0.00)', line=dict(color='black', width=2, dash='dash')))

            fig_econ.update_layout(title=f"Simulated Economic Indicator Paths (τ = {tau:.2f})",
                                   xaxis_title="Time (Years)", yaxis_title="Indicator Value", height=400, margin=dict(l=20, r=20, t=40, b=20)) # Adjust margins
            st.plotly_chart(fig_econ, use_container_width=True)


    # --- Tab 2: Historical Backtest ---
    with tab2:
        st.header(f"Historical Simulation for {ticker_symbol}")
        st.markdown("""
        Simulate how the modified option pricing model would have performed against standard Black-Scholes
        using historical stock data under a *hypothetical* past tariff scenario.
        **Note:** This does *not* use historical option prices (hard to obtain), but compares the *model's output*
        with and without the tariff adjustment applied to the historical stock path.
        """)

        col2a, col2b = st.columns([1, 2]) # Adjusted ratio
        with col2a:
            # Default dates: 1 year period ending today
            default_end_date = datetime.today()
            default_start_date = default_end_date - timedelta(days=365)
            hist_start_date = st.date_input("Start Date", value=default_start_date)
            hist_end_date = st.date_input("End Date", value=default_end_date)
            hist_tau = st.slider("Hypothetical Tariff Rate for Period (τ_hist)", 0.0, 1.0, tau, step=0.01, key='hist_tau') # Use sidebar tau as default
            # Use K from Tab 1 if available, else use default guess based on current_S
            hist_K_default = K if K is not None else round(current_S / 5) * 5
            hist_K = st.number_input("Strike Price for Simulation (K_hist)", value=hist_K_default, key='hist_K', step=1.0)
            hist_T_opt = st.number_input("Option Maturity at Start (Years, T_hist)", value=0.5, min_value=0.05, step=0.05, key='hist_T')
            run_backtest = st.button("Run Historical Simulation")

        # Make sure calculation runs only when button is pressed and dates are valid
        if run_backtest:
             if hist_start_date >= hist_end_date:
                  with col2b:
                       st.error("Error: Start date must be before end date.")
             else:
                hist_data = get_historical_data(ticker_symbol, hist_start_date, hist_end_date)

                if hist_data is not None and not hist_data.empty:
                    hist_data = hist_data.dropna(subset=['Close']) # Ensure no missing prices
                    if not hist_data.empty:
                        # Calculate option prices day-by-day
                        results = []
                        start_date_dt = pd.to_datetime(hist_start_date) # Convert to datetime if needed

                        for date_idx, row in hist_data.iterrows():
                            # Ensure date_idx is timezone-naive for comparison
                            current_date_naive = date_idx.tz_localize(None) if date_idx.tzinfo else date_idx

                            current_hist_S = row['Close']
                            # Time to maturity decreases each day
                            days_passed = (current_date_naive - start_date_dt).days
                            current_hist_T = max(0.00001, hist_T_opt - (days_passed / 365.25)) # Avoid T=0

                            if np.isclose(current_hist_T, 0): continue # Stop if option effectively expires

                            price_call_mod_hist = modified_black_scholes(current_hist_S, hist_K, current_hist_T, r, sigma_opt, hist_tau, 'call', lambda_sens)
                            price_put_mod_hist = modified_black_scholes(current_hist_S, hist_K, current_hist_T, r, sigma_opt, hist_tau, 'put', lambda_sens)
                            price_call_std_hist = modified_black_scholes(current_hist_S, hist_K, current_hist_T, r, sigma_opt, 0.0, 'call', lambda_sens)
                            price_put_std_hist = modified_black_scholes(current_hist_S, hist_K, current_hist_T, r, sigma_opt, 0.0, 'put', lambda_sens)

                            results.append({
                                "Date": date_idx, # Keep original index
                                "Stock Price": current_hist_S,
                                "Call Mod": price_call_mod_hist,
                                "Put Mod": price_put_mod_hist,
                                "Call Std": price_call_std_hist,
                                "Put Std": price_put_std_hist,
                                "Call Diff": price_call_mod_hist - price_call_std_hist if not (np.isnan(price_call_mod_hist) or np.isnan(price_call_std_hist)) else 0,
                                "Put Diff": price_put_mod_hist - price_put_std_hist if not (np.isnan(price_put_mod_hist) or np.isnan(price_put_std_hist)) else 0,
                            })

                        if not results:
                             with col2b:
                                st.warning("No valid results generated. Check date range and parameters (e.g., option might expire too quickly).")

                        else:
                            results_df = pd.DataFrame(results).set_index("Date")

                            with col2b:
                                st.subheader("Backtest Results")
                                # Plot Stock Price
                                fig_hist_stock = go.Figure()
                                fig_hist_stock.add_trace(go.Scatter(x=results_df.index, y=results_df['Stock Price'], mode='lines', name='Stock Price'))
                                fig_hist_stock.update_layout(title=f"{ticker_symbol} Historical Price", yaxis_title="Price ($)", height=250, margin=dict(t=30, b=10, l=10, r=10))
                                st.plotly_chart(fig_hist_stock, use_container_width=True)

                                # Plot Option Prices (Calls)
                                fig_hist_call = go.Figure()
                                fig_hist_call.add_trace(go.Scatter(x=results_df.index, y=results_df['Call Mod'], mode='lines', name=f'Modified Call (τ={hist_tau:.2f})'))
                                fig_hist_call.add_trace(go.Scatter(x=results_df.index, y=results_df['Call Std'], mode='lines', name='Standard Call (τ=0.00)', line=dict(dash='dash')))
                                fig_hist_call.update_layout(title=f"Simulated Call Option Price (K={hist_K})", yaxis_title="Option Price ($)", height=300, margin=dict(t=30, b=10, l=10, r=10))
                                st.plotly_chart(fig_hist_call, use_container_width=True)

                                # Plot Option Price Difference (Calls)
                                fig_hist_call_diff = go.Figure()
                                fig_hist_call_diff.add_trace(go.Scatter(x=results_df.index, y=results_df['Call Diff'], mode='lines', name='Call Price Difference (Mod - Std)', line=dict(color='green')))
                                fig_hist_call_diff.update_layout(title="Call Price Difference due to Tariff", yaxis_title="Price Difference ($)", height=200, margin=dict(t=30, b=10, l=10, r=10))
                                st.plotly_chart(fig_hist_call_diff, use_container_width=True)

                                # (Optional: Add Put plots similarly)

                    else:
                         with col2b:
                            st.warning("Historical data was fetched but contained only NaN values for 'Close' price after cleaning.")
                else:
                     with col2b:
                        st.error("Failed to fetch or process historical data for the selected range.")


    # --- Tab 3: Sensitivity Analysis ---
    with tab3:
        st.header("Sensitivity Analysis")
        st.markdown("Explore how the modified option price changes with different parameters.")

        # Use K and T_opt from Tab 1 if available, otherwise provide defaults
        base_K = K if K is not None else round(current_S / 5) * 5
        base_T = T_opt if T_opt is not None else 1.0

        # Check if base_K and base_T are valid before proceeding
        if base_K <= 0 or base_T <= 0:
             st.warning("Strike Price (K) or Time to Maturity (T) from Tab 1 is invalid or not set. Using defaults for sensitivity analysis.")
             base_K = round(current_S / 5) * 5
             base_T = 1.0

        sens_param = st.selectbox("Select Parameter to Vary:", ["Tariff Rate (τ)", "Option Volatility (σ_opt)", "Time to Maturity (T)"])
        sens_range_mult = st.slider("Analysis Range (+/- % of current value)", 10, 100, 50, step=5, key="sens_range") / 100.0
        sens_steps = 21 # Number of points to calculate

        call_prices_sens = []
        put_prices_sens = []
        param_values = []

        # Use current values from sidebar/live data as baseline
        base_S = current_S
        base_r = r
        base_sigma = sigma_opt
        base_tau = tau
        base_lambda = lambda_sens

        # Generate parameter ranges carefully
        if sens_param == "Tariff Rate (τ)":
            center_val = base_tau
            half_range = center_val * sens_range_mult if center_val > 0 else 0.1 * sens_range_mult # Avoid 0 range if base is 0
            start_val = max(0.0, center_val - half_range)
            end_val = min(1.0, center_val + half_range) # Cap tariff at 1.0
            param_values = np.linspace(start_val, end_val, sens_steps)
            x_axis_title = "Tariff Rate (τ)"
            for p_val in param_values:
                call_prices_sens.append(modified_black_scholes(base_S, base_K, base_T, base_r, base_sigma, p_val, 'call', base_lambda))
                put_prices_sens.append(modified_black_scholes(base_S, base_K, base_T, base_r, base_sigma, p_val, 'put', base_lambda))

        elif sens_param == "Option Volatility (σ_opt)":
            center_val = base_sigma
            half_range = center_val * sens_range_mult
            start_val = max(0.01, center_val - half_range) # Volatility must be > 0
            end_val = center_val + half_range
            param_values = np.linspace(start_val, end_val, sens_steps)
            x_axis_title = "Option Volatility (σ_opt)"
            for p_val in param_values:
                call_prices_sens.append(modified_black_scholes(base_S, base_K, base_T, base_r, p_val, base_tau, 'call', base_lambda))
                put_prices_sens.append(modified_black_scholes(base_S, base_K, base_T, base_r, p_val, base_tau, 'put', base_lambda))

        elif sens_param == "Time to Maturity (T)":
            center_val = base_T
            half_range = center_val * sens_range_mult
            start_val = max(0.001, center_val - half_range) # Time must be > 0
            end_val = center_val + half_range
            param_values = np.linspace(start_val, end_val, sens_steps)
            x_axis_title = "Time to Maturity (T, years)"
            for p_val in param_values:
                call_prices_sens.append(modified_black_scholes(base_S, base_K, p_val, base_r, base_sigma, base_tau, 'call', base_lambda))
                put_prices_sens.append(modified_black_scholes(base_S, base_K, p_val, base_r, base_sigma, base_tau, 'put', base_lambda))

        # Plot Sensitivity using Plotly
        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(x=param_values, y=call_prices_sens, mode='lines+markers', name='Call Price'))
        fig_sens.add_trace(go.Scatter(x=param_values, y=put_prices_sens, mode='lines+markers', name='Put Price'))
        fig_sens.update_layout(
             title=f"Option Price Sensitivity to {sens_param}<br>(S={base_S:.2f}, K={base_K}, T={base_T:.2f}, r={base_r:.3f}, σ={base_sigma:.2f}, τ={base_tau:.2f})", # Add baseline params to title
             xaxis_title=x_axis_title,
             yaxis_title="Option Price ($)",
             height=450,
             margin=dict(t=60, b=40, l=40, r=20) # Adjust margin for longer title
        )
        st.plotly_chart(fig_sens, use_container_width=True)


    # --- Tab 4: Risk Metrics (VaR/ES) ---
    with tab4:
        st.header("Risk Metrics from Economic Simulation")
        st.markdown("""
        Value-at-Risk (VaR) and Expected Shortfall (ES) calculated from the distribution of the
        final simulated values of the *conceptual economic indicator* (from the first tab).
        These metrics give an idea of the potential downside risk in the simulated indicator under the tariff scenario.
        """)

        # Check if sim_data exists and is valid before proceeding
        if sim_data is not None and isinstance(sim_data, np.ndarray) and sim_data.size > 0 :
            final_values = sim_data[:, -1] # Get the last column (final values)
            confidence_level = st.slider("Confidence Level for VaR/ES", 0.90, 0.99, 0.95, step=0.01, key="risk_confidence")

            var, es = calculate_vaR_es(final_values, confidence_level)

            # Format metrics, handle potential NaN
            def format_metric(v):
                return f"{v:.2f}" if v is not None and not np.isnan(v) else "N/A"

            st.metric(f"Value-at-Risk (VaR) at {confidence_level:.0%}", format_metric(var),
                      help=f"The value at the {(1-confidence_level)*100:.0f}th percentile. There is a {(1-confidence_level)*100:.0f}% chance the indicator will be below this value.")
            st.metric(f"Expected Shortfall (ES) at {confidence_level:.0%}", format_metric(es),
                      help=f"The average value of the indicator in the worst {(1-confidence_level)*100:.0f}% of cases.")

            # Plot histogram with VaR/ES lines
            fig_hist_risk = go.Figure()
            fig_hist_risk.add_trace(go.Histogram(x=final_values, name='Distribution', nbinsx=50, histnorm='probability density'))

            # Add lines only if VaR/ES are valid numbers
            if var is not None and not np.isnan(var):
                fig_hist_risk.add_vline(x=var, line_dash="dash", line_color="red", annotation_text=f"VaR ({confidence_level:.0%}) = {var:.2f}", annotation_position="top left")
            if es is not None and not np.isnan(es):
                 fig_hist_risk.add_vline(x=es, line_dash="dot", line_color="orange", annotation_text=f"ES ({confidence_level:.0%}) = {es:.2f}", annotation_position="bottom left")

            # Use T_sim_econ which should be set in Tab 1
            fig_hist_risk.update_layout(title=f"Distribution of Final Indicator Value (T={T_sim_econ:.1f} Years)",
                                        xaxis_title="Final Indicator Value", yaxis_title="Density", height=450)
            st.plotly_chart(fig_hist_risk, use_container_width=True)

        else:
            st.warning("Economic simulation data not available. Please run the simulation on the 'Live Pricing & Eco Sim' tab first.")


    # --- Tab 5: Model Explanations ---
    with tab5:
        st.header("Model Explanations & Methodology")

        st.subheader("Core Concepts")
        st.markdown("""
        - **Tariff Impact:** Tariffs are assumed to create additional risk and potentially dampen economic activity (or specific asset growth).
        - **Risk Premium:** This extra risk is modeled as a premium `λ(τ)` added to the risk-free rate `r` in the option pricing formula's drift term (`r_adj = r + λ(τ)`). This increases call prices and decreases put prices, reflecting the higher expected growth needed to compensate for risk (in the risk-neutral world).
        - **Lambda Sensitivity:** The `λ factor` parameter controls how strongly the tariff rate `τ` translates into the risk premium `λ(τ) = τ * λ_factor`.
        """)

        st.subheader("Economic Simulation")
        st.markdown(r"""
        - Uses Geometric Brownian Motion (GBM) to model a *conceptual* indicator:
          $$ dX_t = \mu_{adj} X_t dt + \sigma_{econ} X_t dW_t $$
        - The tariff `τ` adjusts the drift: $\mu_{adj} = \mu_{econ} (1 - \tau)$.
        - **Purpose:** Primarily visualizes the *potential effect* of tariffs on growth/volatility trends and provides the basis for calculating VaR/ES on this conceptual indicator. It does *not* directly simulate the specific stock price `S`.
        """)

        st.subheader("Modified Black-Scholes")
        st.markdown(r"""
        - Standard Black-Scholes adjusted for the tariff risk premium `λ(τ)`:
          $$ d_1 = \frac{\ln(S/K) + (r + \lambda(\tau) + \frac{1}{2}\sigma_{opt}^2)T}{\sigma_{opt}\sqrt{T}} $$
          $$ d_2 = d_1 - \sigma_{opt}\sqrt{T} $$
        - Call Price: $C = S N(d_1) - K e^{-rT} N(d_2)$
        - Put Price: $P = K e^{-rT} N(-d_2) - S N(-d_1)$
        - **Note:** The discount factor $e^{-rT}$ uses the original risk-free rate `r`. The premium $\lambda(\tau)$ affects the risk-neutral *drift* of the underlying asset `S`.
        """)

        st.subheader("Data Fetching (`yfinance`)")
        st.markdown("""
        - Uses the `yfinance` library to fetch near real-time stock prices and option chain data (expiry dates, strikes, bid/ask/last prices).
        - Fetches historical stock closing prices for the backtesting module.
        - **Limitations:** Data is typically delayed (15-20 min). Historical *option* data is not readily available via this free API. Volatility (`σ_opt`) is currently a manual input, though implied volatility could potentially be derived from market option prices in a more advanced version. Data fetching may occasionally fail due to API issues or network problems.
        """)

        st.subheader("Historical Backtest")
        st.markdown("""
        - Fetches historical stock prices for the chosen ticker and period.
        - Calculates the *Modified B-S* and *Standard B-S* option prices for each day in the period, using the historical stock price `S` and assuming a constant *hypothetical* tariff `τ_hist` and volatility `σ_opt`.
        - Time to maturity `T` decreases daily.
        - **Comparison:** Shows the *difference* in theoretical option prices generated by the model (with vs. without tariff) when applied to past stock movements. It does *not* compare against actual traded historical option prices.
        """)

        st.subheader("Sensitivity Analysis")
        st.markdown("""
        - Calculates how the Modified B-S call and put prices change as one input parameter (Tariff Rate, Volatility, or Time to Maturity) is varied across a specified range, keeping all other parameters constant.
        - Helps understand the model's sensitivity to its inputs.
        """)

        st.subheader("Risk Metrics (VaR/ES)")
        st.markdown("""
        - **Value-at-Risk (VaR):** Estimates the maximum potential loss (or lowest value for the indicator) over a given time horizon at a specific confidence level (e.g., 95%). Calculated as the percentile of the simulated final indicator values.
        - **Expected Shortfall (ES):** Also known as Conditional VaR (CVaR). Estimates the average loss (or average value) *given* that the loss exceeds the VaR threshold. Calculated as the mean of the simulated final values that fall below the VaR percentile.
        - These metrics are based on the *conceptual economic simulation*, not directly on the option prices or the specific stock.
        """)

# --- Run the App ---
if __name__ == '__main__':
    main()

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm, poisson
from scipy.optimize import minimize # Potentially for future calibration
import yfinance as yf
from datetime import datetime, timedelta, date
import pandas_datareader.data as pdr
from arch import arch_model # For GARCH
import warnings
import math

# --- Configuration & Constants ---
DEFAULT_TICKER = "SPY" # Default ticker
FRED_API_KEY = '578cb05ad0638ffd9d413e6e1a3321b9' # Add your FRED API key here if needed, often works without for public series
RISK_FREE_RATE_SERIES = 'TB3MS' # 3-Month Treasury Bill Secondary Market Rate
MIN_HIST_DATA_POINTS = 100 # Minimum data points needed for GARCH fit
MERTON_SUM_N = 20 # Number of terms to sum for Merton model approximation

# Suppress warnings from arch model fitting for cleaner UI
warnings.filterwarnings("ignore", category=UserWarning) # GARCH convergence warnings
warnings.filterwarnings("ignore", category=FutureWarning) # Pandas future warnings

# --- Mathematical Models ---

# (economic_model_simulation remains the same as v2.1)
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


def black_scholes_merton(S, K, T, r, sigma, option_type='call'):
    """ Standard Black-Scholes formula (used as base for Modified/Merton) """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0: return np.nan
    if np.isclose(sigma, 0) or np.isclose(T, 0):
        return max(0, S - K) if option_type == 'call' else max(0, K - S)

    sigma_sqrt_T = sigma * np.sqrt(T)
    if np.isclose(sigma_sqrt_T, 0): return max(0, S - K) if option_type == 'call' else max(0, K - S)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / sigma_sqrt_T
    d2 = d1 - sigma_sqrt_T

    try:
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == 'put':
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else: return np.nan
    except Exception: return np.nan
    return max(0, price)

def modified_black_scholes_with_tariff(S, K, T, r, sigma, tau, option_type='call', lambda_sensitivity=0.1):
    """ Modified Black-Scholes incorporating tariff risk premium lambda(tau)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0: return np.nan

    lambda_tau = tau * lambda_sensitivity
    r_adj = r + lambda_tau # Adjusted risk-neutral drift for tariff

    # Use the standard BSM function but with the adjusted risk-free rate for the drift component
    # We need to be careful here. The standard BSM uses 'r' for both drift and discounting.
    # Our modification adjusts the drift part. Let's adjust the BSM formula directly.

    if np.isclose(sigma, 0) or np.isclose(T, 0):
        # For sigma=0 or T=0, the tariff premium affects the expected forward price
        if option_type == 'call': price = max(0, S * np.exp(lambda_tau * T) - K * np.exp(-r * T))
        else: price = max(0, K * np.exp(-r * T) - S * np.exp(lambda_tau * T))
        return price

    sigma_sqrt_T = sigma * np.sqrt(T)
    if np.isclose(sigma_sqrt_T, 0): return max(0, S - K) if option_type == 'call' else max(0, K - S)

    # Use r_adj in d1/d2 calculation, but original 'r' for discounting K
    d1 = (np.log(S / K) + (r_adj + 0.5 * sigma**2) * T) / sigma_sqrt_T
    d2 = d1 - sigma_sqrt_T

    try:
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == 'put':
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else: return np.nan
    except Exception: return np.nan
    return max(0, price)


def merton_jump_diffusion(S, K, T, r, sigma, tau, jump_intensity, jump_mean, jump_vol, option_type='call', lambda_sensitivity=0.1, n_terms=MERTON_SUM_N):
    """ Merton Jump Diffusion option pricing model including tariff premium. """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0 or jump_intensity < 0 or jump_vol < 0: return np.nan

    lambda_tau = tau * lambda_sensitivity
    r_base = r # Original risk-free rate for discounting

    # Adjust drift for jumps and tariff premium
    # Merton's adjustment to risk-neutral drift: lambda_jump * (exp(mu_jump + 0.5*delta^2) - 1)
    # We also add our tariff premium lambda_tau
    kappa = np.exp(jump_mean + 0.5 * jump_vol**2) - 1
    r_jump_adj = jump_intensity * kappa # Risk-neutral drift adjustment due to jumps
    r_eff = r_base + lambda_tau - r_jump_adj # Effective drift combining r, tariff premium, and jump compensation

    total_price = 0.0
    lambda_prime = jump_intensity * np.exp(jump_mean + 0.5 * jump_vol**2) # Adjusted intensity used in Poisson probability

    for n in range(n_terms + 1):
        poisson_prob = (np.exp(-lambda_prime * T) * (lambda_prime * T)**n) / np.math.factorial(n)

        # Adjust parameters for the BSM calculation within the sum
        rn = r_eff + (n * (jump_mean + 0.5 * jump_vol**2)) / T - 0.5 * (n * jump_vol**2) / T # This seems complex, let's re-derive Merton's params
        # Simpler formulation from Hull/Wikipedia:
        r_n = r_base - jump_intensity * kappa + (n * np.log(1 + kappa)) / T + lambda_tau # Include tariff premium here
        sigma_n_sq = sigma**2 + (n * jump_vol**2) / T
        sigma_n = np.sqrt(sigma_n_sq) if sigma_n_sq >= 0 else 0

        # Calculate BSM price for this jump scenario 'n'
        # Use r_n for drift component, r_base for discounting
        # We need a BSM function that allows separate drift and discount rates, or adjust formula here.
        # Let's adapt the formula directly:

        if np.isclose(sigma_n, 0) or np.isclose(T, 0):
            price_n = max(0, S * np.exp((r_n - r_base)*T) - K * np.exp(-r_base * T)) if option_type == 'call' else max(0, K * np.exp(-r_base * T) - S * np.exp((r_n-r_base)*T))
        else:
             sigma_n_sqrt_T = sigma_n * np.sqrt(T)
             if np.isclose(sigma_n_sqrt_T, 0):
                  price_n = max(0, S * np.exp((r_n - r_base)*T) - K * np.exp(-r_base * T)) if option_type == 'call' else max(0, K * np.exp(-r_base * T) - S * np.exp((r_n-r_base)*T))
             else:
                d1_n = (np.log(S / K) + (r_n + 0.5 * sigma_n**2) * T) / sigma_n_sqrt_T
                d2_n = d1_n - sigma_n_sqrt_T
                try:
                    if option_type == 'call':
                        price_n = S * norm.cdf(d1_n) - K * np.exp(-r_base * T) * norm.cdf(d2_n)
                    elif option_type == 'put':
                        price_n = K * np.exp(-r_base * T) * norm.cdf(-d2_n) - S * norm.cdf(-d1_n)
                    else: price_n = 0
                except Exception: price_n = 0

        total_price += poisson_prob * price_n

    return max(0, total_price)


# (calculate_vaR_es remains the same as v2.1)
def calculate_vaR_es(final_values, confidence_level=0.95):
    """Calculates Value-at-Risk (VaR) and Expected Shortfall (ES)."""
    if final_values is None or len(final_values) == 0:
        return np.nan, np.nan
    final_values = np.asarray(final_values)
    final_values = final_values[~np.isnan(final_values)]
    if len(final_values) == 0: return np.nan, np.nan

    sorted_values = np.sort(final_values)
    index = int((1 - confidence_level) * len(sorted_values))
    if index >= len(sorted_values): index = len(sorted_values) - 1
    if index < 0: index = 0

    var = sorted_values[index]
    if index > 0: es = np.mean(sorted_values[:index])
    else: es = sorted_values[0]
    return var, es

# --- Data Fetching & Volatility Modeling ---

# (get_stock_price remains the same as v2.1)
@st.cache_data(ttl=600)
def get_stock_price(ticker_symbol):
    """Fetches current stock price ONLY. Returns None on failure."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        history = ticker.history(period="5d")
        current_price = None
        if not history.empty: current_price = history['Close'].iloc[-1]
        if current_price is None or np.isnan(current_price):
             info = ticker.info
             current_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('ask') or info.get('bid') or info.get('previousClose')
        if current_price is None or np.isnan(current_price):
             st.error(f"Could not reliably fetch current price for {ticker_symbol}.")
             return None
        return float(current_price)
    except Exception as e:
        st.error(f"Error fetching price data for {ticker_symbol}: {e}")
        return None

# (get_option_chain remains the same as v2.1, but using cache_resource might be safer for complex objects)
# Let's try cache_resource as OptionChain object might be complex
@st.cache_resource(ttl=600)
def get_option_chain(_ticker_symbol, expiry_date): # Use underscore to indicate arg used for caching key
    """Fetches the option chain for a specific expiry date."""
    try:
        ticker_obj = yf.Ticker(_ticker_symbol)
        return ticker_obj.option_chain(expiry_date)
    except Exception as e:
        # st.warning(f"Could not fetch option chain for {_ticker_symbol} on {expiry_date}: {e}") # Can be noisy
        print(f"Cache miss or error fetching option chain for {_ticker_symbol} on {expiry_date}: {e}")
        return None

# (get_historical_data remains the same as v2.1)
@st.cache_data(ttl=3600)
def get_historical_data(ticker_symbol, start_date, end_date):
    """Fetches historical stock data."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(start=start_date, end=end_date)
        if isinstance(hist.index, pd.DatetimeIndex) and hist.index.tz is not None:
             hist.index = hist.index.tz_convert(None)
        if hist.empty:
            st.warning(f"No historical data found for {ticker_symbol} in the specified range.")
            return None
        # Ensure data is sorted chronologically
        hist = hist.sort_index()
        return hist
    except Exception as e:
        st.error(f"Error fetching historical data for {ticker_symbol}: {e}")
        return None

@st.cache_data(ttl=86400) # Cache FRED data daily
def get_fred_rate(series_id=RISK_FREE_RATE_SERIES, api_key=FRED_API_KEY):
    """Fetches the latest risk-free rate from FRED."""
    try:
        end_date = date.today()
        # Fetch data for the last year to ensure we get the latest available point
        start_date = end_date - timedelta(days=365)
        fred_data = pdr.DataReader(series_id, 'fred', start_date, end_date, api_key=api_key)
        if not fred_data.empty:
            latest_rate_percent = fred_data.iloc[-1][series_id]
            return float(latest_rate_percent) / 100.0 # Convert percentage to decimal
        else:
            st.warning(f"Could not fetch data for FRED series {series_id}.")
            return None
    except Exception as e:
        st.error(f"Error fetching FRED data ({series_id}): {e}")
        return None

@st.cache_data(ttl=3600) # Cache GARCH results for an hour
def fit_garch_and_forecast(ticker_symbol, hist_start_date, hist_end_date):
    """Fits GARCH(1,1) model and forecasts next-day volatility."""
    hist_data = get_historical_data(ticker_symbol, hist_start_date, hist_end_date)
    if hist_data is None or hist_data.empty or 'Close' not in hist_data.columns:
        return None, "Insufficient historical data."

    hist_data = hist_data.dropna(subset=['Close'])
    if len(hist_data) < MIN_HIST_DATA_POINTS:
         return None, f"Insufficient data points ({len(hist_data)} < {MIN_HIST_DATA_POINTS}) for GARCH fit."

    # Calculate log returns, multiply by 100 for GARCH stability
    log_returns = 100 * np.log(hist_data['Close'] / hist_data['Close'].shift(1)).dropna()

    if log_returns.empty:
        return None, "Could not calculate returns (maybe only one data point?)."

    try:
        # Fit GARCH(1,1) model
        model = arch_model(log_returns, vol='Garch', p=1, q=1, rescale=False) # Rescale=False as we scaled returns
        results = model.fit(disp='off') # Turn off verbose fitting output

        # Forecast next day's variance
        forecast = results.forecast(horizon=1, reindex=False)
        next_day_variance = forecast.variance.iloc[0, 0]

        # Annualize the volatility forecast (sqrt(variance) * sqrt(252))
        # Remember returns were scaled by 100, so variance is 10000x larger
        annualized_vol = np.sqrt(next_day_variance / 10000) * np.sqrt(252)

        return annualized_vol, f"GARCH(1,1) fit successful. Last Vol: {np.sqrt(results.conditional_volatility[-1]**2 / 10000)*np.sqrt(252):.3f}"

    except Exception as e:
        return None, f"GARCH model fitting error: {e}"


# --- Streamlit UI ---

def main():
    st.set_page_config(layout="wide", page_title="Advanced Tariff Impact Simulator")
    st.title("🌍 Tariff Impact & Derivatives Simulator")
    st.markdown("""
    Analyze tariff impacts using advanced models (Modified Black-Scholes, Merton Jump Diffusion),
    GARCH volatility forecasting, real-time interest rates, backtesting, and stress testing.
    **Disclaimer:** This tool is for educational and analytical purposes only, not financial advice. Market data may be delayed. Backtesting limitations apply.
    """)

    # --- Sidebar ---
    st.sidebar.header("⚙️ Core Parameters")

    with st.sidebar.expander("Market & Model Selection", expanded=True):
        ticker_symbol = st.text_input("Stock Ticker", value=DEFAULT_TICKER, help="e.g., SPY, AAPL, MSFT").upper()
        pricing_model = st.selectbox("Select Pricing Model", ["Modified Black-Scholes", "Merton Jump Diffusion"])

    with st.sidebar.expander("Tariff & Risk Premium"):
        tau = st.slider("Assumed Tariff Rate (τ)", 0.0, 1.0, 0.1, 0.01, format="%.2f", help="Hypothetical tariff level (0% to 100%).")
        lambda_sens = st.slider("Tariff Sensitivity (λ factor)", 0.0, 0.5, 0.1, 0.01, format="%.2f", help="Strength of tariff effect on risk premium (r_adj = r + τ * λ_factor).")

    with st.sidebar.expander("Interest Rate (r)"):
         r_source = st.radio("Risk-Free Rate Source", ["Manual Input", "Fetch FRED (TB3MS)"], index=1)
         r_manual = st.slider("Manual Risk-Free Rate (r)", 0.0, 0.2, 0.05, 0.005, format="%.3f", help="Annualized rate if Manual Source selected.")
         r = r_manual # Default to manual
         if r_source == "Fetch FRED (TB3MS)":
              fetched_r = get_fred_rate()
              if fetched_r is not None:
                   r = fetched_r
                   st.success(f"Using FRED {RISK_FREE_RATE_SERIES}: {r:.3%}")
              else:
                   st.warning(f"Failed to fetch FRED rate. Using manual value: {r_manual:.3%}")
                   r = r_manual
         else:
              st.info(f"Using manual rate: {r:.3%}")


    with st.sidebar.expander("Volatility (σ)"):
        vol_source = st.radio("Volatility Source", ["Manual Input", "GARCH(1,1) Forecast"], index=0, key="vol_source")
        sigma_opt_manual = st.slider("Manual Option Volatility (σ_opt)", 0.01, 1.0, 0.20, 0.01, format="%.2f", help="Annualized volatility if Manual Source selected.")
        sigma_opt = sigma_opt_manual # Default

        if vol_source == "GARCH(1,1) Forecast":
            garch_hist_days = st.number_input("Days for GARCH Fit", 252, 100, 1000, 50, help=f"Number of past trading days to use (min {MIN_HIST_DATA_POINTS}).")
            garch_end_date = date.today()
            garch_start_date = garch_end_date - timedelta(days=int(garch_hist_days * 1.5)) # Fetch slightly more data initially

            garch_vol, garch_status = fit_garch_and_forecast(ticker_symbol, garch_start_date, garch_end_date)
            if garch_vol is not None:
                sigma_opt = garch_vol
                st.success(f"Using GARCH Forecast: {sigma_opt:.3f}")
                st.caption(f"Status: {garch_status}")
            else:
                st.warning(f"GARCH forecast failed: {garch_status}. Using manual value: {sigma_opt_manual:.3f}")
                sigma_opt = sigma_opt_manual
        else:
             st.info(f"Using manual volatility: {sigma_opt:.3f}")


    # Merton Jump Diffusion Parameters (only show if Merton model selected)
    jump_params = {}
    if pricing_model == "Merton Jump Diffusion":
        with st.sidebar.expander("Merton Jump Parameters"):
            jump_params['jump_intensity'] = st.slider("Jump Intensity (λ_jump)", 0.0, 5.0, 0.5, 0.1, format="%.1f", help="Average number of jumps per year.")
            jump_params['jump_mean'] = st.slider("Mean Jump Size (μ_jump)", -0.5, 0.5, 0.0, 0.01, format="%.2f", help="Average log-return size of a jump.")
            jump_params['jump_vol'] = st.slider("Jump Volatility (σ_jump)", 0.0, 1.0, 0.2, 0.01, format="%.2f", help="Volatility of the jump size itself.")

    with st.sidebar.expander("Backtest Transaction Costs"):
         st.caption("Simple simulation for backtest results:")
         commission = st.number_input("Commission per Option Contract ($)", 0.0, 10.0, 0.65, 0.05)
         slippage_pct = st.number_input("Slippage per Trade (%)", 0.0, 1.0, 0.05, 0.01) / 100.0


    # --- Fetch Market Data ---
    current_S = get_stock_price(ticker_symbol)
    option_expiries = []
    if current_S is not None:
        st.sidebar.success(f"Current {ticker_symbol} Price: ${current_S:,.2f}")
        try:
            # Fetch expiries using the Ticker object (not cached, needs price success)
            ticker_obj_exp = yf.Ticker(ticker_symbol)
            option_expiries = list(ticker_obj_exp.options)
        except Exception: option_expiries = []
    else:
        st.sidebar.error(f"Failed to fetch {ticker_symbol} price. Using default $100.00.")
        current_S = 100.0

    # --- Tabs ---
    tab_keys = ["live", "backtest", "sensitivity", "stress", "risk", "explain"]
    tab_names = ["📊 Live Pricing", "⏳ Backtest", "📈 Sensitivity", "💥 Stress Test", "📉 Risk Metrics", "ℹ️ Explain"]
    tabs = st.tabs(tab_names)

    # Initialize variables
    K, T_opt, sim_data, time_axis_econ, T_sim_econ = None, None, None, None, 1.0

    # --- Tab 1: Live Pricing ---
    with tabs[0]: # Live Pricing
        st.header(f"Live Option Pricing ({pricing_model})")
        col1a, col1b = st.columns([1, 2])

        with col1a:
            st.subheader("Option Parameters")
            # Expiry Selection (same logic as v2.1)
            if not option_expiries:
                st.warning("No expiry dates available.")
                default_expiry = datetime.today() + timedelta(days=30)
                expiry_select_date = st.date_input("Select Option Expiry Date", value=default_expiry, key=f"{tab_keys[0]}_exp_date")
                expiry_str = expiry_select_date.strftime('%Y-%m-%d')
            else:
                today = datetime.today()
                future_expiries = [exp for exp in option_expiries if datetime.strptime(exp, '%Y-%m-%d') > today]
                default_expiry_index = 0
                if future_expiries:
                    target_date = today + timedelta(days=30)
                    best_expiry = min(future_expiries, key=lambda d: abs(datetime.strptime(d, '%Y-%m-%d') - target_date))
                    if best_expiry in option_expiries: default_expiry_index = option_expiries.index(best_expiry)
                expiry_str = st.selectbox("Select Option Expiry Date", options=option_expiries, index=default_expiry_index, key=f"{tab_keys[0]}_exp_sel")

            # Fetch option chain (using cached function)
            option_chain_data = None
            if expiry_str:
                 option_chain_data = get_option_chain(ticker_symbol, expiry_str) # Pass symbol string

            # Strike Selection (same logic as v2.1)
            available_strikes = []
            default_strike = round(current_S / 5) * 5
            K = default_strike # Init K

            if option_chain_data is not None and not option_chain_data.calls.empty:
                available_strikes = sorted(option_chain_data.calls['strike'].unique())
                if available_strikes:
                    default_strike = min(available_strikes, key=lambda x:abs(x-current_S))
                    if default_strike not in available_strikes: default_strike_index = 0
                    else: default_strike_index = available_strikes.index(default_strike)
                    K = st.selectbox("Select Strike Price (K)", options=available_strikes, index=default_strike_index, key=f"{tab_keys[0]}_strike_sel")
                else:
                    K = st.number_input("Strike Price (K)", value=default_strike, step=1.0, key=f"{tab_keys[0]}_strike_manual_empty")
            else:
                 K = st.number_input("Strike Price (K)", value=default_strike, step=1.0, key=f"{tab_keys[0]}_strike_manual_nochain")

            # Time to Maturity (same logic as v2.1)
            try:
                expiry_date = datetime.strptime(expiry_str, '%Y-%m-%d')
                T_opt = max(0.00001, (expiry_date - datetime.today()).days / 365.25)
                st.write(f"Time to Maturity (T): {T_opt:.4f} years")
            except ValueError: T_opt = 0.00001

            # Calculate and Display Prices based on selected model
            st.subheader("Calculated Option Prices")
            price_call_mod, price_put_mod, price_call_std, price_put_std = np.nan, np.nan, np.nan, np.nan # Init

            if K is not None and T_opt is not None and current_S is not None:
                common_args = {'S': current_S, 'K': K, 'T': T_opt, 'r': r, 'sigma': sigma_opt}
                if pricing_model == "Modified Black-Scholes":
                    price_call_mod = modified_black_scholes_with_tariff(**common_args, tau=tau, option_type='call', lambda_sensitivity=lambda_sens)
                    price_put_mod = modified_black_scholes_with_tariff(**common_args, tau=tau, option_type='put', lambda_sensitivity=lambda_sens)
                    price_call_std = modified_black_scholes_with_tariff(**common_args, tau=0.0, option_type='call', lambda_sensitivity=lambda_sens) # Standard = ModBS with tau=0
                    price_put_std = modified_black_scholes_with_tariff(**common_args, tau=0.0, option_type='put', lambda_sensitivity=lambda_sens)
                elif pricing_model == "Merton Jump Diffusion":
                    merton_args = {**common_args, **jump_params}
                    price_call_mod = merton_jump_diffusion(**merton_args, tau=tau, option_type='call', lambda_sensitivity=lambda_sens)
                    price_put_mod = merton_jump_diffusion(**merton_args, tau=tau, option_type='put', lambda_sensitivity=lambda_sens)
                    price_call_std = merton_jump_diffusion(**merton_args, tau=0.0, option_type='call', lambda_sensitivity=lambda_sens) # Standard = Merton with tau=0
                    price_put_std = merton_jump_diffusion(**merton_args, tau=0.0, option_type='put', lambda_sensitivity=lambda_sens)

                # Formatting functions (same as v2.1)
                def format_price(p): return f"${p:.2f}" if p is not None and not np.isnan(p) else "N/A"
                def format_delta(p_mod, p_std):
                     if p_mod is not None and not np.isnan(p_mod) and p_std is not None and not np.isnan(p_std): return f"{p_mod - p_std:+.2f} vs Std ({pricing_model.split()[0]})"
                     return None

                st.metric(f"Model Call (τ={tau:.2f})", format_price(price_call_mod), format_delta(price_call_mod, price_call_std))
                st.metric(f"Model Put (τ={tau:.2f})", format_price(price_put_mod), format_delta(price_put_mod, price_put_std))

                # Market Comparison (same as v2.1)
                st.subheader("Market Comparison (Last Price)")
                # ... (market price display logic remains the same) ...
                if option_chain_data is not None and K is not None:
                    market_call = option_chain_data.calls[option_chain_data.calls['strike'] == K]
                    market_put = option_chain_data.puts[option_chain_data.puts['strike'] == K]
                    market_call_price = "N/A"; market_put_price = "N/A"
                    if not market_call.empty: mc = market_call.iloc[0]; market_call_price = f"${mc['lastPrice']:.2f} (Bid: {mc['bid']:.2f}, Ask: {mc['ask']:.2f})"
                    if not market_put.empty: mp = market_put.iloc[0]; market_put_price = f"${mp['lastPrice']:.2f} (Bid: {mp['bid']:.2f}, Ask: {mp['ask']:.2f})"
                    st.write(f"Market Call: {market_call_price}"); st.write(f"Market Put: {market_put_price}")
                else: st.write("Market prices not available.")
            else: st.warning("Inputs (S, K, T) incomplete. Cannot calculate prices.")

        with col1b:
            # Conceptual Economic Simulation (same as v2.1, but ensure variables are assigned for Tab 5)
            st.subheader("Conceptual Economic Simulation")
            # ... (economic sim parameters and plot logic remains the same) ...
            T0_econ = st.number_input("Initial Indicator Value", 100.0, key='econ_T0')
            mu_econ = st.slider("Indicator Drift (μ_econ)", -0.1, 0.2, 0.02, step=0.01, key='econ_mu')
            sigma_econ_val = st.slider("Indicator Volatility (σ_econ)", 0.05, 0.5, 0.15, step=0.01, key='econ_sigma')
            T_sim_econ = st.slider("Simulation Horizon (Years)", 0.5, 5.0, 1.0, step=0.5, key='econ_T')
            paths_econ = st.select_slider("Simulation Paths", options=[100, 250, 500, 1000], value=250, key='econ_paths')
            steps_per_year_econ = 50

            dt_econ = 1.0 / steps_per_year_econ
            total_steps_econ = int(T_sim_econ * steps_per_year_econ)
            sim_data, time_axis_econ = economic_model_simulation(T0_econ, tau, mu_econ, sigma_econ_val, dt_econ, total_steps_econ, paths_econ)
            # ... (Plotting logic using fig_econ remains the same) ...
            fig_econ = go.Figure()
            num_paths_to_plot = min(paths_econ, 50)
            for i in range(num_paths_to_plot): fig_econ.add_trace(go.Scatter(x=time_axis_econ, y=sim_data[i, :], mode='lines', line=dict(width=0.5), opacity=0.3, showlegend=False))
            mean_path = np.mean(sim_data, axis=0); fig_econ.add_trace(go.Scatter(x=time_axis_econ, y=mean_path, mode='lines', name=f'Mean Path (τ={tau:.2f})', line=dict(color='red', width=2)))
            mean_no_tariff = T0_econ * np.exp(mu_econ * time_axis_econ); fig_econ.add_trace(go.Scatter(x=time_axis_econ, y=mean_no_tariff, mode='lines', name='Mean Path (τ=0.00)', line=dict(color='black', width=2, dash='dash')))
            fig_econ.update_layout(title=f"Simulated Economic Indicator Paths (τ = {tau:.2f})", xaxis_title="Time (Years)", yaxis_title="Indicator Value", height=400, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_econ, use_container_width=True)


    # --- Tab 2: Historical Backtest ---
    with tabs[1]: # Backtest
        st.header(f"Historical Simulation ({pricing_model})")
        st.markdown("""
        Simulate model performance using historical stock data under a *hypothetical* past tariff scenario.
        **Limitation:** Compares model outputs (Tariff vs. No Tariff), does *not* use actual historical option prices. Includes simple transaction cost estimates.
        """)

        col2a, col2b = st.columns([1, 2])
        with col2a:
            # Inputs (same as v2.1)
            default_end_date = datetime.today(); default_start_date = default_end_date - timedelta(days=365)
            hist_start_date = st.date_input("Start Date", value=default_start_date, key=f"{tab_keys[1]}_start")
            hist_end_date = st.date_input("End Date", value=default_end_date, key=f"{tab_keys[1]}_end")
            hist_tau = st.slider("Hypothetical Tariff Rate (τ_hist)", 0.0, 1.0, tau, 0.01, key=f"{tab_keys[1]}_tau")
            hist_K_default = K if K is not None else round(current_S / 5) * 5
            hist_K = st.number_input("Strike Price (K_hist)", value=hist_K_default, key=f"{tab_keys[1]}_K", step=1.0)
            hist_T_opt = st.number_input("Option Maturity at Start (Years, T_hist)", 0.05, 10.0, 0.5, 0.05, key=f"{tab_keys[1]}_T")
            # Use sidebar vol source for backtest vol
            hist_sigma_opt = sigma_opt # Use the vol determined in sidebar (manual or GARCH)
            st.info(f"Using Volatility (σ_opt): {hist_sigma_opt:.4f} (from sidebar settings)")
            run_backtest = st.button("Run Historical Simulation", key=f"{tab_keys[1]}_run")

        if run_backtest:
            if hist_start_date >= hist_end_date:
                with col2b:
                    st.error("Error: Start date must be before end date.")
            else:
                hist_data = get_historical_data(ticker_symbol, hist_start_date, hist_end_date)
                if hist_data is not None and not hist_data.empty:
                    hist_data = hist_data.dropna(subset=['Close'])
                    if not hist_data.empty:
                        results = []
                        start_date_dt = pd.to_datetime(hist_start_date)

                        for date_idx, row in hist_data.iterrows():
                            current_date_naive = date_idx.tz_localize(None) if date_idx.tzinfo else date_idx
                            current_hist_S = row['Close']
                            days_passed = (current_date_naive - start_date_dt).days
                            current_hist_T = max(0.00001, hist_T_opt - (days_passed / 365.25))
                            if np.isclose(current_hist_T, 0):
                                continue

                            # Set up common arguments for pricing functions
                            common_args_hist = {
                                'S': current_hist_S,
                                'K': hist_K,
                                'T': current_hist_T,
                                'r': r,
                                'sigma': hist_sigma_opt
                            }
                            if pricing_model == "Modified Black-Scholes":
                                price_call_mod_hist = modified_black_scholes_with_tariff(**common_args_hist, tau=hist_tau, option_type='call', lambda_sensitivity=lambda_sens)
                                price_put_mod_hist = modified_black_scholes_with_tariff(**common_args_hist, tau=hist_tau, option_type='put', lambda_sensitivity=lambda_sens)
                                price_call_std_hist = modified_black_scholes_with_tariff(**common_args_hist, tau=0.0, option_type='call', lambda_sensitivity=lambda_sens)
                                price_put_std_hist = modified_black_scholes_with_tariff(**common_args_hist, tau=0.0, option_type='put', lambda_sensitivity=lambda_sens)
                            elif pricing_model == "Merton Jump Diffusion":
                                merton_args_hist = {**common_args_hist, **jump_params}
                                price_call_mod_hist = merton_jump_diffusion(**merton_args_hist, tau=hist_tau, option_type='call', lambda_sensitivity=lambda_sens)
                                price_put_mod_hist = merton_jump_diffusion(**merton_args_hist, tau=hist_tau, option_type='put', lambda_sensitivity=lambda_sens)
                                price_call_std_hist = merton_jump_diffusion(**merton_args_hist, tau=0.0, option_type='call', lambda_sensitivity=lambda_sens)
                                price_put_std_hist = merton_jump_diffusion(**merton_args_hist, tau=0.0, option_type='put', lambda_sensitivity=lambda_sens)
                            else:
                                price_call_mod_hist = np.nan
                                price_put_mod_hist = np.nan
                                price_call_std_hist = np.nan
                                price_put_std_hist = np.nan

                            results.append({
                                "Date": date_idx,
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
                                st.warning("No valid results generated.")
                        else:
                            results_df = pd.DataFrame(results).set_index("Date")
                            # Add simple transaction cost simulation
                            entry_price_call = results_df['Call Mod'].iloc[0]
                            exit_price_call = results_df['Call Mod'].iloc[-1]
                            pnl_call = exit_price_call - entry_price_call
                            cost_call = 2 * commission + slippage_pct * (entry_price_call + exit_price_call)
                            net_pnl_call = pnl_call - cost_call

                            with col2b:
                                st.subheader("Backtest Results")
                                # Plot historical stock price
                                fig_hist_stock = go.Figure()
                                fig_hist_stock.add_trace(go.Scatter(x=results_df.index, y=results_df['Stock Price'], mode='lines', name='Stock Price'))
                                fig_hist_stock.update_layout(title=f"{ticker_symbol} Historical Price", yaxis_title="Price ($)", height=250, margin=dict(t=30, b=10, l=10, r=10))
                                st.plotly_chart(fig_hist_stock, use_container_width=True)

                                # Plot call option prices
                                fig_hist_call = go.Figure()
                                fig_hist_call.add_trace(go.Scatter(x=results_df.index, y=results_df['Call Mod'], mode='lines', name=f'Model Call (τ={hist_tau:.2f})'))
                                fig_hist_call.add_trace(go.Scatter(x=results_df.index, y=results_df['Call Std'], mode='lines', name=f'Model Call (τ=0.00)', line=dict(dash='dash')))
                                fig_hist_call.update_layout(title=f"Simulated Call Price (K={hist_K})", yaxis_title="Option Price ($)", height=300, margin=dict(t=30, b=10, l=10, r=10))
                                st.plotly_chart(fig_hist_call, use_container_width=True)

                                # Plot call price difference
                                fig_hist_call_diff = go.Figure()
                                fig_hist_call_diff.add_trace(go.Scatter(x=results_df.index, y=results_df['Call Diff'], mode='lines', name='Call Price Difference (Mod - Std)', line=dict(color='green')))
                                fig_hist_call_diff.update_layout(title="Call Price Difference due to Tariff", yaxis_title="Price Difference ($)", height=200, margin=dict(t=30, b=10, l=10, r=10))
                                st.plotly_chart(fig_hist_call_diff, use_container_width=True)

                                # Display PnL Simulation
                                st.subheader("Simple PnL Simulation (Call Option)")
                                st.markdown("Assumes buying 1 contract at start, selling at end.")
                                st.metric("Gross PnL per Contract", f"${pnl_call*100:.2f}")
                                st.metric("Estimated Costs per Contract", f"${cost_call*100:.2f}",
                                          help=f"Commission: ${2*commission:.2f}, Slippage: ${slippage_pct * (entry_price_call + exit_price_call)*100:.2f}")
                                st.metric("Net PnL per Contract", f"${net_pnl_call*100:.2f}")
                    else:
                        with col2b:
                            st.warning("Historical data cleaned resulted in empty dataframe.")
                else:
                    with col2b:
                        st.error("Failed to fetch/process historical data.")

    # --- Tab 3: Sensitivity Analysis ---
    with tabs[2]: # Sensitivity
        st.header(f"Sensitivity Analysis ({pricing_model})")
        st.markdown("Explore how the model option price changes with different parameters.")

        # Use K and T_opt from Tab 1 if available
        base_K = K if K is not None else round(current_S / 5) * 5
        base_T = T_opt if T_opt is not None else 1.0
        if base_K <= 0 or base_T <= 0:
             st.warning("Using defaults for sensitivity (K/T from Tab 1 invalid).")
             base_K = round(current_S / 5) * 5; base_T = 1.0

        # Define parameters available for sensitivity based on model
        sens_params_available = ["Tariff Rate (τ)", "Volatility (σ_opt)", "Time to Maturity (T)"]
        if pricing_model == "Merton Jump Diffusion":
            sens_params_available.extend(["Jump Intensity (λ_jump)", "Mean Jump Size (μ_jump)", "Jump Volatility (σ_jump)"])

        sens_param = st.selectbox("Select Parameter to Vary:", sens_params_available, key=f"{tab_keys[2]}_param")
        sens_range_mult = st.slider("Analysis Range (+/- %)", 10, 100, 50, 5, key=f"{tab_keys[2]}_range") / 100.0
        sens_steps = 21

        call_prices_sens, put_prices_sens, param_values = [], [], []
        base_S, base_r, base_sigma, base_tau, base_lambda = current_S, r, sigma_opt, tau, lambda_sens
        base_jump_params = jump_params # From sidebar

        # Logic to generate param_values and calculate prices
        # ... (This section needs careful implementation for each parameter) ...
        current_val = 0.0
        if sens_param == "Tariff Rate (τ)": current_val = base_tau; min_val=0.0; max_val=1.0
        elif sens_param == "Volatility (σ_opt)": current_val = base_sigma; min_val=0.01; max_val=None
        elif sens_param == "Time to Maturity (T)": current_val = base_T; min_val=0.001; max_val=None
        elif sens_param == "Jump Intensity (λ_jump)": current_val = base_jump_params.get('jump_intensity', 0.5); min_val=0.0; max_val=None
        elif sens_param == "Mean Jump Size (μ_jump)": current_val = base_jump_params.get('jump_mean', 0.0); min_val=None; max_val=None # Can be negative
        elif sens_param == "Jump Volatility (σ_jump)": current_val = base_jump_params.get('jump_vol', 0.2); min_val=0.0; max_val=None

        half_range = abs(current_val * sens_range_mult) if not np.isclose(current_val, 0) else 0.1 * sens_range_mult
        start_val = current_val - half_range
        end_val = current_val + half_range
        if min_val is not None: start_val = max(min_val, start_val)
        if max_val is not None: end_val = min(max_val, end_val)
        # Ensure start_val is not greater than end_val if range is small or capped
        if start_val > end_val and min_val is not None: start_val = min_val
        if start_val >= end_val: # Handle edge case where range collapses
             param_values = np.linspace(start_val, start_val + half_range*0.1, sens_steps) # Create small range if needed
        else:
             param_values = np.linspace(start_val, end_val, sens_steps)


        for p_val in param_values:
            c_args = {'S': base_S, 'K': base_K, 'T': base_T, 'r': base_r, 'sigma': base_sigma, 'tau': base_tau, 'lambda_sensitivity': base_lambda}
            m_args = {**c_args, **base_jump_params}

            # Update the varying parameter
            if sens_param == "Tariff Rate (τ)": c_args['tau'] = p_val; m_args['tau'] = p_val
            elif sens_param == "Volatility (σ_opt)": c_args['sigma'] = p_val; m_args['sigma'] = p_val
            elif sens_param == "Time to Maturity (T)": c_args['T'] = p_val; m_args['T'] = p_val
            elif sens_param == "Jump Intensity (λ_jump)": m_args['jump_intensity'] = p_val
            elif sens_param == "Mean Jump Size (μ_jump)": m_args['jump_mean'] = p_val
            elif sens_param == "Jump Volatility (σ_jump)": m_args['jump_vol'] = p_val

            # Calculate prices based on model
            if pricing_model == "Modified Black-Scholes":
                call_prices_sens.append(modified_black_scholes_with_tariff(**c_args, option_type='call'))
                put_prices_sens.append(modified_black_scholes_with_tariff(**c_args, option_type='put'))
            elif pricing_model == "Merton Jump Diffusion":
                call_prices_sens.append(merton_jump_diffusion(**m_args, option_type='call'))
                put_prices_sens.append(merton_jump_diffusion(**m_args, option_type='put'))

        # Plotting (same structure as v2.1, but uses calculated sens arrays)
        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(x=param_values, y=call_prices_sens, mode='lines+markers', name='Call Price'))
        fig_sens.add_trace(go.Scatter(x=param_values, y=put_prices_sens, mode='lines+markers', name='Put Price'))
        fig_sens.update_layout(title=f"Option Price Sensitivity to {sens_param}", xaxis_title=sens_param, yaxis_title="Option Price ($)", height=450, margin=dict(t=60, b=40, l=40, r=20))
        st.plotly_chart(fig_sens, use_container_width=True)


    # --- Tab 4: Stress Test ---
    with tabs[3]: # Stress Test
        st.header(f"Stress Testing ({pricing_model})")
        st.markdown("Analyze the immediate impact of hypothetical market shocks on the calculated option price.")

        col4a, col4b = st.columns(2)
        with col4a:
            st.subheader("Define Shock Scenario")
            shock_S_pct = st.slider("Shock to Underlying Price S (%)", -50.0, 50.0, 0.0, 1.0) / 100.0
            shock_sigma_abs = st.slider("Shock to Volatility σ (Absolute Change)", -0.2, 0.5, 0.0, 0.01)
            shock_tau_abs = st.slider("Shock to Tariff Rate τ (Absolute Change)", -base_tau, 1.0-base_tau, 0.0, 0.05) # Allow increase up to 1.0

        # Baseline parameters (ensure K and T_opt are valid)
        base_K_stress = K if K is not None else round(current_S / 5) * 5
        base_T_stress = T_opt if T_opt is not None else 1.0
        if base_K_stress <= 0 or base_T_stress <= 0:
             st.warning("Using defaults for stress test (K/T from Tab 1 invalid).")
             base_K_stress = round(current_S / 5) * 5; base_T_stress = 1.0

        # Calculate baseline price
        base_args_c = {'S': current_S, 'K': base_K_stress, 'T': base_T_stress, 'r': r, 'sigma': sigma_opt, 'tau': tau, 'lambda_sensitivity': lambda_sens}
        base_args_m = {**base_args_c, **jump_params}

        if pricing_model == "Modified Black-Scholes":
            base_call = modified_black_scholes_with_tariff(**base_args_c, option_type='call')
            base_put = modified_black_scholes_with_tariff(**base_args_c, option_type='put')
        elif pricing_model == "Merton Jump Diffusion":
            base_call = merton_jump_diffusion(**base_args_m, option_type='call')
            base_put = merton_jump_diffusion(**base_args_m, option_type='put')
        else: base_call, base_put = np.nan, np.nan

        # Calculate shocked parameters
        shocked_S = current_S * (1 + shock_S_pct)
        shocked_sigma = max(0.01, sigma_opt + shock_sigma_abs) # Ensure vol > 0
        shocked_tau = max(0.0, min(1.0, tau + shock_tau_abs)) # Clamp tau between 0 and 1

        # Calculate shocked price
        shock_args_c = {'S': shocked_S, 'K': base_K_stress, 'T': base_T_stress, 'r': r, 'sigma': shocked_sigma, 'tau': shocked_tau, 'lambda_sensitivity': lambda_sens}
        shock_args_m = {**shock_args_c, **jump_params} # Jump params assumed unchanged by these shocks

        if pricing_model == "Modified Black-Scholes":
            shock_call = modified_black_scholes_with_tariff(**shock_args_c, option_type='call')
            shock_put = modified_black_scholes_with_tariff(**shock_args_c, option_type='put')
        elif pricing_model == "Merton Jump Diffusion":
            shock_call = merton_jump_diffusion(**shock_args_m, option_type='call')
            shock_put = merton_jump_diffusion(**shock_args_m, option_type='put')
        else: shock_call, shock_put = np.nan, np.nan

        with col4b:
            st.subheader("Stress Test Results")
            # Format prices
            def format_stress(p): return f"${p:.3f}" if p is not None and not np.isnan(p) else "N/A"
            def format_stress_delta(p_shock, p_base):
                 if p_shock is not None and not np.isnan(p_shock) and p_base is not None and not np.isnan(p_base):
                     delta = p_shock - p_base
                     delta_pct = (delta / p_base * 100) if not np.isclose(p_base, 0) else np.inf
                     return f"{delta:+.3f} ({delta_pct:+.1f}%)"
                 return None

            st.metric("Baseline Call Price", format_stress(base_call))
            st.metric("Shocked Call Price", format_stress(shock_call), format_stress_delta(shock_call, base_call))
            st.metric("Baseline Put Price", format_stress(base_put))
            st.metric("Shocked Put Price", format_stress(shock_put), format_stress_delta(shock_put, base_put))
            st.caption(f"Shock applied: S {shock_S_pct:+.1%}, σ change {shock_sigma_abs:+.3f}, τ change {shock_tau_abs:+.3f}")


    # --- Tab 5: Risk Metrics (VaR/ES) ---
    with tabs[4]: # Risk Metrics
        # (Logic remains the same as v2.1, uses sim_data from Tab 1)
        st.header("Risk Metrics from Economic Simulation")
        st.markdown("VaR and ES calculated from the final values of the *conceptual economic indicator* simulation (from the first tab).")
        if sim_data is not None and isinstance(sim_data, np.ndarray) and sim_data.size > 0 :
            final_values = sim_data[:, -1]
            confidence_level = st.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01, key=f"{tab_keys[4]}_conf")
            var, es = calculate_vaR_es(final_values, confidence_level)
            def format_metric(v): return f"{v:.2f}" if v is not None and not np.isnan(v) else "N/A"
            st.metric(f"Value-at-Risk (VaR) at {confidence_level:.0%}", format_metric(var))
            st.metric(f"Expected Shortfall (ES) at {confidence_level:.0%}", format_metric(es))
            # Plotting logic remains the same...
            fig_hist_risk = go.Figure(); fig_hist_risk.add_trace(go.Histogram(x=final_values, name='Distribution', nbinsx=50, histnorm='probability density'))
            if var is not None and not np.isnan(var): fig_hist_risk.add_vline(x=var, line_dash="dash", line_color="red", annotation_text=f"VaR ({confidence_level:.0%}) = {var:.2f}", annotation_position="top left")
            if es is not None and not np.isnan(es): fig_hist_risk.add_vline(x=es, line_dash="dot", line_color="orange", annotation_text=f"ES ({confidence_level:.0%}) = {es:.2f}", annotation_position="bottom left")
            fig_hist_risk.update_layout(title=f"Distribution of Final Indicator Value (T={T_sim_econ:.1f} Years)", xaxis_title="Final Indicator Value", yaxis_title="Density", height=450)
            st.plotly_chart(fig_hist_risk, use_container_width=True)
        else: st.warning("Economic simulation data not available. Run simulation on 'Live Pricing' tab.")


    # --- Tab 6: Model Explanations ---
    with tabs[5]: # Explain
        st.header("Model Explanations & Methodology")
        st.subheader("Why Use This Tool?")
        st.markdown("""
        This simulator helps analyze the *potential* impact of tariffs and market jumps on derivative prices beyond standard models. It allows you to:
        - **Compare Models:** Evaluate prices under standard assumptions (Black-Scholes), tariff-adjusted B-S, and jump-diffusion (Merton) models.
        - **Incorporate Real Data:** Use near real-time stock prices and interest rates (from FRED).
        - **Forecast Volatility:** Employ GARCH models based on historical data for potentially more realistic volatility inputs.
        - **Simulate History:** Backtest model behavior (with/without tariff) against historical stock price movements (though *not* historical option prices).
        - **Stress Test:** See the immediate price impact of hypothetical market shocks.
        - **Understand Sensitivity:** Analyze how prices react to changes in key parameters.

        While not a predictive trading tool due to inherent model limitations and data constraints (especially lack of free historical option data), it serves as a powerful analytical framework for understanding risk, exploring 'what-if' scenarios, and appreciating the complexities beyond basic models.
        """)

        st.subheader("Core Concepts & Tariff Premium")
        # ... (Keep explanation of Tariff Impact, Risk Premium λ(τ), Lambda Sensitivity) ...
        st.markdown("""
        - **Tariff Impact:** Tariffs create additional risk and potentially dampen economic activity/asset growth.
        - **Risk Premium:** Modeled as `λ(τ) = τ * λ_factor` added to the risk-free rate `r` in the drift term (`r_adj = r + λ(τ)`). Increases call prices, decreases put prices.
        - **Lambda Sensitivity:** The `λ factor` controls the strength of this effect.
        """)


        st.subheader("Pricing Models Used")
        with st.expander("Modified Black-Scholes with Tariff"):
             st.markdown(r"""
             - Standard Black-Scholes adjusted for the tariff risk premium `λ(τ)` in the drift:
               $$ d_1 = \frac{\ln(S/K) + (r + \lambda(\tau) + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}} $$
               $$ d_2 = d_1 - \sigma\sqrt{T} $$
             - Call Price: $C = S N(d_1) - K e^{-rT} N(d_2)$
             - Put Price: $P = K e^{-rT} N(-d_2) - S N(-d_1)$
             - Uses the original risk-free rate `r` for discounting $K$. $\lambda(\tau)$ affects the risk-neutral drift.
             """)
        with st.expander("Merton Jump Diffusion with Tariff"):
             st.markdown(r"""
             - Extends Black-Scholes to allow for sudden jumps in the underlying asset price, modeled as a Poisson process.
             - Assumes jumps follow a log-normal distribution with mean `μ_jump` and volatility `σ_jump`.
             - Jump Intensity `λ_jump` is the average number of jumps per year.
             - The price is calculated as a weighted average of Black-Scholes prices across different numbers of possible jumps (`n`), weighted by the Poisson probability of `n` jumps occurring.
             - The tariff premium `λ(τ)` is added to the risk-neutral drift calculation within each Black-Scholes component of the sum.
             - The formula involves an infinite sum, approximated here by summing up to `n_terms` (default 20).
             - Let $\kappa = e^{\mu_j + 0.5 \sigma_j^2} - 1$. The risk-neutral drift used in the BSM components is adjusted: $r_n = r_{base} - \lambda_{jump} \kappa + \frac{n \ln(1 + \kappa)}{T} + \lambda(\tau)$.
             - The volatility used in the BSM components is also adjusted: $\sigma_n^2 = \sigma^2 + \frac{n \sigma_j^2}{T}$.
             - Final Price: $ V = \sum_{n=0}^{\infty} \frac{e^{-\lambda'_{jump} T} (\lambda'_{jump} T)^n}{n!} \times BS(S, K, T, r_n, \sigma_n) $ where $\lambda'_{jump} = \lambda_{jump}(1+\kappa)$.
             """)

        st.subheader("Volatility Modeling (GARCH)")
        with st.expander("GARCH(1,1) Volatility Forecast"):
            st.markdown(r"""
            - Addresses the unrealistic assumption of constant volatility in Black-Scholes.
            - GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models time-varying volatility based on past returns and past volatility.
            - **GARCH(1,1) Model:**
              - Variance Equation: $ \sigma_t^2 = \omega + \alpha_1 \epsilon_{t-1}^2 + \beta_1 \sigma_{t-1}^2 $
              - Where $\sigma_t^2$ is the variance at time $t$, $\epsilon_{t-1}$ is the previous period's shock (return residual), and $\sigma_{t-1}^2$ is the previous period's variance.
              - $\omega, \alpha_1, \beta_1$ are parameters estimated from historical data.
            - **Implementation:**
              - Fetches historical daily closing prices for the specified ticker and period.
              - Calculates daily log returns.
              - Fits a GARCH(1,1) model to these returns using the `arch` library.
              - Forecasts the conditional variance for the next day.
              - Annualizes the square root of the forecasted variance to get the volatility estimate (`σ_opt`).
            - **Usefulness:** Provides a data-driven volatility estimate that reflects recent market conditions, potentially more realistic than a static manual input.
            """)

        st.subheader("Data Sources & Limitations")
        # ... (Update explanation for yfinance, add FRED, reiterate backtesting limits) ...
        st.markdown("""
        - **Stock Prices & Options:** `yfinance` library (near real-time, typically delayed). Option chain data (strikes, expiries, last/bid/ask) is fetched but reliability can vary.
        - **Risk-Free Rate:** `pandas-datareader` fetching from FRED (Federal Reserve Economic Data). Uses 3-Month Treasury Bill rate (`TB3MS`) by default. Requires internet connection.
        - **Historical Data:** `yfinance` for historical stock closing prices.
        - **Major Limitation - Backtesting:** This tool **cannot** backtest against actual historical *option* prices due to the lack of freely available, reliable historical options data. The backtest compares the *model's* theoretical outputs (with vs. without tariff) based on historical *stock* prices. Its value lies in understanding the *model's sensitivity* to past market movements under different assumptions, not in verifying real-world trading P&L.
        - **Model Risk:** All models used (GBM, BSM, Merton, GARCH) are simplifications of reality and rely on assumptions that may not hold. Results should be interpreted with caution.
        """)

        st.subheader("Other Features")
        # ... (Explain Economic Sim, Sensitivity, Stress Test, Risk Metrics, Transaction Costs) ...
        st.markdown("""
        - **Conceptual Economic Sim:** Visualizes potential tariff impact on a generic indicator (GBM model). Used for VaR/ES.
        - **Sensitivity Analysis:** Shows how option prices change when varying one input parameter.
        - **Stress Testing:** Calculates immediate price impact from user-defined shocks to S, σ, or τ.
        - **Risk Metrics (VaR/ES):** Calculated from the *conceptual economic simulation*, indicating potential downside risk in that indicator.
        - **Transaction Costs (Backtest):** Simple estimation of commission and slippage impact on the hypothetical backtest P&L for illustrative purposes.
        """)


# --- Run the App ---
if __name__ == '__main__':
    main()


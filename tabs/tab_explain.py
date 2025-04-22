# tabs/tab_explain.py
import streamlit as st
from typing import Any

# Import constants to display in text
from utils import RISK_FREE_RATE_SERIES # Use Polygon constants if needed

def render_tab_explain(
    st_tab: Any,
    risk_free_rate_series: str
    ):
    """Renders the content for the Methodology/Explain tab."""
    with st_tab:
        st.header("Methodology & Explanations")

        st.subheader("Core Concepts")
        st.markdown(rf"""
*   **Tariff Impact Model:** Primary effect modeled via risk-neutral *drift rate* adjustment. $r_{{drift}} = r + \lambda_{{sensitivity}} \times \tau$. ($r$: base rate, $\tau$: tariff rate, $\lambda_{{sensitivity}}$: user factor).
*   **Volatility ($\sigma$):** Expected annualized std dev of log returns. Manual or GARCH(1,1) estimate (see below).
*   **Risk-Free Rate (r):** From FRED (`{risk_free_rate_series}`) or manual.
*   **Time to Maturity (T):** Remaining life in years. (Expiry Date - Current Date) / 365.25.
*   **Jump Parameters (Merton Model):** If selected: $\lambda_j$: avg jumps/year, $\mu_j$: avg log jump size, $\sigma_j$: log jump size volatility.
""")

        st.subheader("Option Pricing Models Used")
        with st.expander("Modified Black-Scholes with Tariff"): st.markdown(r"""Adapts BSM. Incorporates tariff-adjusted drift ($r_{d} = r + \tau \lambda$) in $d_1, d_2$, but uses original rate $r$ for discounting $K$. Uses `black_scholes_merton_split_rates`. """)
        with st.expander("Merton Jump Diffusion with Tariff"): st.markdown(r"""Adds discrete jumps (Poisson process) to BSM diffusion. Price is weighted average of BSM prices over $n$ jumps, weighted by Poisson probabilities $P(n) = \frac{e^{-\lambda'_j T}(\lambda'_j T)^n}{n!}$ (where $\lambda'_j = \lambda_j (1 + \kappa)$). Uses compensated drift and adjusted volatility for each $n$. Sum approximated.""")

        st.subheader("Volatility Estimation & Data Handling")
        with st.expander("GARCH(1,1) Volatility Forecast"): st.markdown(r"""Estimates/forecasts volatility using historical data (`yfinance`), capturing volatility clustering ($\sigma_t^2 = \omega + \alpha_1 \epsilon_{t-1}^2 + \beta_1 \sigma_{t-1}^2$). Fitted on scaled log returns. Forecasts next day variance, then annualized: $\sigma_{annual} = \sqrt{\text{Forecasted Var}_t / 100^2 \times 252}$. Falls back if fails/unrealistic.""")
        with st.expander("Implied Volatility (IV) and Greeks"): st.markdown(r"""
*   **Implied Volatility (IV):** The $\sigma$ that makes **standard BSM price** = market price (from *live* `yfinance` data). Reflects market's vol expectation. Calculated numerically (Brent's method). *Volatility Smile/Skew* tab plots IV vs. Strike.
*   **Greeks:** Sensitivities of the *theoretical* price (Mod. BSM or Merton) to input changes. Calculated via finite differences. ($\Delta$: vs S, $\Gamma$: vs $\Delta$, $\mathcal{V}$: vs $\sigma$, $\Theta$: vs time, $\rho$: vs r).
""")
        # FIX: Update backtest data source description
        with st.expander("Data Sources and Limitations"):
            st.markdown(f"""
*   **Live Stock Price:** Polygon Previous Close (`/v2/aggs/ticker/{{ticker}}/prev`) with fallback to `yfinance`.
*   **Live Options Data (Expirations):** Polygon.io API (`/v3/reference/options/contracts`). Used to populate expiry dropdown.
*   **Live Options Data (Chain Details):** `yfinance` (Yahoo Finance). Provides live bid, ask, volume, OI, source IV for the selected expiry. Used in Live Analysis and Vol Smile tabs. Delays/quality vary.
*   **Historical Stock:** `yfinance` (adjusted).
*   **Historical Options (Backtest):** Polygon.io API (`/v1/open-close/{{optionsTicker}}/{{date}}`). Provides **End-of-Day OHLCV** data. Fetched day-by-day. **Requires Polygon API Key.**
    *   **Limitations:** This EOD endpoint **does not provide historical bid, ask, IV, or Greeks.** The backtest compares model prices against the Polygon EOD `close` price. Historical volatility for model pricing relies *solely* on the "Volatility Fallback σ" set in the sidebar (Manual or GARCH). Data availability for specific options on specific past dates is not guaranteed. Rate limits (5 calls/min) are handled with delays, making long backtests slow.
*   **Risk-Free Rate:** FRED (`{risk_free_rate_series}`). Fetched via `pandas_datareader`.
*   **News/Sentiment:** `NewsAPI` (key needed), `VADER`. API limits, source bias, sentiment limits apply.
*   **Model Limits:** Standard assumptions. Tariff effect via drift is simplistic.
*   **Backtest Accuracy:** Depends heavily on using EOD close as market proxy, using sidebar sigma for historical vol, availability of Polygon EOD data, and constant parameter assumptions.

**Overall Disclaimer:** Educational tool ONLY. NOT financial advice. Consult qualified professionals.
""")
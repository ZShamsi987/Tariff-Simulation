# tabs/tab_explain.py
import streamlit as st
from typing import Any

def render_tab_explain(
    st_tab: Any,
    dolthub_owner: str,
    dolthub_repo: str,
    risk_free_rate_series: str
    ):
    """Renders the content for the Methodology/Explain tab."""
    with st_tab:
        st.header("Methodology & Explanations")

        st.subheader("Core Concepts")
        st.markdown(rf"""
This tool analyzes option prices, incorporating potential impacts from tariffs and market jumps, alongside standard financial factors.

*   **Tariff Impact Model:** Primary effect modeled via risk-neutral *drift rate* adjustment. $r_{{drift}} = r + \lambda_{{sensitivity}} \times \tau$. ($r$: base rate, $\tau$: tariff rate, $\lambda_{{sensitivity}}$: user factor).
*   **Volatility ($\sigma$):** Expected annualized std dev of log returns. Manual or GARCH(1,1) estimate (see below).
*   **Risk-Free Rate (r):** From FRED (`{risk_free_rate_series}`) or manual.
*   **Time to Maturity (T):** Remaining life in years. (Expiry Date - Current Date) / 365.25.
*   **Jump Parameters (Merton Model):** If selected: $\lambda_j$: avg jumps/year, $\mu_j$: avg log jump size, $\sigma_j$: log jump size volatility.
""")

        st.subheader("Option Pricing Models Used")
        with st.expander("Modified Black-Scholes with Tariff"):
            st.markdown(r"""Adapts BSM. Incorporates tariff-adjusted drift ($r_{d} = r + \tau \lambda$) in $d_1, d_2$, but uses original rate $r$ for discounting $K$.
1.  **Drift:** $r_{d} = r + \tau \lambda$
2.  **d1/d2:** $ d_1 = \frac{\ln(S/K) + (r_{d} + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}} $, $ d_2 = d_1 - \sigma\sqrt{T} $
3.  **Prices:** Call $C = S N(d_1) - K e^{-rT} N(d_2)$, Put $P = K e^{-rT} N(-d_2) - S N(-d_1)$. Uses `black_scholes_merton_split_rates`.
""")

        with st.expander("Merton Jump Diffusion with Tariff"):
            st.markdown(r"""Adds discrete jumps (Poisson process) to BSM diffusion.
1.  **Process:** Changes due to diffusion ($\sigma$), tariff/jump-compensated drift, and random jumps (rate $\lambda_j$, size ~ log-normal($\mu_j, \sigma_j^2$)).
2.  **Drift Comp.:** $\kappa = e^{\mu_j + \frac{1}{2}\sigma_j^2} - 1$. Compensated Drift: $r_{drift\_base} = r_{base} + \tau \lambda - \lambda_j \kappa$.
3.  **Price:** Weighted average of BSM prices over $n$ jumps, weighted by Poisson $P(n) = \frac{e^{-\lambda'_j T}(\lambda'_j T)^n}{n!}$ (where $\lambda'_j = \lambda_j (1 + \kappa)$).
    $$ V_{MJD} = \sum_{n=0}^{\infty} P(n) \times BS_{split}(S, K, T, r_n, r_{base}, \sigma_n) $$
    *   $r_n = r_{drift\_base} + \frac{n \ln(1+\kappa)}{T}$
    *   $\sigma_n^2 = \sigma^2 + \frac{n \sigma_j^2}{T}$
    *   Sum approximated.
""")

        st.subheader("Volatility Estimation & Data Handling")
        with st.expander("GARCH(1,1) Volatility Forecast"):
            st.markdown(r"""Estimates/forecasts volatility using historical data, capturing volatility clustering.
1.  **Model:** $\sigma_t^2 = \omega + \alpha_1 \epsilon_{t-1}^2 + \beta_1 \sigma_{t-1}^2$. ($\omega$: const, $\alpha_1$: shock reaction, $\beta_1$: persistence).
2.  **Implementation:** Fitted using `arch` library on scaled historical daily log returns. Forecasts next day's variance, then annualized: $\sigma_{annual} = \sqrt{\text{Forecasted Var}_t / 100^2 \times 252}$. Falls back if fails/unrealistic.
""")
        with st.expander("Implied Volatility (IV) and Greeks"):
            st.markdown(r"""
*   **Implied Volatility (IV):** The $\sigma$ that makes **standard BSM price** = market price. Reflects market's vol expectation. Calculated numerically (Brent's method). *Volatility Smile/Skew* plots IV vs. Strike, showing non-constant market IV.
*   **Greeks:** Sensitivities of the *theoretical* price (Mod. BSM or Merton) to input changes. Calculated via finite differences.
    *   $\Delta$: Price change / $1 S change.
    *   $\Gamma$: Delta change / $1 S change.
    *   $\mathcal{V}$ (Vega): Price change / 1% vol change.
    *   $\Theta$: Price change / 1 day decrease in T (time decay).
    *   $\rho$: Price change / 1% rate change.
""")
        with st.expander("Data Sources and Limitations"):
            st.markdown(f"""
*   **Live Stock/Options:** `yfinance` (Yahoo Finance). Delays/quality vary.
*   **Historical Stock:** `yfinance` (adjusted).
*   **Historical Options (Backtest):** DoltHub SQL API (`{dolthub_owner}/{dolthub_repo}`). **Public DB - quality/completeness NOT guaranteed.** Requires `DOLTHUB_API_KEY`. *Assumes 'options' table.*
*   **Risk-Free Rate:** FRED (`{risk_free_rate_series}`). Fetched via `pandas_datareader`.
*   **News/Sentiment:** `NewsAPI` (key needed), `VADER`. API limits, source bias, sentiment limits apply.
*   **Model Limits:** Standard assumptions. Tariff effect via drift is simplistic.
*   **Backtest Accuracy:** Highly dependent on DoltHub data quality, historical vol handling, historical rates, constant parameter assumptions.

**Overall Disclaimer:** Educational tool ONLY. NOT financial advice. Consult qualified professionals.
""")
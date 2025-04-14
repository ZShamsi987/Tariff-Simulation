# utils.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm
from scipy.optimize import brentq
import yfinance as yf
from datetime import datetime, timedelta, date
import pandas_datareader.data as pdr
from arch import arch_model
import warnings
import math
import re
import requests
import json
from typing import Optional, Dict, Any, Tuple, List, Union, Callable
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- Constants ---
DEFAULT_TICKER: str = "SPY"
DEFAULT_S_FALLBACK: float = 100.0
CACHE_TTL: int = 600
MIN_PRICE_LEVEL: float = 1e-6
MIN_TIME_LEVEL: float = 1e-7
MIN_VOL_LEVEL: float = 1e-6
MAX_VOL_LEVEL: float = 5.0
IV_LOW_BOUND: float = 1e-5
IV_HIGH_BOUND: float = 5.0
IV_TOL: float = 1e-6
IV_MAX_ITER: int = 100
MERTON_SUM_N: int = 25
RISK_FREE_RATE_SERIES: str = 'TB3MS'
DEFAULT_R_FALLBACK: float = 0.05
MIN_HIST_DATA_POINTS: int = 100

# DoltHub API Config
DOLTHUB_API_KEY: Optional[str] = "dhat.v1.97u5r458m2e94rfc85o8m9o9ubjmivut4l6a2omuc6btgiddvgvg"
DOLTHUB_OWNER: str = 'post-no-preference'
DOLTHUB_REPO: str = 'options'
DOLTHUB_BRANCH: str = 'master'
DOLTHUB_API_BASE_URL: str = f"https://www.dolthub.com/api/v1alpha1/{DOLTHUB_OWNER}/{DOLTHUB_REPO}"
# FIX: Increased timeout further
DOLTHUB_TIMEOUT_SEC: int = 45 # Increased timeout to 45 seconds

# --- Warning Filters ---
# ... (Keep as is) ...
warnings.filterwarnings("ignore", category=UserWarning, message=".*Non-stationary.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*optimization failed to converge.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*'nopython' keyword.*")
warnings.filterwarnings("ignore", category=RuntimeWarning)


# --- Mathematical Models ---
# ... (Keep as is) ...
def black_scholes_merton(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call', **kwargs) -> float:
    if T <= MIN_TIME_LEVEL or sigma < 0 or S <= MIN_PRICE_LEVEL or K <= MIN_PRICE_LEVEL: return np.nan
    if np.isclose(sigma, 0) or T < MIN_TIME_LEVEL: price = max(0, S - K * np.exp(-r * T)) if option_type == 'call' else max(0, K * np.exp(-r * T) - S); return price
    sigma_sqrt_T = sigma * np.sqrt(T)
    if np.isclose(sigma_sqrt_T, 0): price = max(0, S - K * np.exp(-r * T)) if option_type == 'call' else max(0, K * np.exp(-r * T) - S); return price
    try:
        if S <= 0 or K <= 0: return np.nan
        log_S_K = np.log(S / K)
    except (ValueError, FloatingPointError): return np.nan
    d1 = (log_S_K + (r + 0.5 * sigma**2) * T) / sigma_sqrt_T; d2 = d1 - sigma_sqrt_T
    try:
        if option_type == 'call': price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == 'put': price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else: return np.nan
    except Exception: return np.nan
    return max(0, price)

def black_scholes_merton_split_rates(S: float, K: float, T: float, r_drift: float, r_discount: float, sigma: float, option_type: str = 'call') -> float:
    if T <= MIN_TIME_LEVEL or sigma < 0 or S <= MIN_PRICE_LEVEL or K <= MIN_PRICE_LEVEL: return np.nan
    if np.isclose(sigma, 0) or T < MIN_TIME_LEVEL: price = max(0, S - K * np.exp(-r_discount * T)) if option_type == 'call' else max(0, K * np.exp(-r_discount * T) - S); return price
    sigma_sqrt_T = sigma * np.sqrt(T)
    if np.isclose(sigma_sqrt_T, 0): price = max(0, S - K * np.exp(-r_discount * T)) if option_type == 'call' else max(0, K * np.exp(-r_discount * T) - S); return price
    try:
        if S <= 0 or K <= 0: return np.nan
        log_S_K = np.log(S / K)
    except (ValueError, FloatingPointError): return np.nan
    d1 = (log_S_K + (r_drift + 0.5 * sigma**2) * T) / sigma_sqrt_T; d2 = d1 - sigma_sqrt_T
    try:
        if option_type == 'call': price = S * norm.cdf(d1) - K * np.exp(-r_discount * T) * norm.cdf(d2)
        elif option_type == 'put': price = K * np.exp(-r_discount * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else: return np.nan
    except Exception: return np.nan
    return max(0, price)

def modified_black_scholes_with_tariff(S: float, K: float, T: float, r: float, sigma: float, tau: float, lambda_sensitivity: float, option_type: str = 'call', **kwargs) -> float:
    if T <= MIN_TIME_LEVEL or sigma < 0 or S <= MIN_PRICE_LEVEL or K <= MIN_PRICE_LEVEL or tau < 0 or lambda_sensitivity < 0: return np.nan
    lambda_tau = tau * lambda_sensitivity; r_adj_drift = r + lambda_tau
    if np.isclose(sigma, 0) or T < MIN_TIME_LEVEL: price = max(0, S - K * np.exp(-r * T)) if option_type == 'call' else max(0, K * np.exp(-r * T) - S); return price
    try: price = black_scholes_merton_split_rates(S, K, T, r_adj_drift, r, sigma, option_type)
    except Exception: return np.nan
    return max(0, price)

def merton_jump_diffusion(S: float, K: float, T: float, r: float, sigma: float, tau: float, jump_intensity: float, jump_mean: float, jump_vol: float, lambda_sensitivity: float, option_type: str = 'call', n_terms: int = MERTON_SUM_N, **kwargs) -> float:
    if T <= MIN_TIME_LEVEL or sigma < 0 or S <= MIN_PRICE_LEVEL or K <= MIN_PRICE_LEVEL or jump_intensity < 0 or jump_vol < 0 or tau < 0 or lambda_sensitivity < 0: return np.nan
    r_base = r; lambda_tau = tau * lambda_sensitivity
    try:
        if not np.isclose(jump_vol, 0): kappa = np.exp(jump_mean + 0.5 * jump_vol**2) - 1
        else: kappa = np.exp(jump_mean) - 1
    except OverflowError: return np.nan
    if np.isinf(kappa) or np.isnan(kappa): return np.nan
    r_jump_adj = jump_intensity * kappa; r_eff_drift_base = r_base + lambda_tau - r_jump_adj
    total_price = 0.0
    if not np.isclose(kappa, -1.0): lambda_prime = max(0.0, jump_intensity * (1 + kappa))
    else: lambda_prime = 0.0
    factorials = {};
    try:
        for n in range(n_terms + 1): factorials[n] = math.factorial(n)
    except (ValueError, OverflowError): return np.nan
    try:
        exp_neg_lambda_T = np.exp(-lambda_prime * T)
        if np.isinf(exp_neg_lambda_T): exp_neg_lambda_T = 0.0
    except OverflowError: exp_neg_lambda_T = 0.0
    for n in range(n_terms + 1):
        try:
            poisson_term_power = (lambda_prime * T) ** n
            if np.isinf(poisson_term_power) or np.isnan(poisson_term_power): break
            fac_n = factorials.get(n)
            if fac_n is None or np.isclose(fac_n, 0): continue
            poisson_prob = exp_neg_lambda_T * poisson_term_power / fac_n
            if np.isclose(poisson_prob, 0) or np.isnan(poisson_prob): continue
            if T > MIN_TIME_LEVEL and not np.isclose(kappa, -1.0):
                try: log_1_kappa = np.log1p(kappa)
                except ValueError: log_1_kappa = -np.inf
                r_n_drift = r_eff_drift_base + (n * log_1_kappa) / T
                sigma_n_sq = sigma**2 + (n * jump_vol**2) / T
            else: r_n_drift = r_eff_drift_base; sigma_n_sq = sigma**2
            sigma_n = np.sqrt(sigma_n_sq) if sigma_n_sq >= 1e-12 else 0.0
            price_n = black_scholes_merton_split_rates(S, K, T, r_n_drift, r_base, sigma_n, option_type)
            if price_n is None or np.isnan(price_n): price_n = 0.0
            total_price += poisson_prob * price_n
        except OverflowError: break
        except Exception: continue
    return max(0, total_price)

def calculate_implied_volatility(target_price: float, S: float, K: float, T: float, r: float, option_type: str, low_bound: float = IV_LOW_BOUND, high_bound: float = IV_HIGH_BOUND, tol: float = IV_TOL, max_iter: int = IV_MAX_ITER) -> float:
    if target_price <= MIN_PRICE_LEVEL or T <= MIN_TIME_LEVEL or S <= MIN_PRICE_LEVEL or K <= MIN_PRICE_LEVEL: return np.nan
    try: intrinsic = max(0, S - K * np.exp(-r * T)) if option_type == 'call' else max(0, K * np.exp(-r * T) - S)
    except OverflowError: intrinsic = 0
    if target_price < intrinsic - tol: return low_bound
    def objective(sigma: float) -> float:
        if sigma < low_bound or sigma > high_bound: return 1e12
        try: bsm_args = {'S': S, 'K': K, 'T': T, 'r': r, 'sigma': sigma, 'option_type': option_type}; price = black_scholes_merton(**bsm_args); return price - target_price if pd.notna(price) else 1e12
        except Exception: return 1e12
    try:
        f_low = objective(low_bound + 1e-9); f_high = objective(high_bound)
        if abs(f_low) < tol: return low_bound
        if abs(f_high) < tol: return high_bound
        if f_low * f_high > 0: return np.nan
        iv = brentq(objective, low_bound, high_bound, xtol=tol, rtol=tol, maxiter=max_iter)
        return iv if low_bound <= iv <= high_bound else np.nan
    except (ValueError, RuntimeError): return np.nan
    except Exception: return np.nan

def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float, tau: float, option_type: str, model_func: Callable, model_args: Optional[Dict] = None, h_S_mult: float = 0.005, h_T_div: float = 365.25, h_sigma_abs: float = 0.001, h_r_abs: float = 0.0001) -> Dict[str, float]:
    if model_args is None: model_args = {}
    g_nan = {'delta': np.nan, 'gamma': np.nan, 'vega': np.nan, 'theta': np.nan, 'rho': np.nan}
    if T <= MIN_TIME_LEVEL or sigma < 0 or S <= MIN_PRICE_LEVEL or K <= MIN_PRICE_LEVEL: return g_nan
    h_S = max(S * h_S_mult, 1e-4); h_T = max(1.0 / h_T_div, MIN_TIME_LEVEL); h_sig = max(h_sigma_abs, 1e-5); h_r = max(h_r_abs, 1e-6)
    base_args = {'S': S, 'K': K, 'T': T, 'r': r, 'sigma': sigma, 'tau': tau, **model_args, 'option_type': option_type}
    def get_price(**kwargs_override) -> float:
        args = base_args.copy(); args.update(kwargs_override)
        if args['S'] <= MIN_PRICE_LEVEL or args['sigma'] < 0: return np.nan
        args['T'] = max(args['T'], 0.0)
        try: price = model_func(**args); return price if pd.notna(price) else np.nan
        except Exception: return np.nan
    p0=get_price(); pS_p=get_price(S=S+h_S); pS_m=get_price(S=S-h_S); pSig_p=get_price(sigma=sigma+h_sig); pSig_m=get_price(sigma=sigma-h_sig)
    pT_m=get_price(T=T-h_T); pR_p=get_price(r=r+h_r); pR_m=get_price(r=r-h_r)
    d, g, v, t, rh = np.nan, np.nan, np.nan, np.nan, np.nan
    if pd.notna(pS_p) and pd.notna(pS_m) and not np.isclose(h_S, 0): d = (pS_p - pS_m) / (2 * h_S)
    if pd.notna(pS_p) and pd.notna(pS_m) and pd.notna(p0): denom_g = h_S**2;
    if denom_g > 1e-12: g = (pS_p - 2*p0 + pS_m) / denom_g
    if pd.notna(pSig_p) and pd.notna(pSig_m) and not np.isclose(h_sig, 0): v = (pSig_p - pSig_m) / (2 * h_sig) / 100.0
    if pd.notna(pT_m) and pd.notna(p0) and not np.isclose(h_T, 0): t = (pT_m - p0) / h_T / 365.25
    if pd.notna(pR_p) and pd.notna(pR_m) and not np.isclose(h_r, 0): rh = (pR_p - pR_m) / (2 * h_r) / 100.0
    return {'delta': d, 'gamma': g, 'vega': v, 'theta': t, 'rho': rh}

# --- Data Fetching ---
# ... (Keep yfinance and FRED fetching functions as they were) ...
@st.cache_data(ttl=CACHE_TTL)
def get_stock_price(ticker_symbol: str) -> Optional[float]:
    if not ticker_symbol: return None
    try:
        ticker=yf.Ticker(ticker_symbol); info=ticker.fast_info; price=info.get('last_price') or info.get('market_price') or info.get('regularMarketPrice')
        if price is None: full_info=ticker.info; price=full_info.get('currentPrice') or full_info.get('regularMarketPrice') or full_info.get('bid') or full_info.get('ask') or full_info.get('previousClose')
        if price is not None and isinstance(price,(int,float)) and price > MIN_PRICE_LEVEL: return float(price)
        hist=ticker.history(period="5d",interval="1d")
        if not hist.empty and 'Close' in hist: last_close=hist['Close'].dropna().iloc[-1];
        if pd.notna(last_close) and isinstance(last_close,(int,float)) and last_close > MIN_PRICE_LEVEL: return float(last_close)
        return None
    except Exception: return None

@st.cache_resource(ttl=CACHE_TTL)
def get_yf_ticker_obj(_ticker_symbol: str) -> Optional[yf.Ticker]:
    if not _ticker_symbol: return None
    try:
        ticker = yf.Ticker(_ticker_symbol)
        if not ticker.info:
             if ticker.history(period="1d").empty: return None
        return ticker
    except Exception as e: return None

@st.cache_data(ttl=CACHE_TTL)
def get_option_expiries(_ticker_symbol: str) -> List[str]:
    if not _ticker_symbol: return []
    try:
        ticker = get_yf_ticker_obj(_ticker_symbol)
        if ticker and hasattr(ticker, 'options') and ticker.options:
            expiries = ticker.options
            if isinstance(expiries, (list, tuple)): return list(expiries)
            else: return []
        else: return []
    except Exception: return []

@st.cache_data(ttl=CACHE_TTL)
def get_option_chain_data(_ticker_symbol: str, expiry_date: str) -> Optional[Dict[str, pd.DataFrame]]:
    if not _ticker_symbol or not expiry_date: return None
    try:
        ticker=get_yf_ticker_obj(_ticker_symbol);
        if not ticker: return None
        chain=ticker.option_chain(expiry_date)
        if chain and hasattr(chain,'calls') and hasattr(chain,'puts'):
            proc_chain={}; cols_num=['strike','lastPrice','bid','ask','volume','openInterest','impliedVolatility']
            cols_keep=['contractSymbol','strike','lastPrice','bid','ask','volume','openInterest','impliedVolatility','inTheMoney']
            for typ in ['calls','puts']:
                df_orig=getattr(chain, typ, None)
                if df_orig is not None and not df_orig.empty:
                    df=df_orig.copy(); existing_cols_keep=[col for col in cols_keep if col in df.columns]; df=df[existing_cols_keep]
                    for col in cols_num:
                        if col in df.columns: df[col]=pd.to_numeric(df[col],errors='coerce')
                    df['mid']=np.nan
                    if 'bid' in df.columns and 'ask' in df.columns: valid_bid_ask=(df['bid'].notna()&df['ask'].notna()&(df['ask']>=df['bid'])&(df['bid']>=0)); df.loc[valid_bid_ask,'mid']=(df.loc[valid_bid_ask,'bid']+df.loc[valid_bid_ask,'ask'])/2.0
                    if 'lastPrice' in df.columns: valid_last_price=df['lastPrice'].notna()&(df['lastPrice']>=0); df['mid']=df['mid'].fillna(df.loc[valid_last_price,'lastPrice'])
                    df['mid']=pd.to_numeric(df['mid'],errors='coerce'); df['mid']=df['mid'].apply(lambda x: x if pd.notna(x)and x>=0 else np.nan)
                    if 'impliedVolatility' in df.columns: df['impliedVolatility']=df['impliedVolatility'].apply(lambda x: x if pd.notna(x)and IV_LOW_BOUND<=x<=IV_HIGH_BOUND else np.nan)
                    proc_chain[typ]=df.dropna(subset=['strike'])
                else: proc_chain[typ]=pd.DataFrame(columns=cols_keep+['mid'])
            if not proc_chain.get('calls',pd.DataFrame()).empty or not proc_chain.get('puts',pd.DataFrame()).empty: return proc_chain
        return None
    except Exception: return None

@st.cache_data(ttl=3600)
def get_historical_data(ticker_symbol: str, start_date: Union[str, date], end_date: Union[str, date]) -> Optional[pd.DataFrame]:
    if not ticker_symbol: return None
    try:
        start_dt=pd.to_datetime(start_date); end_dt=pd.to_datetime(end_date); ticker_obj=yf.Ticker(ticker_symbol)
        hist=ticker_obj.history(start=start_dt,end=end_dt+timedelta(days=1),auto_adjust=True);
        if hist.empty: return None
        if not isinstance(hist.index,pd.DatetimeIndex): hist.index=pd.to_datetime(hist.index)
        if hist.index.tz is not None: hist.index=hist.index.tz_convert(None)
        hist=hist.loc[start_dt.strftime('%Y-%m-%d'):end_dt.strftime('%Y-%m-%d')].copy().sort_index()
        if hist.empty: return None
        required_cols=['Open','High','Low','Close','Volume']; missing_cols=[col for col in required_cols if col not in hist.columns]
        if missing_cols: return None
        hist=hist[hist['Close']>MIN_PRICE_LEVEL]
        if hist.empty: return None
        return hist[required_cols]
    except Exception: return None

@st.cache_data(ttl=86400)
def get_fred_rate(series_id: str = RISK_FREE_RATE_SERIES, api_key: Optional[str] = None) -> float:
    if not series_id: return DEFAULT_R_FALLBACK
    start_date = date.today() - timedelta(days=90); end_date = date.today()
    try:
        fred = pdr.DataReader(series_id,'fred',start_date,end_date,api_key=api_key)
        if not fred.empty:
            last_valid_rate=fred[series_id].dropna().iloc[-1]
            if pd.notna(last_valid_rate):
                return float(last_valid_rate)/100.0
    except Exception: pass # Silently pass on error, return default below
    return DEFAULT_R_FALLBACK

@st.cache_data(ttl=86400)
def get_historical_fred_rates(series_id: str = RISK_FREE_RATE_SERIES, start_date: Optional[Union[str, date]] = None, end_date: Optional[Union[str, date]] = None, api_key: Optional[str] = None) -> Optional[pd.Series]:
    if not series_id: return None
    if start_date is None: start_date=date.today()-timedelta(days=365*5)
    if end_date is None: end_date=date.today()
    start_dt=pd.to_datetime(start_date); end_dt=pd.to_datetime(end_date)
    try:
        fred_data=pdr.DataReader(series_id,'fred',start_dt,end_dt,api_key=api_key)
        if not fred_data.empty:
            rates=fred_data[series_id].dropna()/100.0
            if rates.empty: return None
            rates_daily=rates.reindex(pd.date_range(start_dt,end_dt,freq='D')).ffill().bfill()
            if rates_daily.isnull().all(): return None
            return rates_daily
        else: return None
    except Exception: return None # Return None on exception

@st.cache_data(ttl=3600)
def fit_garch_and_forecast(ticker_symbol: str, hist_start_date: date, hist_end_date: date, min_hist_points: int) -> Tuple[Optional[float], str]:
    if not ticker_symbol: return None, "No ticker symbol for GARCH."
    hist = get_historical_data(ticker_symbol, hist_start_date, hist_end_date)
    simple_vol: Optional[float] = None; status: str = "GARCH pre-computation failed."
    try:
        if hist is not None and not hist.empty and len(hist)>1: log_ret_sv=np.log(hist['Close']/hist['Close'].shift(1)).dropna();
        if not log_ret_sv.empty and log_ret_sv.std()>1e-9: simple_vol=log_ret_sv.std()*np.sqrt(252); simple_vol=np.clip(simple_vol,MIN_VOL_LEVEL,MAX_VOL_LEVEL)
    except Exception: pass
    if hist is None or hist.empty: status="No historical data for GARCH."; fallback_msg=f" Simple vol: {simple_vol:.4f}" if simple_vol is not None else " No fallback vol."; return simple_vol, status+fallback_msg
    hist = hist.dropna(subset=['Close'])
    if len(hist) < min_hist_points: status=f"Insufficient data ({len(hist)}<{min_hist_points}) for GARCH."; fallback_msg=f" Simple vol: {simple_vol:.4f}" if simple_vol is not None else " No fallback vol."; return simple_vol, status+fallback_msg
    try: log_ret=100*np.log(hist['Close']/hist['Close'].shift(1)).dropna()
    except Exception as e: status=f"Log return error: {e}."; fallback_msg=f" Simple vol: {simple_vol:.4f}" if simple_vol is not None else " No fallback vol."; return simple_vol, status+fallback_msg
    if log_ret.empty or log_ret.isnull().all(): status="No valid log returns."; fallback_msg=f" Simple vol: {simple_vol:.4f}" if simple_vol is not None else " No fallback vol."; return simple_vol, status+fallback_msg
    if log_ret.std() < 1e-8: status="Log returns std dev near zero."; fallback_msg=f" Simple vol: {simple_vol:.4f}" if simple_vol is not None else " No fallback vol."; return simple_vol, status+fallback_msg
    try:
        model=arch_model(log_ret,vol='Garch',p=1,q=1,rescale=False); res=model.fit(disp='off',show_warning=False,options={'maxiter':300})
        if res is None or not hasattr(res,'params'): raise ValueError("GARCH fit returned None/no params.")
        if res.convergence_flag != 0: status=f"GARCH converge failed (flag={res.convergence_flag})."; fallback_msg=f" Simple vol: {simple_vol:.4f}" if simple_vol is not None else " No fallback vol."; return simple_vol, status+fallback_msg
        forecast=res.forecast(horizon=1,reindex=False); next_day_variance_scaled=forecast.variance.iloc[-1,0]
        min_scaled_var=(MIN_VOL_LEVEL*100)**2; status_prefix="GARCH OK. "
        if next_day_variance_scaled<min_scaled_var:
            last_cond_variance_scaled=res.conditional_volatility.iloc[-1]**2
            if last_cond_variance_scaled<min_scaled_var: status="GARCH forecast & last var too low."; fallback_msg=f" Simple vol: {simple_vol:.4f}" if simple_vol is not None else " No fallback vol."; return simple_vol, status+fallback_msg
            else: next_day_variance_scaled=last_cond_variance_scaled; status_prefix="GARCH forecast low, used last cond var. "
        forecasted_vol=np.sqrt(next_day_variance_scaled)/100.0*np.sqrt(252)
        param_summary=f"ω={res.params.get('omega',np.nan):.3f}, α={res.params.get('alpha[1]',np.nan):.3f}, β={res.params.get('beta[1]',np.nan):.3f}"
        if not (MIN_VOL_LEVEL <= forecasted_vol <= MAX_VOL_LEVEL):
            status=f"GARCH vol ({forecasted_vol:.4f}) outside range [{MIN_VOL_LEVEL:.1f}-{MAX_VOL_LEVEL:.1f}]."
            forecasted_vol_clipped=np.clip(forecasted_vol,MIN_VOL_LEVEL,MAX_VOL_LEVEL)
            status += f" Clipping to {forecasted_vol_clipped:.4f}."
            return forecasted_vol_clipped, status_prefix+param_summary+" "+status
        status = status_prefix+param_summary
        return forecasted_vol, status
    except Exception as e: status=f"GARCH Model Error: {type(e).__name__}."; fallback_msg=f" Simple vol: {simple_vol:.4f}" if simple_vol is not None else " No fallback vol."; return simple_vol, status+fallback_msg

# --- DoltHub API Functions ---
# ... (Keep DoltHub functions as they were in the previous good version) ...
@st.cache_data(ttl=CACHE_TTL)
def fetch_historical_options_via_api(ticker: str, start_date: Union[str, date], end_date: Union[str, date], strike: Optional[float]=None, expiry: Optional[Union[str, date]]=None, option_type: Optional[str]=None) -> pd.DataFrame:
    if not DOLTHUB_API_KEY: st.warning("DoltHub API Key missing."); return pd.DataFrame()
    if not ticker: st.warning("Ticker required."); return pd.DataFrame()
    table_name = "option_chain"; select_cols_map = {"quote_date": "`date`", "expiration": "expiration", "strike": "strike", "type": "call_put", "implied_volatility": "vol", "delta": "delta", "gamma": "gamma", "theta": "theta", "vega": "vega", "rho": "rho", "bid": "bid", "ask": "ask"}
    select_cols_str = ", ".join(select_cols_map.values()); ticker_col = "act_symbol"; type_col = "call_put"; date_col = "`date`"
    where_clauses = [f"{ticker_col} = '{ticker.replace("'", "''")}'"]
    try: start_s = pd.to_datetime(start_date).strftime('%Y-%m-%d'); end_s = pd.to_datetime(end_date).strftime('%Y-%m-%d'); where_clauses.append(f"{date_col} BETWEEN '{start_s}' AND '{end_s}'")
    except ValueError: st.warning("Invalid date range."); return pd.DataFrame()
    if strike is not None:
        try: where_clauses.append(f"strike = {float(strike):.2f}")
        except ValueError: st.warning(f"Invalid strike '{strike}'."); return pd.DataFrame()
    if expiry is not None:
        try: where_clauses.append(f"expiration = '{pd.to_datetime(expiry).strftime('%Y-%m-%d')}'")
        except ValueError: st.warning(f"Invalid expiry '{expiry}'."); return pd.DataFrame()
    if option_type == 'call': where_clauses.append(f"{type_col} = 'C'")
    elif option_type == 'put': where_clauses.append(f"{type_col} = 'P'")
    sql_query = f"SELECT {select_cols_str} FROM {table_name} WHERE {' AND '.join(where_clauses)} ORDER BY {date_col} ASC"
    api_url = f"{DOLTHUB_API_BASE_URL}/{DOLTHUB_BRANCH}"; headers = {'Authorization': DOLTHUB_API_KEY, 'Accept': 'application/json'}; params_api = {'q': sql_query}; df = pd.DataFrame()
    try:
        response = requests.get(api_url, headers=headers, params=params_api, timeout=DOLTHUB_TIMEOUT_SEC); response.raise_for_status(); data = response.json()
        if data.get('query_execution_status') != 'Success': msg = data.get('query_execution_message', 'Unknown error.');
        if "table not found" in msg.lower(): st.error(f"DoltHub Error: Table '{table_name}' not found.")
        else: st.warning(f"DoltHub API query failed: {msg}. Query: {sql_query[:500]}...")
        return df
        rows = data.get('rows');
        if not rows: return df
        df = pd.DataFrame(rows);
        if df.empty: return df
        rename_map = {v.strip('`'): k for k, v in select_cols_map.items()}; df.rename(columns=rename_map, inplace=True)
        if 'quote_date' in df.columns: df['quote_date'] = pd.to_datetime(df['quote_date'], errors='coerce')
        if 'expiration' in df.columns: df['expiration'] = pd.to_datetime(df['expiration'], errors='coerce')
        df = df.dropna(subset=['quote_date', 'expiration'])
        num_cols = ['strike', 'implied_volatility', 'delta', 'gamma', 'theta', 'vega', 'rho', 'bid', 'ask']
        for col in num_cols:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['strike'])
        if 'type' in df.columns: df['type'] = df['type'].map({'C': 'call', 'P': 'put'}).fillna(df['type'])
        df['mid'] = np.nan
        if 'bid' in df.columns and 'ask' in df.columns:
             valid = (df['bid'].notna() & df['ask'].notna() & (df['ask'] >= df['bid']) & (df['bid'] >= 0))
             df.loc[valid, 'mid'] = (df.loc[valid, 'bid'] + df.loc[valid, 'ask']) / 2.0
             df['mid'] = df['mid'].apply(lambda x: x if pd.notna(x) and x >= 0 else np.nan)
        df = df.set_index('quote_date').sort_index()
        df['TTM'] = (df['expiration'] - df.index).dt.days / 365.25; df['TTM'] = df['TTM'].apply(lambda x: max(x, 0.0)); df['TTM'] = df['TTM'].apply(lambda x: max(x, MIN_TIME_LEVEL) if x > 0 else 0.0)
        if 'implied_volatility' in df.columns: df['implied_volatility'] = df['implied_volatility'].apply(lambda x: x if pd.notna(x) and IV_LOW_BOUND <= x <= MAX_VOL_LEVEL else np.nan)
        for col in ['open', 'high', 'low', 'close', 'volume', 'open_interest']:
             if col not in df.columns: df[col] = np.nan
        return df
    except requests.exceptions.RequestException as e: st.warning(f"DoltHub API Request Error: {e}"); return df
    except json.JSONDecodeError: st.warning(f"DoltHub API Error: Failed JSON decode. Response: {response.text[:200]}..."); return df
    except Exception as e: st.warning(f"DoltHub Data Processing Error: {type(e).__name__} - {e}"); return df

@st.cache_data(ttl=CACHE_TTL)
def fetch_recent_distinct_options_api(ticker: str, limit: int = 50) -> pd.DataFrame:
    if not DOLTHUB_API_KEY or not ticker: return pd.DataFrame()
    table_name = "option_chain"; ticker_col = "act_symbol"; expiry_col = "expiration"; strike_col = "strike"; type_col = "call_put"; date_col = "`date`"
    sql_query = f"SELECT DISTINCT {expiry_col}, {strike_col}, {type_col} FROM {table_name} WHERE {ticker_col} = '{ticker.replace("'", "''")}' ORDER BY {date_col} DESC LIMIT {limit}"
    api_url = f"{DOLTHUB_API_BASE_URL}/{DOLTHUB_BRANCH}"; headers = {'Authorization': DOLTHUB_API_KEY, 'Accept': 'application/json'}; params_api = {'q': sql_query}; df = pd.DataFrame()
    try:
        response = requests.get(api_url, headers=headers, params=params_api, timeout=DOLTHUB_TIMEOUT_SEC); response.raise_for_status(); data = response.json()
        if data.get('query_execution_status') != 'Success': return df
        rows = data.get('rows');
        if not rows: return df
        df = pd.DataFrame(rows); df.rename(columns={expiry_col: 'expiration', strike_col: 'strike', type_col: 'type'}, inplace=True)
        df['expiration'] = pd.to_datetime(df['expiration'], errors='coerce'); df['strike'] = pd.to_numeric(df['strike'], errors='coerce')
        df['type'] = df['type'].map({'C': 'CALL', 'P': 'PUT'}).fillna(df['type']); df = df.dropna()
        df = df.sort_values(by=['expiration', 'strike', 'type']).reset_index(drop=True); return df
    except Exception: return df

# --- News Fetching & Sentiment ---
def clean_text(text: Any) -> str:
    if not isinstance(text, str): return ""
    text=str(text); text=re.sub(r'http\S+|www\.\S+',' ',text); text=re.sub(r'<.*?>',' ',text); text=re.sub(r'\[.*?\]',' ',text); text=re.sub(r'[\n\r\t]',' ',text); text=re.sub(r'[^A-Za-z0-9\s.,!?$%]','',text); text=re.sub(r'\s+',' ',text).strip(); return text.lower()

@st.cache_data(ttl=1800)
def fetch_news_and_sentiment(_newsapi_client: Optional[NewsApiClient], analyzer: SentimentIntensityAnalyzer, keywords: List[str], lang: str = 'en', page_size: int = 20) -> Tuple[pd.DataFrame, str]:
    if not _newsapi_client: return pd.DataFrame(), "NewsAPI Client not initialized."
    if not keywords: return pd.DataFrame(), "No keywords provided."
    q_parts=[f'"{k}"' if' 'in k else k for k in keywords if k and isinstance(k,str)and re.match(r'^[A-Za-z0-9\s"-]+$',k)]; query=" OR ".join(q_parts)
    if not query: return pd.DataFrame(), "Invalid keywords."
    status_msg=f"Fetching news for: '{query}'..."
    try:
        from_date=(datetime.now()-timedelta(days=7)).strftime('%Y-%m-%d'); sort_by='relevancy'
        articles_response=_newsapi_client.get_everything(q=query,language=lang,sort_by=sort_by,page_size=min(page_size,100),from_param=from_date)
        if articles_response['status']=='ok':
            articles=articles_response['articles'];
            if not articles: return pd.DataFrame(), f"No articles found for: '{query}'."
            processed_articles=[]; seen_urls=set()
            for article in articles:
                url=article.get('url');
                if not url or url in seen_urls: continue
                seen_urls.add(url); title=article.get('title','')or"[No Title]"; desc=article.get('description','')or""; content=article.get('content','')or""
                text=f"{title}. {desc} {content}".strip(); cleaned=clean_text(text)
                if not cleaned or len(cleaned)<25: continue
                scores=analyzer.polarity_scores(cleaned); sentiment=scores['compound']
                pub_str=article.get('publishedAt'); pub_fmt="[No Date]"
                try: pub_dt=pd.to_datetime(pub_str).tz_convert(None); pub_fmt=pub_dt.strftime('%Y-%m-%d %H:%M')
                except Exception: pass
                processed_articles.append({'Pub':pub_fmt,'Src':article.get('source',{}).get('name','[No Source]'),'Title':title,'Sent':sentiment,'URL':url})
            if not processed_articles: return pd.DataFrame(), "Articles found, but none met analysis criteria."
            news_df=pd.DataFrame(processed_articles); status_msg=f"Fetched and analyzed {len(news_df)} articles for: '{query}'."
            return news_df.sort_values('Pub',ascending=False), status_msg
        else: err_code=articles_response.get('code','?'); err_msg=articles_response.get('message','Unknown error'); status_msg=f"NewsAPI Error ({err_code}): {err_msg}"; st.error(status_msg); return pd.DataFrame(), status_msg
    except Exception as e: status_msg=f"News Fetch/Sentiment Error: {type(e).__name__} - {e}"; st.error(status_msg); return pd.DataFrame(), status_msg

# --- Greeks Visualization ---
def create_gauge_chart(value: Optional[float], title: str, range_min: float, range_max: float, greek_symbol: str = "") -> go.Figure:
    fig = go.Figure(); display_value=value if pd.notna(value)else 0; value_text=f"{value:.4f}" if pd.notna(value)else"N/A"
    fig.add_trace(go.Indicator(mode="gauge+number",value=display_value,number={'valueformat':'.4f','suffix':''},title={'text':f"<span style='font-size:0.9em;'>{greek_symbol} {title}</span><br><span style='font-size:0.7em;'>{value_text}</span>",'font':{'size':14}},gauge={'axis':{'range':[range_min,range_max],'tickwidth':1,'tickcolor':"darkblue"},'bar':{'color':"#1f77b4",'thickness':0.3},'bgcolor':"white",'borderwidth':1,'bordercolor':"#cccccc",'steps':[{'range':[min(0,range_min),max(0,range_max)],'color':'lightgray'}],'threshold':{'line':{'color':"red",'width':3},'thickness':0.75,'value':display_value}}))
    fig.update_layout(height=220,margin=dict(l=20,r=20,t=40,b=10),font=dict(size=12)); return fig

def display_greeks_gauges(greeks: Dict[str, float], option_type: str):
    if not isinstance(greeks,dict) or not greeks or all(v is None or np.isnan(v) for v in greeks.values()): st.write("Greeks: N/A"); return
    ranges={'gamma':(0,0.5),'vega':(0,1.0),'theta':(-1.0,0.1)}
    if option_type=='put': ranges.update({'delta':(-1.0,0.0),'rho':(-0.5,0.1)})
    else: ranges.update({'delta':(0.0,1.0),'rho':(-0.1,0.5)})
    greek_info={'delta':{'name':'Delta','symbol':'Δ','tip':"Option price change per $1 change in underlying."},'gamma':{'name':'Gamma','symbol':'Γ','tip':"Delta change per $1 change in underlying."},'vega':{'name':'Vega','symbol':'ν','tip':"Option price change per 1% change in volatility."},'theta':{'name':'Theta','symbol':'Θ','tip':"Option price change per day decrease in time (decay)."},'rho':{'name':'Rho','symbol':'ρ','tip':"Option price change per 1% change in interest rate."}}
    cols=st.columns(len(greek_info))
    for i, key in enumerate(greek_info):
        val=greeks.get(key); info=greek_info[key]; range_min, range_max=ranges.get(key,(-1,1))
        with cols[i]: st.plotly_chart(create_gauge_chart(val,info['name'],range_min,range_max,info['symbol']),use_container_width=True,config={'displayModeBar':False}); st.caption(f"*{info['tip']}*",unsafe_allow_html=True)
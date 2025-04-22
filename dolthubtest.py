# dolthubtest_advanced.py
import requests
import pandas as pd
# FIX: Added datetime imports
from datetime import date, timedelta, datetime
import yfinance as yf
import time
# FIX: Import Optional from typing
from typing import Optional, Union # Union might be needed for date type hint

# --- Constants ---
DOLTHUB_API_KEY = "dhat.v1.97u5r458m2e94rfc85o8m9o9ubjmivut4l6a2omuc6btgiddvgvg"
DOLTHUB_OWNER = 'post-no-preference'
DOLTHUB_REPO = 'options'
DOLTHUB_BRANCH = 'master'
DOLTHUB_API_BASE_URL = f"https://www.dolthub.com/api/v1alpha1/{DOLTHUB_OWNER}/{DOLTHUB_REPO}"
DOLTHUB_TIMEOUT_SEC = 45 # Keep increased timeout for testing

# --- Helper Functions (Copied/adapted from utils.py for standalone testing) ---

# FIX: Removed @st.cache_data decorator
# FIX: Adjusted type hint for dates to allow string or date objects
def get_historical_stock_data(ticker_symbol: str, start_date: Union[str, date], end_date: Union[str, date]) -> Optional[pd.DataFrame]:
    """Fetches historical stock data using yfinance."""
    if not ticker_symbol: return None
    try:
        # Convert input dates to datetime objects if they are strings
        start_dt=pd.to_datetime(start_date); end_dt=pd.to_datetime(end_date)
        ticker_obj=yf.Ticker(ticker_symbol)
        # Use auto_adjust=False because we might only need 'Close' and index
        hist=ticker_obj.history(start=start_dt,end=end_dt+timedelta(days=1),auto_adjust=False)
        if hist.empty: return None
        # Ensure index is DatetimeIndex and timezone-naive
        if not isinstance(hist.index,pd.DatetimeIndex): hist.index=pd.to_datetime(hist.index)
        if hist.index.tz is not None: hist.index=hist.index.tz_convert(None)
        # Filter exactly to the requested date range
        hist=hist.loc[start_dt.strftime('%Y-%m-%d'):end_dt.strftime('%Y-%m-%d')].copy().sort_index()
        if hist.empty: return None
        # Ensure 'Close' exists before filtering
        if 'Close' not in hist.columns:
            print(f"Warning: 'Close' column missing in yfinance history for {ticker_symbol}")
            return None
        hist=hist[hist['Close']> 1e-6] # Basic check for valid price
        if hist.empty: return None
        return hist # Return the DataFrame with available columns
    except Exception as e:
        print(f"Error fetching yfinance history for {ticker_symbol}: {e}")
        return None

# --- Test Functions ---

def test_dolt_connection():
    """Tests basic connectivity by fetching table list."""
    print("--- Testing Basic Connection (SHOW TABLES) ---")
    api_url = f"{DOLTHUB_API_BASE_URL}/{DOLTHUB_BRANCH}"
    headers = {'Authorization': DOLTHUB_API_KEY, 'Accept': 'application/json'}
    params = {'q': 'SHOW TABLES;'}
    success = False
    try:
        response = requests.get(api_url, headers=headers, params=params, timeout=DOLTHUB_TIMEOUT_SEC)
        print(f"Status Code: {response.status_code}")
        response.raise_for_status()
        data = response.json()
        # print("Response JSON:") # Can be verbose
        # print(data)
        if data.get('query_execution_status') == 'Success':
            print("Connection Test: SUCCESS")
            key_name = f"Tables_in_{DOLTHUB_REPO}"
            tables = [row.get(key_name) for row in data.get('rows', []) if key_name in row]
            print("Tables found:", tables)
            success = True
        else:
            print("Connection Test: FAILED (Query Status != Success)")
            print("Message:", data.get('query_execution_message'))
    except requests.exceptions.Timeout: print("Connection Test: FAILED (Timeout)")
    except requests.exceptions.RequestException as e: print(f"Connection Test: FAILED (Request Exception: {e})")
    except Exception as e: print(f"Connection Test: FAILED (Other Exception: {e})")
    print("-" * 50)
    return success

def test_fetch_recent_distinct(ticker, limit=50): # Increased limit
    """Tests fetching recent distinct options."""
    print(f"--- Testing Fetch Recent Distinct Options ({ticker}, Limit={limit}) ---")
    if not DOLTHUB_API_KEY or not ticker: print("API Key or Ticker missing."); return None
    table_name="option_chain"; ticker_col="act_symbol"; expiry_col="expiration"; strike_col="strike"; type_col="call_put"; date_col="`date`"
    sql_query = f"SELECT DISTINCT {expiry_col}, {strike_col}, {type_col} FROM {table_name} WHERE {ticker_col} = '{ticker.replace("'", "''")}' ORDER BY {date_col} DESC LIMIT {limit}"
    api_url = f"{DOLTHUB_API_BASE_URL}/{DOLTHUB_BRANCH}"; headers = {'Authorization': DOLTHUB_API_KEY, 'Accept': 'application/json'}; params_api = {'q': sql_query}
    print(f"Query: {sql_query}")
    df = None
    try:
        response = requests.get(api_url, headers=headers, params=params_api, timeout=DOLTHUB_TIMEOUT_SEC); response.raise_for_status(); data = response.json()
        if data.get('query_execution_status') == 'Success':
            rows = data.get('rows');
            if rows:
                df = pd.DataFrame(rows); df.rename(columns={expiry_col: 'expiration', strike_col: 'strike', type_col: 'type'}, inplace=True)
                df['expiration'] = pd.to_datetime(df['expiration'], errors='coerce'); df['strike'] = pd.to_numeric(df['strike'], errors='coerce')
                df['type'] = df['type'].map({'C': 'CALL', 'P': 'PUT'}).fillna(df['type']); df = df.dropna()
                df = df.sort_values(by=['expiration', 'strike', 'type']).reset_index(drop=True)
                print("Fetch Recent Distinct: SUCCESS"); print(f"Found {len(df)} distinct options:"); print(df.head())
            else: print("Fetch Recent Distinct: SUCCESS (No rows returned)")
        else: print("Fetch Recent Distinct: FAILED (Query Status != Success)"); print("Message:", data.get('query_execution_message'))
    except requests.exceptions.Timeout: print("Fetch Recent Distinct: FAILED (Timeout)")
    except requests.exceptions.RequestException as e: print(f"Fetch Recent Distinct: FAILED (Request Exception: {e})")
    except Exception as e: print(f"Fetch Recent Distinct: FAILED (Other Exception: {e})")
    print("-" * 50)
    return df


def test_fetch_historical_range(ticker, strike_val, expiry_date_str, option_type_val, start_date_str, end_date_str):
    """Tests fetching data for a specific option over a date range."""
    print(f"\n--- Testing Fetch Historical Range ({ticker}, K={strike_val}, Exp={expiry_date_str}, Type={option_type_val}, Dates={start_date_str} to {end_date_str}) ---")
    if not DOLTHUB_API_KEY or not ticker: print("API Key or Ticker missing."); return None

    table_name = "option_chain"; select_cols_map = {"quote_date": "`date`", "expiration": "expiration", "strike": "strike", "type": "call_put", "implied_volatility": "vol", "delta": "delta", "gamma": "gamma", "theta": "theta", "vega": "vega", "rho": "rho", "bid": "bid", "ask": "ask"}
    select_cols_str = ", ".join(select_cols_map.values()); ticker_col = "act_symbol"; type_col = "call_put"; date_col = "`date`"
    where_clauses = [f"{ticker_col} = '{ticker.replace("'", "''")}'"]
    try: where_clauses.append(f"{date_col} BETWEEN '{start_date_str}' AND '{end_date_str}'")
    except ValueError: print("Invalid date format."); return None
    try: where_clauses.append(f"strike = {float(strike_val):.2f}")
    except ValueError: print(f"Invalid strike '{strike_val}'."); return None
    try: where_clauses.append(f"expiration = '{pd.to_datetime(expiry_date_str).strftime('%Y-%m-%d')}'")
    except ValueError: print(f"Invalid expiry '{expiry_date_str}'."); return None
    # Allow C/CALL or P/PUT
    option_type_upper = option_type_val.upper()
    if option_type_upper == 'CALL' or option_type_upper == 'C': where_clauses.append(f"{type_col} = 'C'")
    elif option_type_upper == 'PUT' or option_type_upper == 'P': where_clauses.append(f"{type_col} = 'P'")
    else: print("Invalid option type"); return None

    sql_query = f"SELECT {select_cols_str} FROM {table_name} WHERE {' AND '.join(where_clauses)} ORDER BY {date_col} ASC"
    api_url = f"{DOLTHUB_API_BASE_URL}/{DOLTHUB_BRANCH}"; headers = {'Authorization': DOLTHUB_API_KEY, 'Accept': 'application/json'}; params_api = {'q': sql_query}
    print(f"Query: {sql_query}")
    df = None
    start_time = time.time()
    try:
        response = requests.get(api_url, headers=headers, params=params_api, timeout=DOLTHUB_TIMEOUT_SEC); elapsed_time = time.time() - start_time
        print(f"Status Code: {response.status_code} (Took {elapsed_time:.2f}s)")
        response.raise_for_status(); data = response.json()
        if data.get('query_execution_status') == 'Success':
            rows = data.get('rows');
            if rows:
                df = pd.DataFrame(rows)
                rename_map = {v.strip('`'): k for k, v in select_cols_map.items()}; df.rename(columns=rename_map, inplace=True)
                print("Fetch Historical Range: SUCCESS")
                print(f"Found {len(df)} records:")
                print("\nDataFrame Info:")
                df.info() # Print info first to check types
                # Convert types after checking info
                if 'quote_date' in df.columns: df['quote_date'] = pd.to_datetime(df['quote_date'], errors='coerce')
                if 'expiration' in df.columns: df['expiration'] = pd.to_datetime(df['expiration'], errors='coerce')
                num_cols = ['strike', 'implied_volatility', 'delta', 'gamma', 'theta', 'vega', 'rho', 'bid', 'ask']
                for col in num_cols:
                    if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
                if 'type' in df.columns: df['type'] = df['type'].map({'C': 'call', 'P': 'put'}).fillna(df['type'])

                print("\nDataFrame Head:")
                print(df.head())
                print("\nDataFrame Description (Numeric Cols):")
                # Select only numeric columns for describe to avoid warnings
                print(df.select_dtypes(include=np.number).describe())
                print("\nMissing Values per Column:")
                print(df.isna().sum())
            else:
                print("Fetch Historical Range: SUCCESS (No rows returned for this range)")
        else:
            print("Fetch Historical Range: FAILED (Query Status != Success)")
            print("Message:", data.get('query_execution_message'))
    except requests.exceptions.Timeout: print(f"Fetch Historical Range: FAILED (Timeout after {elapsed_time:.2f}s)")
    except requests.exceptions.RequestException as e: print(f"Fetch Historical Range: FAILED (Request Exception: {e})")
    except Exception as e: print(f"Fetch Historical Range: FAILED (Other Exception: {type(e).__name__} - {e})")
    print("-" * 50)
    return df

# --- Main Execution for Testing ---
if __name__ == "__main__":
    print("Starting DoltHub API Tests...")

    if not test_dolt_connection():
        print("Aborting further tests due to connection failure.")

    else:
        test_ticker = "SPY"
        recent_options_df = test_fetch_recent_distinct(test_ticker, limit=30)

        if recent_options_df is not None and not recent_options_df.empty:
            print("\n--- Testing Historical Data for Recently Found Options ---")

            print(f"Fetching recent trading days for {test_ticker}...")
            end_date_hist = date.today() - timedelta(days=1)
            start_date_hist = end_date_hist - timedelta(days=60) # Look back further for stock data
            spy_hist = get_historical_stock_data(test_ticker, start_date_hist, end_date_hist)

            if spy_hist is not None and not spy_hist.empty:
                # Use last 15 available trading days from stock history
                recent_trading_days = spy_hist.index.strftime('%Y-%m-%d').tolist()[-15:]
                if recent_trading_days:
                    test_start = recent_trading_days[0]
                    test_end = recent_trading_days[-1]
                    print(f"Using recent trading day range: {test_start} to {test_end}")

                    # Test the first 5 distinct options found over the recent trading days
                    options_to_test = min(5, len(recent_options_df))
                    print(f"Testing historical data for the first {options_to_test} distinct options...")
                    for i in range(options_to_test):
                        test_option = recent_options_df.iloc[i]
                        test_strike = test_option['strike']
                        test_expiry = test_option['expiration'].strftime('%Y-%m-%d')
                        test_type = test_option['type'] # Should be CALL or PUT now

                        test_fetch_historical_range(
                            test_ticker, test_strike, test_expiry, test_type,
                            test_start, test_end
                        )
                        time.sleep(1) # Add a small delay between API calls
                else:
                    print("Could not determine recent trading days from stock history.")
            else:
                print(f"Could not fetch recent stock history for {test_ticker} to determine trading days.")
        else:
            print("\nSkipping historical range tests because recent options fetch failed or returned no options.")

    print("\nDoltHub API Tests Finished.")
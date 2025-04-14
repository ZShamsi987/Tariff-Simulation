# dolthubtest.py
import requests
import pandas as pd
from datetime import date, timedelta, datetime

# --- Constants (Copy relevant ones from utils.py) ---
DOLTHUB_API_KEY = "dhat.v1.97u5r458m2e94rfc85o8m9o9ubjmivut4l6a2omuc6btgiddvgvg" # Use your key
DOLTHUB_OWNER = 'post-no-preference'
DOLTHUB_REPO = 'options'
DOLTHUB_BRANCH = 'master'
DOLTHUB_API_BASE_URL = f"https://www.dolthub.com/api/v1alpha1/{DOLTHUB_OWNER}/{DOLTHUB_REPO}"
DOLTHUB_TIMEOUT_SEC = 45 # Keep increased timeout for testing

def test_dolt_connection():
    """Tests basic connectivity by fetching table list."""
    print("--- Testing Basic Connection (SHOW TABLES) ---")
    api_url = f"{DOLTHUB_API_BASE_URL}/{DOLTHUB_BRANCH}"
    headers = {'Authorization': DOLTHUB_API_KEY, 'Accept': 'application/json'}
    params = {'q': 'SHOW TABLES;'}
    try:
        response = requests.get(api_url, headers=headers, params=params, timeout=DOLTHUB_TIMEOUT_SEC)
        print(f"Status Code: {response.status_code}")
        response.raise_for_status()
        data = response.json()
        print("Response JSON:")
        print(data)
        if data.get('query_execution_status') == 'Success':
            print("Connection Test: SUCCESS")
            # FIX: Correctly access the table name from the response structure
            tables = [row.get('Tables_in_options') for row in data.get('rows', []) if 'Tables_in_options' in row]
            print("Tables found:", tables)
        else:
            print("Connection Test: FAILED (Query Status != Success)")
            print("Message:", data.get('query_execution_message'))
    except requests.exceptions.Timeout:
        print("Connection Test: FAILED (Timeout)")
    except requests.exceptions.RequestException as e:
        print(f"Connection Test: FAILED (Request Exception: {e})")
    except Exception as e:
        print(f"Connection Test: FAILED (Other Exception: {e})")
    print("-" * 50)

def test_fetch_recent_distinct(ticker, limit=20):
    """Tests fetching recent distinct options."""
    print(f"--- Testing Fetch Recent Distinct Options ({ticker}, Limit={limit}) ---")
    if not DOLTHUB_API_KEY or not ticker:
        print("API Key or Ticker missing.")
        return

    # Use correct table and column names
    table_name = "option_chain"; ticker_col = "act_symbol"; expiry_col = "expiration"
    strike_col = "strike"; type_col = "call_put"; date_col = "`date`" # Use backticks
    sql_query = f"SELECT DISTINCT {expiry_col}, {strike_col}, {type_col} FROM {table_name} WHERE {ticker_col} = '{ticker.replace("'", "''")}' ORDER BY {date_col} DESC LIMIT {limit}"

    api_url = f"{DOLTHUB_API_BASE_URL}/{DOLTHUB_BRANCH}"
    headers = {'Authorization': DOLTHUB_API_KEY, 'Accept': 'application/json'}
    params_api = {'q': sql_query}
    print(f"Query: {sql_query}")

    try:
        response = requests.get(api_url, headers=headers, params=params_api, timeout=DOLTHUB_TIMEOUT_SEC)
        print(f"Status Code: {response.status_code}")
        response.raise_for_status()
        data = response.json()
        print("Response JSON:")
        print(data)
        if data.get('query_execution_status') == 'Success':
            rows = data.get('rows')
            if rows:
                df = pd.DataFrame(rows)
                # Rename columns to match expected usage ('expiration', 'strike', 'type')
                df.rename(columns={expiry_col: 'expiration', strike_col: 'strike', type_col: 'type'}, inplace=True)
                df['expiration'] = pd.to_datetime(df['expiration'], errors='coerce')
                df['strike'] = pd.to_numeric(df['strike'], errors='coerce')
                # Map type 'C'/'P' back to 'CALL'/'PUT' for display consistency if needed
                df['type'] = df['type'].map({'C': 'CALL', 'P': 'PUT'}).fillna(df['type'])
                df = df.dropna()

                print("Fetch Recent Distinct: SUCCESS")
                print(f"Found {len(df)} distinct options:")
                print(df.head())
            else:
                print("Fetch Recent Distinct: SUCCESS (No rows returned)")
        else:
            print("Fetch Recent Distinct: FAILED (Query Status != Success)")
            print("Message:", data.get('query_execution_message'))
    except requests.exceptions.Timeout:
        print("Fetch Recent Distinct: FAILED (Timeout)")
    except requests.exceptions.RequestException as e:
        print(f"Fetch Recent Distinct: FAILED (Request Exception: {e})")
    except Exception as e:
        print(f"Fetch Recent Distinct: FAILED (Other Exception: {e})")
    print("-" * 50)


def test_fetch_historical(ticker, start_date_str, end_date_str, strike_val, expiry_date_str, option_type_val):
    """Tests fetching historical data for a specific option."""
    print(f"--- Testing Fetch Historical ({ticker}, K={strike_val}, Exp={expiry_date_str}, Type={option_type_val}, Dates={start_date_str} to {end_date_str}) ---")
    if not DOLTHUB_API_KEY or not ticker:
        print("API Key or Ticker missing.")
        return

    # Use correct table and columns
    table_name = "option_chain"; select_cols_map = {"quote_date": "`date`", "expiration": "expiration", "strike": "strike", "type": "call_put", "implied_volatility": "vol", "delta": "delta", "gamma": "gamma", "theta": "theta", "vega": "vega", "rho": "rho", "bid": "bid", "ask": "ask"}
    select_cols_str = ", ".join(select_cols_map.values()); ticker_col = "act_symbol"; type_col = "call_put"; date_col = "`date`"
    where_clauses = [f"{ticker_col} = '{ticker.replace("'", "''")}'"]
    try: where_clauses.append(f"{date_col} BETWEEN '{start_date_str}' AND '{end_date_str}'")
    except ValueError: print("Invalid date format."); return
    try: where_clauses.append(f"strike = {float(strike_val):.2f}")
    except ValueError: print(f"Invalid strike '{strike_val}'."); return
    try: where_clauses.append(f"expiration = '{pd.to_datetime(expiry_date_str).strftime('%Y-%m-%d')}'")
    except ValueError: print(f"Invalid expiry '{expiry_date_str}'."); return
    if option_type_val == 'call': where_clauses.append(f"{type_col} = 'C'")
    elif option_type_val == 'put': where_clauses.append(f"{type_col} = 'P'")
    else: print("Invalid option type"); return

    sql_query = f"SELECT {select_cols_str} FROM {table_name} WHERE {' AND '.join(where_clauses)} ORDER BY {date_col} ASC"
    api_url = f"{DOLTHUB_API_BASE_URL}/{DOLTHUB_BRANCH}"; headers = {'Authorization': DOLTHUB_API_KEY, 'Accept': 'application/json'}; params_api = {'q': sql_query}
    print(f"Query: {sql_query}")

    try:
        response = requests.get(api_url, headers=headers, params=params_api, timeout=DOLTHUB_TIMEOUT_SEC)
        print(f"Status Code: {response.status_code}")
        response.raise_for_status()
        data = response.json()
        print("Response JSON (first 500 chars):")
        print(str(data)[:500] + "...")
        if data.get('query_execution_status') == 'Success':
            rows = data.get('rows')
            if rows:
                df = pd.DataFrame(rows)
                print("Fetch Historical: SUCCESS")
                print(f"Found {len(df)} records:")
                rename_map = {v.strip('`'): k for k, v in select_cols_map.items()}; df.rename(columns=rename_map, inplace=True)
                print("DataFrame head after rename:")
                print(df.head())
            else:
                print("Fetch Historical: SUCCESS (No rows returned)")
        else:
            print("Fetch Historical: FAILED (Query Status != Success)")
            print("Message:", data.get('query_execution_message'))
    except requests.exceptions.Timeout:
        print("Fetch Historical: FAILED (Timeout)")
    except requests.exceptions.RequestException as e:
        print(f"Fetch Historical: FAILED (Request Exception: {e})")
    except Exception as e:
        print(f"Fetch Historical: FAILED (Other Exception: {e})")
    print("-" * 50)

# --- Main Execution for Testing ---
if __name__ == "__main__":
    print("Starting DoltHub API Tests...")
    test_dolt_connection()
    test_ticker = "SPY"
    test_fetch_recent_distinct(test_ticker)

    # --- Example parameters for historical fetch ---
    # Try a shorter, more recent date range first
    test_start = (date.today() - timedelta(days=30)).strftime('%Y-%m-%d') # Look back 30 days
    test_end = (date.today() - timedelta(days=1)).strftime('%Y-%m-%d')

    # Use parameters from the successful "Fetch Recent Distinct" output if possible
    # Example based on your output:
    test_strike = 640.00
    test_expiry = '2025-05-30'
    test_type = 'call' # Change to 'put' if needed

    print(f"\nAttempting fetch for {test_ticker} {test_strike} {test_type} expiring {test_expiry} from {test_start} to {test_end}")
    test_fetch_historical(test_ticker, test_start, test_end, test_strike, test_expiry, test_type)

    # Example 2: Try a slightly different combination
    test_strike_2 = 555.00
    test_expiry_2 = '2025-05-30'
    test_type_2 = 'put'
    print(f"\nAttempting fetch for {test_ticker} {test_strike_2} {test_type_2} expiring {test_expiry_2} from {test_start} to {test_end}")
    test_fetch_historical(test_ticker, test_start, test_end, test_strike_2, test_expiry_2, test_type_2)

    print("\nDoltHub API Tests Finished.")
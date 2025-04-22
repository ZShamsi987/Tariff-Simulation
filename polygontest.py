# polygontest.py
import requests
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
import yfinance as yf
import time
from typing import Optional, Union, Any, Dict # Added Dict

# --- Constants ---
POLYGON_API_KEY = "op1ReRo2JXPVfoV1i6LAUudkBeyokKIn" # Your Polygon Key
POLYGON_BASE_URL = "https://api.polygon.io"
POLYGON_TIMEOUT_SEC = 30
POLYGON_MAX_LIMIT = 1000

# --- API Request Helper ---
def _polygon_api_request(endpoint: str, params: Optional[Dict] = None, expected_status: str = 'OK') -> Optional[Dict]:
    if not POLYGON_API_KEY: print("Warning: Polygon API Key not configured."); return None
    base_params = {'apiKey': POLYGON_API_KEY}
    if params: base_params.update(params)
    try:
        response = requests.get(f"{POLYGON_BASE_URL}{endpoint}", params=base_params, timeout=POLYGON_TIMEOUT_SEC)
        print(f"DEBUG URL: {response.url}") # Print URL for debugging
        response.raise_for_status()
        data = response.json()
        status = data.get('status')
        if status != expected_status and status != 'DELAYED':
            print(f"Warning: Polygon API Error ({endpoint}): Status '{status}'. Message: {data.get('error', data.get('message', 'Unknown'))}")
            return None
        return data
    except requests.exceptions.Timeout: print(f"Warning: Polygon API request timed out ({endpoint})."); return None
    except requests.exceptions.RequestException as e: print(f"Warning: Polygon API request failed ({endpoint}): {e}"); return None
    except json.JSONDecodeError: print(f"Warning: Polygon API Error: Failed to decode JSON response ({endpoint})."); return None
    except Exception as e: print(f"Warning: Polygon API Unexpected Error ({endpoint}): {type(e).__name__} - {e}"); return None


# --- Test Functions ---
def test_get_expirations(ticker):
    print(f"\n--- Testing Get Expirations ({ticker}) ---")
    all_expirations = set()
    endpoint = "/v3/reference/options/contracts"
    params = {"underlying_ticker": ticker, "limit": POLYGON_MAX_LIMIT, "expired": "false", "order": "asc", "sort": "expiration_date"}
    page = 1
    while True:
        print(f"Fetching expirations page {page}...")
        data = _polygon_api_request(endpoint, params)
        time.sleep(12.5) # Rate limit
        if not data or 'results' not in data: break
        results = data.get('results', [])
        if not results: break
        found_new = False
        for contract in results:
            if 'expiration_date' in contract:
                exp_date = contract['expiration_date']
                if exp_date not in all_expirations: all_expirations.add(exp_date); found_new = True
        next_url = data.get('next_url')
        if next_url:
            try: cursor = next_url.split('cursor=')[-1].split('&')[0]; params['cursor'] = cursor; page += 1
            except Exception as e: print(f"Could not parse cursor: {e}"); break
        else: break
    today_str = date.today().strftime('%Y-%m-%d')
    future_expiries = sorted([exp for exp in all_expirations if exp >= today_str])
    print(f"Found {len(future_expiries)} future expiration dates. First 10: {future_expiries[:10]}")
    print("-" * 50)
    return future_expiries

def test_get_strikes(ticker, expiration):
    print(f"\n--- Testing Get Strikes ({ticker} / {expiration}) ---")
    all_strikes = set()
    endpoint = "/v3/reference/options/contracts"
    params = {"underlying_ticker": ticker, "expiration_date": expiration, "limit": POLYGON_MAX_LIMIT, "order":"asc", "sort":"strike_price"}
    page = 1
    while True:
        print(f"Fetching strikes page {page}...")
        data = _polygon_api_request(endpoint, params)
        time.sleep(12.5) # Rate limit
        if not data or 'results' not in data: break
        results = data.get('results', [])
        if not results: break
        for contract in results:
            if 'strike_price' in contract: all_strikes.add(contract['strike_price'])
        next_url = data.get('next_url')
        if next_url:
            try: cursor = next_url.split('cursor=')[-1].split('&')[0]; params['cursor'] = cursor; page += 1
            except Exception as e: print(f"Could not parse cursor: {e}"); break
        else: break
    sorted_strikes = sorted(list(all_strikes))
    print(f"Found {len(sorted_strikes)} unique strikes. Sample: {sorted_strikes[:5]}...{sorted_strikes[-5:]}")
    print("-" * 50)
    return sorted_strikes

def generate_polygon_ticker(underlying: str, exp_date: Union[str, date], strike: float, opt_type: str) -> str:
    exp_obj = pd.to_datetime(exp_date); exp_str = exp_obj.strftime('%y%m%d')
    opt_char = 'C' if opt_type.lower() == 'call' else 'P'
    strike_int = int(round(strike * 1000)); strike_str = f"{strike_int:08d}"
    return f"O:{underlying.upper()}{exp_str}{opt_char}{strike_str}"

def test_fetch_eod(option_ticker, query_date_str):
    print(f"\n--- Testing Fetch EOD ({option_ticker} / {query_date_str}) ---")
    endpoint = f"/v1/open-close/{option_ticker}/{query_date_str}"
    params = {'adjusted': 'true'}
    data = _polygon_api_request(endpoint, params)
    time.sleep(12.5) # Rate limit (might need adjustment if called rapidly)
    if data and data.get('status') == 'OK':
        print("Fetch EOD: SUCCESS")
        print("Data:", data)
    elif data and data.get('status') == 'NOT_FOUND':
        print("Fetch EOD: SUCCESS (Status: NOT_FOUND - No data for this date/ticker)")
    else:
        print("Fetch EOD: FAILED")
        print("Response Data:", data)
    print("-" * 50)
    return data

# --- Main Execution for Testing ---
if __name__ == "__main__":
    print("Starting Polygon API Tests...")
    ticker = "SPY"

    # 1. Test Expirations
    expirations = test_get_expirations(ticker)

    if expirations:
        # 2. Test Strikes for a near-term expiration
        near_expiry = expirations[0] # Test the first available expiration
        strikes = test_get_strikes(ticker, near_expiry)

        if strikes:
            # 3. Test EOD fetch for a specific contract near ATM on a recent date
            # Find near-the-money strike (crude method)
            try:
                stock_price_data = _polygon_api_request(f"/v2/aggs/ticker/{ticker}/prev", {'adjusted': 'true'})
                time.sleep(12.5)
                current_price = float(stock_price_data['results'][0]['c']) if stock_price_data and stock_price_data.get('results') else 500.0
                atm_strike = min(strikes, key=lambda x: abs(x-current_price))
                print(f"\nSelected ATM strike {atm_strike} for EOD test based on price {current_price:.2f}")
            except Exception:
                atm_strike = strikes[len(strikes)//2] # Fallback to middle strike
                print(f"\nCould not get current price, using middle strike {atm_strike} for EOD test.")

            test_date = (date.today() - timedelta(days=5)).strftime('%Y-%m-%d') # Try 5 days ago

            # Generate Call Ticker
            call_ticker = generate_polygon_ticker(ticker, near_expiry, atm_strike, 'call')
            test_fetch_eod(call_ticker, test_date)

            # Generate Put Ticker
            put_ticker = generate_polygon_ticker(ticker, near_expiry, atm_strike, 'put')
            test_fetch_eod(put_ticker, test_date)
        else:
            print("\nSkipping EOD test because strikes fetch failed.")
    else:
        print("\nSkipping strike/EOD tests because expiration fetch failed.")

    print("\nPolygon API Tests Finished.")
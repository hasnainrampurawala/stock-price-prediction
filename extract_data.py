import os
import requests
from datetime import datetime
from dotenv import load_dotenv
import time
import json

# load api key from .env file
load_dotenv()

API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
BASE_URL = "https://www.alphavantage.co/query"

# semiconductor companies
SYMBOLS = ["NVDA", "INTC", "AMD"]

FUNCTION = "TIME_SERIES_MONTHLY_ADJUSTED"

def get_last_5_years_monthly_data(symbol, max_retries=3):
    params = {
        "function": FUNCTION,
        "symbol": symbol,
        "apikey": API_KEY,
        "datatype": "json"
    }

    for attempt in range(1, max_retries + 1):
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()

        # handling rate limits or api errors
        if "Note" in data:
            print(f"Rate limit hit for {symbol}")
            time.sleep(60)
            continue

        time_series_key = "Monthly Adjusted Time Series"

        monthly_data = data[time_series_key]

        # setting the cut off date to last 5 years
        cutoff_date = datetime.now().replace(year=datetime.now().year - 5)

        filtered_data = {
            date: values
            for date, values in monthly_data.items()
            if datetime.strptime(date, "%Y-%m-%d") >= cutoff_date
        }

        return filtered_data

    raise RuntimeError(f"Failed to fetch data for {symbol} after {max_retries} retries")

if __name__ == "__main__":
    if not API_KEY:
        raise EnvironmentError("API key not found")

    all_stocks_data = {}

    # compiling stock data into a json
    for symbol in SYMBOLS:
        print(f"Fetching data for {symbol}")
        all_stocks_data[symbol] = get_last_5_years_monthly_data(symbol)
        print(f"Finished {symbol}")
        time.sleep(15)

    output_file = "semiconductor_ai_stocks_5yr.json"
    with open(output_file, "w") as f:
        json.dump(all_stocks_data, f, indent=4)

    print(f"\nData successfully saved to {output_file}")
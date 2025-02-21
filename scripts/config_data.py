import yfinance as yf
import pandas as pd

def fetch_btc_data(start_date="2018-01-01", end_date="2024-01-01"):
    btc_data = yf.download("BTC-USD", start=start_date, end=end_date, interval="1d")
    btc_data = btc_data[['Adj Close']]
    btc_data['Returns'] = btc_data['Adj Close'].pct_change()
    btc_data.dropna(inplace=True)
    btc_data.to_csv("data/btc_prices.csv")
    print("Bitcoin price data saved successfully!")

if __name__ == "__main__":
    fetch_btc_data()
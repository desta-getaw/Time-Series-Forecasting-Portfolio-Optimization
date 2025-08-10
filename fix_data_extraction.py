import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

# --- 1. Data Extraction (Fixed Version) ---
# Define the tickers and the date range for the data extraction.
tickers = ['TSLA', 'BND', 'SPY']
start_date = '2015-07-01'
end_date = '2025-07-31'

print(f"Fetching data for {', '.join(tickers)} from {start_date} to {end_date}...")

# Download data for each ticker individually to handle timeouts better
data_dict = {}
for ticker in tickers:
    try:
        print(f"Downloading {ticker}...")
        ticker_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if not ticker_data.empty:
            data_dict[ticker] = ticker_data['Adj Close']
            print(f"Successfully downloaded {ticker}")
        else:
            print(f"No data available for {ticker}")
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")

# Combine all successful downloads
if data_dict:
    data = pd.DataFrame(data_dict)
    print(f"Successfully downloaded data for: {list(data.columns)}")
else:
    print("No data was downloaded successfully")
    data = pd.DataFrame()

print("Data fetching complete.")
print("-" * 50)

# Check if we have data before proceeding
if data.empty or len(data.columns) == 0:
    print("ERROR: No data was downloaded. Please check your internet connection and try again.")
    print("You can also try downloading fewer tickers or a shorter time period.")
else:
    print("Data shape:", data.shape)
    print("Available columns:", list(data.columns))
    
    # Display the first few rows
    print("\nFirst 5 rows of the data:")
    print(data.head())
    
    # Check for missing values
    print("\nMissing values in each column:")
    print(data.isnull().sum())
    
    # Handle missing values
    data = data.ffill()  # Use ffill() instead of deprecated fillna(method='ffill')
    
    print("\nMissing values after forward fill:")
    print(data.isnull().sum())
    
    # Now you can safely use the data
    print("\nData is ready for analysis!")
    print("Available tickers:", list(data.columns))

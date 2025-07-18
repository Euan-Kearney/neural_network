import yfinance as yf
import pandas as pd
import numpy as np
import sys

def get_data() -> yf.Ticker:

    print("Enter a Wall Street ticker symbol: ")
    ticker_symbol = input().strip().upper()
    data = yf.Ticker(ticker_symbol)

    # Check if the ticker symbol is valid
    while data.info.get('regularMarketPrice') is None:
        print(f"Ticker symbol '{ticker_symbol}' is not available. Please enter a valid Wall Street ticker symbol: ")
        ticker_symbol = input().strip().upper()
        data = yf.Ticker(ticker_symbol)
    print(f"Data for {ticker_symbol} retrieved successfully.")
    return data



def main():
    
    data = get_data()
    

if __name__ == "__main__":
    main()

import yfinance as yf
import pandas as pd
import numpy as np
from info import DataInput
import sys

def get_data() -> yf.Ticker:
    """
    Retrieves data for a Wall Street ticker symbol using the yfinance library
    Returns a yf.Ticker object containing the data for the specified ticker symbol.
    """

    #prompt the user for a Wall Street ticker symbol
    print("Enter a Wall Street ticker symbol: ")
    ticker_symbol = input().strip().upper()
    data = yf.Ticker(ticker_symbol)

    # Check if the ticker symbol is valid
    while data.info.get('regularMarketPrice') is None:
        print(f"Ticker symbol '{ticker_symbol}' is not available. Please enter a valid Wall Street ticker symbol: ")
        ticker_symbol = input().strip().upper()
        data = yf.Ticker(ticker_symbol)

    # Data retrieval is successful
    print(f"Data for {ticker_symbol} retrieved successfully.")
    return data

#def parse_data(data: yf.Ticker) -> pd.DataFrame:

def main():
    
    data = get_data()
    input_data = DataInput(data)
    processed_data = input_data.get_data()
    
     
    print(processed_data.tail(30))  
    
    

if __name__ == "__main__":
    main()

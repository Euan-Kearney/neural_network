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

def get_prediction_period() -> int:
    """
    Prompts the user for a prediction period in days.
    Returns the prediction period as an integer.
    """
    print("Enter the prediction period in days (1 <= predictions <= 30 ): ")
    #prompt the user for a prediction period
    #ensure the input is a valid integer between 1 and 30
    while True:
        try:
            prediction_period = int(input().strip())
            if 1 <= prediction_period <= 30:
                return prediction_period
            else:
                print("Please enter a valid number between 1 and 30.")
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 30.")
            continue
        
    

def main():
    
    data = get_data()
    prediction_period = get_prediction_period()
    input_data = DataInput(data, prediction_period)
    processed_data = input_data.get_data()
    print(processed_data)
    

if __name__ == "__main__":
    main()

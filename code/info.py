import yfinance as yf
import pandas as pd
import numpy as np



class DataInput:
    """
    Class to initilaise and store data retrieved from yfinance for neural network input.
    """
    def __init__(self, data: yf.Ticker):
        self.data = data
        self.info = data.history(period="3y")
        
        self.open = self.info['Open']
        self.close = self.info['Close']
        self.high = self.info['High']
        self.low = self.info['Low']
        self.volume = self.info['Volume']
        
        self.daily_returns = self.info['Close'].pct_change()
        self.price_change = self.info['Close'] - self.info['Open']
        self.high_low_diff = self.info['High'] - self.info['Low']
        self.close_open_ration = self.info['Close'] / self.info['Open']
        self.volume_change = self.info['Volume'].pct_change()
        
        self.sma_5 = self.info['Close'].rolling(window=5).mean()
        self.sma_10 = self.info['Close'].rolling(window=10).mean()
        self.ema_10 = self.info['Close'].ewm(span=10).mean()
        self.rolling_std_10 = self.info['Close'].rolling(window=10).std()
        
    def get_data(self) -> pd.DataFrame:
        """
        Returns a DataFrame containing the processed data for neural network input.
        """
        return pd.DataFrame({
            'Open': self.open,
            'Close': self.close,
            'High': self.high,
            'Low': self.low,
            'Volume': self.volume,
            'Daily Returns': self.daily_returns,
            'Price Change': self.price_change,
            'High-Low Diff': self.high_low_diff,
            'Close/Open Ratio': self.close_open_ration,
            'Volume Change': self.volume_change,
            'SMA 5': self.sma_5,
            'SMA 10': self.sma_10,
            'EMA 10': self.ema_10,
            'Rolling Std 10': self.rolling_std_10
        }).dropna()
        
        
        
        
        
        
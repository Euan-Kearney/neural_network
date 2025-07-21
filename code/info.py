import yfinance as yf
import pandas as pd
import numpy as np



class DataInput:
    """
    Class to initilaise and store data retrieved from yfinance for neural network input.
    """
    def __init__(self, data: yf.Ticker, prediction_period: int):
        """
        Initializes the DataInput class with data from a yf.Ticker object.
        Args:
            data (yf.Ticker): The yf.Ticker object containing stock data.
            prediction_period (int): The number of days to predict.
        """
        # Gathers the necessary data from the yf.Ticker object
        self.data = data
        self.prediction_period = prediction_period
        self.lookback_period = get_lookback_period(prediction_period)
        self.info = data.history(period= f'{self.lookback_period}d')
        
        #basic data attributes
        self.open = self.info['Open']
        self.close = self.info['Close']
        self.high = self.info['High']
        self.low = self.info['Low']
        self.volume = self.info['Volume']
        
        #derived features
        self.daily_returns = self.info['Close'].pct_change()
        self.price_change = self.info['Close'] - self.info['Open']
        self.high_low_diff = self.info['High'] - self.info['Low']
        self.close_open_ratio = self.info['Close'] / self.info['Open']
        self.volume_change = self.info['Volume'].pct_change()
        
        #technical indicators
        self.sma_5 = self.info['Close'].rolling(window=5).mean()
        self.sma_10 = self.info['Close'].rolling(window=10).mean()
        self.ema_10 = self.info['Close'].ewm(span=10).mean()
        self.rolling_std_10 = self.info['Close'].rolling(window=10).std()
        self.rsi = calculate_rsi(self.info['Close'])
        
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
            'Close/Open Ratio': self.close_open_ratio,
            'Volume Change': self.volume_change,
            'SMA 5': self.sma_5,
            'SMA 10': self.sma_10,
            'EMA 10': self.ema_10,
            'Rolling Std 10': self.rolling_std_10,
            'RSI': self.rsi
        }).dropna().reset_index(drop=True)
        
def get_lookback_period(prediction_period: int, multiplier: int =6) -> int:
    """
    Calculates the lookback period based on the prediction period.
    Args:
        prediction_period (int): The number of days to predict.
    Returns:
        lookback_period (int): The lookback period, which is 6 times the prediction period.
    """
    return prediction_period * multiplier

def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculates the Relative Strength Index (RSI) for a given data series.
    Args:
        data (pd.Series): The data series to calculate RSI for.
        period (int): The period over which to calculate RSI.
    Returns:
        pd.Series: The RSI values.
    """
    delta = data.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
     
    return rsi
        
        
        
        
        
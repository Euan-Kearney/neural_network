import yfinance as yf
import pandas as pd
import numpy as np
import torch
from torch import nn
from info import DataInput
from sklearn.preprocessing import MinMaxScaler


class StockPredictionModel(nn.Module):
    def __init__(self, prediction_period):
        super().__init__()
        self.hidden_size = 32
        self.num_layers = 2
        self.lstm = nn.LSTM(input_size=15,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,     
                            dropout=0.25                  
                            )

        self.fc = nn.Linear(self.hidden_size, prediction_period)
    def forward(self, x):
        
        #initialise long and short term memory to 0
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        #forward pass
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        #takes final index of the timestep
        out = out[:, -1, :]
        
        out = self.fc(out)
        
        return out

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
        
def prepare_data(data_input: DataInput) -> tuple[np.ndarray, np.ndarray]: 
    """
    Prepares the data for training the neural network.
    Args:
        data_input (DataInput): An instance of DataInput containing the data and lookback period.
    Returns:
        tuple: A tuple containing the input features (X) and output labels (Y) as numpy arrays.
    """
    
    data = data_input.get_data() 
    
    features = data.columns.tolist()
    
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaled_data = data.copy()
    scaled_data[features] = scaler_X.fit_transform(data[features])
   
    
    X, Y = [], []
    

    # Create input and output arrays for the neural network
    for i in range(len(scaled_data) - data_input.lookback_period - data_input.prediction_period + 1):
        X.append(scaled_data[features].iloc[i:i + data_input.lookback_period].values)
        Y.append(scaled_data['Close'] \
                 .iloc[i + data_input.lookback_period : i + data_input.lookback_period + data_input.prediction_period].values)
    X, Y = np.array(X), np.array(Y)
  
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    return X, Y
    
    
    
    
    

        
def train_model(data_input: DataInput):  
    
    
    X_numpy, y_numpy = prepare_data(data_input)
    X = torch.from_numpy(X_numpy)
    y = torch.from_numpy(y_numpy)
    print(f"X shape: {X.shape} Y shape: {y.shape}")
    
    

def main():
    
    data = get_data()
    prediction_period = get_prediction_period()
    input_data = DataInput(data, prediction_period)
    
    train_model(input_data)
    

if __name__ == "__main__":
    main()

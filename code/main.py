import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from info import DataInput
from sklearn.preprocessing import MinMaxScaler


def plot_predictions(data_input: DataInput, prediction):

    data_numpy = data_input.close.to_numpy()
    lookback = data_numpy[data_numpy.size - data_input.lookback_period:]
    projection = np.append(lookback, prediction.reshape(-1))
    x_values = np.arange(len(projection))
    
    plt.plot(x_values[:data_input.lookback_period], projection[:data_input.lookback_period], color='royalblue')
    plt.plot(x_values[data_input.lookback_period - 1:], projection[data_input.lookback_period - 1:], color='crimson')
    plt.title('Stock projection')
    plt.savefig(f'{data_input.data.ticker} Projection')
    plt.show()


class StockPredictionModel(nn.Module):
    def __init__(self, prediction_period):
        super().__init__()
        self.hidden_size = 32
        self.num_layers = 2
        self.lstm = nn.LSTM(input_size=15,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=0.25
                            )

        self.fc = nn.Linear(self.hidden_size*2, prediction_period)

    def forward(self, x):

        # initialise long and short term memory to 0
        h0 = torch.zeros(self.num_layers*2, x.size(
            0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers*2, x.size(
            0), self.hidden_size).to(x.device)

        # forward pass
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # takes final index of the timestep
        out = self.fc(out[:, -1, :])
        return out


def get_data() -> yf.Ticker:
    """
    Retrieves the data for a Wall Street stock ticker, based on user input
    Returns:
        yf.Ticker: The RSI values.
    """

    # prompt the user for a Wall Street ticker symbol
    print("Enter a Wall Street ticker symbol: ")
    ticker_symbol = input().strip().upper()
    data = yf.Ticker(ticker_symbol)

    # Check if the ticker symbol is valid
    while data.info.get('regularMarketPrice') is None:
        print(
            f"Ticker symbol '{ticker_symbol}' is not available. Please enter a valid Wall Street ticker symbol: ")
        ticker_symbol = input().strip().upper()
        data = yf.Ticker(ticker_symbol)

    # Data retrieval is successful
    print(f"Data for {ticker_symbol} retrieved successfully.")
    return data


def get_prediction_period() -> int:
    """
    Prompts user for how many days the stock prices should be predicted
    Returns:
        int: prediction period (days)
    """
    print("Enter the prediction period in days (1 <= predictions <= 30 ): ")
    # prompt the user for a prediction period
    # ensure the input is a valid integer between 1 and 30
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


def prepare_data(
        data_input: DataInput, scaler_X: MinMaxScaler, scaler_y: MinMaxScaler) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepares the data for training the neural network.
    Args:
        data_input (DataInput): An instance of DataInput containing the data and lookback period.
    Returns:
        tuple: A tuple containing the input features (X) and output labels (Y) as numpy arrays.
    """

    data = data_input.get_data()
    features = data.columns.tolist()
    # Scale input data
    scaled_input = data.copy()
    scaled_input[features] = scaler_X.fit_transform(data[features])
    # Scale output data
    scaled_output = data['Close'].copy()
    scaled_output = scaler_y.fit_transform(data['Close'].values.reshape(-1, 1))

    X, Y, final_window = [], [], []
    # Create input and output arrays for the neural network
    for i in range(len(scaled_input) - data_input.lookback_period - data_input.prediction_period + 1):
        X.append(scaled_input[features].iloc[i:i +
                 data_input.lookback_period].values)
        # Flatten converts to appropriate shape for pytorch
        Y.append(scaled_output[i + data_input.lookback_period: i +
                 data_input.lookback_period + data_input.prediction_period].flatten())
    X = np.array(X)
    Y = np.array(Y)
    final_window = scaled_input[features].iloc[-data_input.lookback_period:].values
    return X, Y, final_window


def run_model(data_input: DataInput):
    """
    Trains, tests and runs the stock prediction network
    Args:
        data_input (DataInput): An instance of DataInput containing the data and lookback period.
    """
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    X_numpy, y_numpy, final_window_numpy = prepare_data(
        data_input, scaler_X, scaler_y)
    X = torch.from_numpy(X_numpy).float()
    y = torch.from_numpy(y_numpy).float()
    final_window = torch.from_numpy(final_window_numpy).float().unsqueeze(0)
    torch.manual_seed(42)
    test_model = StockPredictionModel(data_input.prediction_period)
    torch.manual_seed(42)
    stock_prediction_model = StockPredictionModel(data_input.prediction_period)
    # train test split of 75%
    train_split = int(len(X) * 0.8)
    X_train, y_train = X[:train_split], y[:train_split]
    X_test, y_test = X[train_split:], y[train_split:]

    loss_fn = nn.MSELoss()
    test_model_optimiser = torch.optim.Adam(test_model.parameters(), lr=0.01)
    prediction_model_optimiser = torch.optim.Adam(
        stock_prediction_model.parameters(), lr=0.01)

    torch.manual_seed(22)
    epochs = 200

    test_model_training_losses = []
    test_model_testing_losses = []
    test_model_epoch_count = []

    print("Training and testing with current data:")
    # Training and testing with present data
    for epoch in range(epochs):
        # Train
        test_model.train()
        y_pred = test_model(X_train)
        loss = loss_fn(y_pred, y_train)
        test_model_optimiser.zero_grad()
        loss.backward()
        test_model_optimiser.step()

        # Test
        with torch.inference_mode():
            test_pred = test_model(X_test)
            test_loss = loss_fn(test_pred, y_test.type(torch.float))
            if epoch % 10 == 0:
                test_model_epoch_count.append(epoch)
                test_model_training_losses.append(loss.detach().numpy())
                test_model_testing_losses.append(test_loss.detach().numpy())
                print(
                    f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")

    torch.manual_seed(22)
    prediction_model_losses = []
    prediction_model_epoch_count = []

    # Training with all data for future predictions
    print("Training with all available")
    for epoch in range(epochs):
        # Train
        stock_prediction_model.train()
        y_pred = stock_prediction_model(X)
        loss = loss_fn(y_pred, y)
        prediction_model_optimiser.zero_grad()
        loss.backward()
        prediction_model_optimiser.step()
        if epoch % 10 == 0:
            prediction_model_epoch_count.append(epoch)
            prediction_model_losses.append(loss.detach().numpy())
            print(f"Epoch: {epoch} | Train Loss: {loss}")

    stock_prediction_model.eval()
    with torch.inference_mode():
        raw_prediction = stock_prediction_model(final_window)
    scaled_prediction = raw_prediction.detach().numpy().squeeze().reshape(-1, 1)
    prediction = scaler_y.inverse_transform(scaled_prediction)
    print(f"prediction shape: {prediction.shape}")
    print(f"Predicted values: \n{prediction}")
    plot_predictions(data_input, prediction)


def main():
    data = get_data()
    prediction_period = get_prediction_period()
    input_data = DataInput(data, prediction_period)
    run_model(input_data)


if __name__ == "__main__":
    main()

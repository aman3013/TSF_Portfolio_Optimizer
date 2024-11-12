import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from tensorflow.compat.v1.keras.models import Sequential
from tensorflow.compat.v1.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import yfinance as yf
from pmdarima import auto_arima

# Load the stock data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    return data

# Plot the data
def plot_data(data, title="Data", xlabel="Date", ylabel="Price"):
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

# Check if the data is stationary using ADF test
def check_stationarity(data):
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(data)
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])

    if result[1] <= 0.05:
        print("The data is stationary.")
    else:
        print("The data is not stationary.")

# Split the data into train and test sets
def split_data(data, test_size=0.2):
    split_point = int(len(data) * (1 - test_size))
    train, test = data[:split_point], data[split_point:]
    return train, test

# Fit ARIMA model
def fit_arima(train, order=(5, 1, 0)):
    model = ARIMA(train, order=order)
    model_fit = model.fit()
    return model_fit

# Fit SARIMA model
def fit_sarima(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    return model_fit

# Prepare data for LSTM
def prepare_lstm_data(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0])
        y.append(data[i, 0])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, y

# Build LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Evaluate model performance
def evaluate_model(test, forecast):
    mae = mean_absolute_error(test, forecast)
    rmse = sqrt(mean_squared_error(test, forecast))
    mape = np.mean(np.abs((test - forecast) / test)) * 100
    return mae, rmse, mape

# Plot the forecasts from different models
def plot_forecasts(test, arima_forecast, sarima_forecast, lstm_forecast):
    plt.figure(figsize=(10, 6))
    plt.plot(test, label='True', color='black')
    plt.plot(test.index, arima_forecast, label='ARIMA', color='blue')
    plt.plot(test.index, sarima_forecast, label='SARIMA', color='red')
    plt.plot(test.index, lstm_forecast, label='LSTM', color='green')
    plt.legend(loc='best')
    plt.title('Model Forecasts Comparison')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

# Portfolio Optimization
def optimize_portfolio():
    # Example placeholder for portfolio optimization logic (can be based on expected returns, variance, etc.)
    # This would usually require the asset returns and covariances
    print("Portfolio optimization logic should be implemented here.")

# Main function to run the analysis
# Main function to run the analysis
def main():
    bnd_data = load_data('../data/cleaned_BND_data.csv')
    spy_data = load_data('../data/cleaned_SPY_data.csv')
    tsla_data = load_data('../data/cleaned_TSLA_data.csv')  # Replace with the actual TSLA dataset file path
    
    # Combine the datasets if necessary, or process them separately
    datasets = {
        'BND': bnd_data,
        'SPY': spy_data,
        'TSLA': tsla_data  # Add your TSLA dataset here
    }
    
    # Step 1: Load and plot the data (Choose one dataset to load for plotting)
    data = bnd_data  # Or choose 'spy_data' or 'tsla_data' based on your analysis
    plot_data(data, title="Stock Data", xlabel="Date", ylabel="Price")
    
    # Step 2: Check stationarity of the data
    check_stationarity(data['Price'])
    
    # Step 3: Split the data into train and test sets (80% train, 20% test)
    train, test = split_data(data['Price'])
    
    # Step 4: Fit ARIMA model
    arima_model = fit_arima(train)
    arima_forecast = arima_model.predict(n_periods=len(test))
    
    # Step 5: Fit SARIMA model
    sarima_model = fit_sarima(train)
    sarima_forecast = sarima_model.predict(start=test.index[0], end=test.index[-1])
    
    # Step 6: Prepare data for LSTM model
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Price'].values.reshape(-1, 1))
    X_train, y_train = prepare_lstm_data(scaled_data[:len(train)], window_size=60)
    X_test, y_test = prepare_lstm_data(scaled_data[len(train)-60:], window_size=60)
    
    # Step 7: Build and train LSTM model
    lstm_model = build_lstm_model((X_train.shape[1], 1))
    lstm_model.fit(X_train, y_train, epochs=10, batch_size=32)
    
    # Step 8: Forecast using LSTM
    lstm_forecast = lstm_model.predict(X_test)
    lstm_forecast = scaler.inverse_transform(lstm_forecast)
    
    # Step 9: Evaluate model performance
    arima_mae, arima_rmse, arima_mape = evaluate_model(test, arima_forecast)
    sarima_mae, sarima_rmse, sarima_mape = evaluate_model(test, sarima_forecast)
    lstm_mae, lstm_rmse, lstm_mape = evaluate_model(test, lstm_forecast)
    
    print(f"ARIMA - MAE: {arima_mae}, RMSE: {arima_rmse}, MAPE: {arima_mape}")
    print(f"SARIMA - MAE: {sarima_mae}, RMSE: {sarima_rmse}, MAPE: {sarima_mape}")
    print(f"LSTM - MAE: {lstm_mae}, RMSE: {lstm_rmse}, MAPE: {lstm_mape}")
    
    # Step 10: Plot forecasts for all models
    plot_forecasts(test, arima_forecast, sarima_forecast, lstm_forecast)
    
    # Step 11: Portfolio optimization
    optimize_portfolio()

if __name__ == "__main__":
    main()


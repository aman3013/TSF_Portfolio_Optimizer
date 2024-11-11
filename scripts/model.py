import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from scipy.stats import norm
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import pickle
import warnings

warnings.filterwarnings('ignore')

def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    return data

def check_missing_values(data):
    print("Missing values in the dataset:")
    print(data.isnull().sum())

def plot_closing_price(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data['Close'])
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Closing Price Over Time')
    plt.show()

def plot_daily_change(data):
    daily_change = data['Close'].pct_change()
    plt.figure(figsize=(10, 6))
    plt.plot(daily_change)
    plt.xlabel('Date')
    plt.ylabel('Daily Percentage Change')
    plt.title('Daily Percentage Change')
    plt.show()
    return daily_change

def analyze_volatility(daily_change):
    rolling_mean = daily_change.rolling(window=30).mean()
    rolling_std = daily_change.rolling(window=30).std()
    plt.figure(figsize=(10, 6))
    plt.plot(daily_change, label='Daily Change')
    plt.plot(rolling_mean, label='Rolling Mean')
    plt.plot(rolling_std, label='Rolling Standard Deviation')
    plt.legend()
    plt.show()

def check_stationarity(data):
    result = adfuller(data)
    if result[1] <= 0.05:
        print('The data is stationary.')
    else:
        print('The data is not stationary.')

def plot_differenced_data(data):
    diff_data = data['Close'].diff().dropna()
    plt.figure(figsize=(10, 6))
    plt.plot(diff_data)
    plt.xlabel('Date')
    plt.ylabel('Differenced Close Price')
    plt.title('Differenced Close Price Over Time')
    plt.show()
    return diff_data

def train_sarima_model(data):
    train_size = int(0.8 * len(data))
    train_data, test_data = data[0:train_size], data[train_size:len(data)]
    
    model = auto_arima(train_data, start_p=1, start_d=1, start_q=1,
                       max_p=3, max_d=2, max_q=3, start_P=0, seasonal=True)
    print(model.summary())
    
    sarima_model = SARIMAX(train_data, order=model.order, seasonal_order=model.seasonal_order)
    sarima_model_fit = sarima_model.fit()
    
    forecast = sarima_model_fit.forecast(steps=len(test_data))
    
    evaluate_model(test_data, forecast, 'SARIMA')
    plot_forecast(train_data, test_data, forecast)
    
    return sarima_model_fit

def train_arima_model(data):
    train_size = int(0.8 * len(data))
    train_data, test_data = data[0:train_size], data[train_size:len(data)]
    
    model_arima = ARIMA(train_data, order=(5,1,0))
    model_arima_fit = model_arima.fit()
    
    forecast_arima = model_arima_fit.forecast(steps=len(test_data))
    
    evaluate_model(test_data, forecast_arima, 'ARIMA')
    plot_forecast(train_data, test_data, forecast_arima)
    
    return model_arima_fit

def train_lstm_model(data):
    train_size = int(0.8 * len(data))
    train_data, test_data = data[0:train_size], data[train_size:len(data)]
    
    X_train, y_train = prepare_data_for_lstm(train_data)
    X_test, y_test = prepare_data_for_lstm(test_data)
    
    model_lstm = build_lstm_model(X_train)
    model_lstm.fit(X_train, y_train, epochs=50, batch_size=1, verbose=2)
    
    forecast_lstm = model_lstm.predict(X_test)
    
    evaluate_model(y_test, forecast_lstm, 'LSTM')
    plot_forecast(y_train, y_test, forecast_lstm)
    
    return model_lstm

def prepare_data_for_lstm(data, look_back=60):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i])
        y.append(data[i])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y

def build_lstm_model(X_train):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def evaluate_model(actual, forecast, model_name):
    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mape = mean_absolute_percentage_error(actual, forecast)
    
    print(f'{model_name} Model Metrics:')
    print('MAE: ', mae)
    print('RMSE: ', rmse)
    print('MAPE: ', mape)

def plot_forecast(train_data, test_data, forecast):
    plt.figure(figsize=(10, 6))
    plt.plot(train_data, label='Training Data')
    plt.plot(test_data, label='Actual Values')
    plt.plot(forecast, label='Forecast', linestyle='--')
    plt.legend()
    plt.show()

def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def main():
    # Load and preprocess data
    tsla_data = load_data('../data/cleaned_TSLA_data.csv')
    check_missing_values(tsla_data)
    plot_closing_price(tsla_data)
    
    daily_change = plot_daily_change(tsla_data)
    analyze_volatility(daily_change)
    
    check_stationarity(tsla_data['Close'])
    diff_data = plot_differenced_data(tsla_data)
    check_stationarity(diff_data)
    
    # Train and evaluate models
    sarima_model = train_sarima_model(diff_data)
    arima_model = train_arima_model(diff_data)
    lstm_model = train_lstm_model(diff_data)
    
    # Save models
    save_model(sarima_model, '../models/sarima_model.pkl')
    save_model(arima_model, '../models/arima_model.pkl')
    lstm_model.save('../models/lstm_model.h5')

if __name__ == "__main__":
    main()

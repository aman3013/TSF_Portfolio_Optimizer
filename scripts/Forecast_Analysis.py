import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
import joblib

warnings.filterwarnings('ignore')

def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    return data

def plot_data(data, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def check_stationarity(data):
    result = adfuller(data)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])

def split_data(data, train_size=0.8):
    train_size = int(len(data) * train_size)
    train, test = data[:train_size], data[train_size:]
    return train, test

def fit_arima(train):
    model = pm.auto_arima(train, seasonal=False, trace=True, stepwise=True)
    return model

def fit_sarima(train, order=(1,1,1), seasonal_order=(1,1,1,5)):
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
    return model.fit()

def evaluate_model(true, pred):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    mape = np.mean(np.abs((true - pred) / true)) * 100
    return mae, rmse, mape

def plot_forecasts(test, arima_forecast, sarima_forecast):
    plt.figure(figsize=(10, 6))
    plt.plot(test.index, test, label='Test Data', color='orange')
    plt.plot(test.index, arima_forecast, label='ARIMA Forecast', color='blue')
    plt.plot(test.index, sarima_forecast, label='SARIMA Forecast', color='green')
    plt.legend()
    plt.title('TSLA Stock Price Forecasting')
    plt.show()

def save_models(arima_model, sarima_model):
    joblib.dump(arima_model, 'arima_model.pkl')
    sarima_model.save('sarima_model.pkl')

def load_models():
    arima_model = joblib.load('arima_model.pkl')
    sarima_model = SARIMAXResults.load('sarima_model.pkl')
    return arima_model, sarima_model

def forecast_future(model, periods):
    forecast = model.get_forecast(steps=periods)
    forecast_values = forecast.predicted_mean
    conf_int = forecast.conf_int(alpha=0.05)
    return forecast_values, conf_int

def plot_future_forecast(historical_data, forecast_values, conf_int):
    plt.figure(figsize=(10, 6))
    plt.plot(historical_data.index, historical_data['Adj Close'], label='Historical Data', color='blue')
    plt.plot(forecast_values.index, forecast_values, label='Forecasted Data', color='red')
    plt.fill_between(forecast_values.index, 
                     conf_int.iloc[:, 0], 
                     conf_int.iloc[:, 1], 
                     color='pink', alpha=0.3, label='95% Confidence Interval')
    plt.title('Tesla Stock Price Forecast', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Load and preprocess data
    data = load_data('../data/cleaned_TSLA_data.csv')
    plot_data(data['Adj Close'], 'TSLA Adjusted Close Price', 'Date', 'Price')
    
    # Check stationarity
    check_stationarity(data['Adj Close'])
    
    # Make data stationary
    data['Adj Close Differenced'] = data['Adj Close'].diff().dropna()
    check_stationarity(data['Adj Close Differenced'].dropna())
    
    # Split data
    train, test = split_data(data['Adj Close'])
    
    # Fit models
    arima_model = fit_arima(train)
    sarima_model = fit_sarima(train)
    
    # Generate forecasts
    arima_forecast = arima_model.predict(n_periods=len(test))
    sarima_forecast = sarima_model.forecast(steps=len(test))
    
    # Evaluate models
    print("ARIMA Performance:", evaluate_model(test, arima_forecast))
    print("SARIMA Performance:", evaluate_model(test, sarima_forecast))
    
    # Plot forecasts
    plot_forecasts(test, arima_forecast, sarima_forecast)
    
    # Save models
    save_models(arima_model, sarima_model)
    
    # Load models for future forecasting
    arima_model, sarima_model = load_models()
    
    # Forecast future prices
    forecast_periods = 252  # Forecast for 1 year (252 business days)
    sarima_forecast, sarima_conf_int = forecast_future(sarima_model, forecast_periods)
    
    # Create a DataFrame for the forecasted values
    forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_periods, freq='B')
    forecast_df = pd.DataFrame(sarima_forecast, index=forecast_index, columns=['Forecast'])
    
    # Plot future forecast
    plot_future_forecast(data, forecast_df, sarima_conf_int)

if __name__ == "__main__":
    main()
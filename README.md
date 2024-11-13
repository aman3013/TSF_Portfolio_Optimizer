# TSF_Portfolio_Optimizer

## Project Overview

This project aims to enhance portfolio management strategies by leveraging time series forecasting models. By predicting market trends and optimizing asset allocation, the goal is to help financial analysts make informed investment decisions. The primary assets analyzed include **Tesla (TSLA)**, **Vanguard Total Bond Market ETF (BND)**, and **S&P 500 ETF (SPY)**. 

The project uses historical financial data obtained from YFinance, and the insights gained are used to recommend changes to client portfolios to optimize returns and minimize risks.

## Business Objective

Guide Me in Finance (GMF) Investments is a forward-thinking financial advisory firm that focuses on personalized portfolio management strategies. By utilizing time series forecasting models, GMF aims to:
- Predict future market trends.
- Optimize asset allocation for clients' portfolios.
- Enhance portfolio performance by capitalizing on market opportunities while minimizing risks.


## Project Tasks

### Task 1: Preprocess and Explore the Data

1. **Data Loading and Exploration:**
   - Fetch historical data for TSLA, BND, and SPY from YFinance.
   - Clean and prepare the data by handling missing values, ensuring proper data types, and normalizing/scaling as needed.

2. **Exploratory Data Analysis (EDA):**
   - Visualize closing prices to identify trends.
   - Calculate and plot daily percentage changes to observe volatility.
   - Perform outlier detection on significant anomalies in returns.
   - Analyze volatility by calculating rolling means and standard deviations.

3. **Seasonality and Trend Decomposition:**
   - Decompose the time series into trend, seasonal, and residual components using `statsmodels`.

4. **Risk Assessment:**
   - Analyze volatility and compute risk metrics like VaR (Value at Risk) and Sharpe Ratio.


### Task 2: Develop Time Series Forecasting Models

1. **Model Selection:**
   - Choose between ARIMA, SARIMA, or LSTM based on the data characteristics (e.g., seasonality, long-term dependencies).
   
2. **Model Training and Evaluation:**
   - Split the data into training and test sets.
   - Train the selected model and forecast future stock prices for Tesla.
   - Use metrics such as MAE, RMSE, and MAPE for model evaluation.

3. **Parameter Optimization:**
   - Optimize model parameters using grid search or the `auto_arima` function from the `pmdarima` library.

### Task 3: Forecast Future Market Trends

1. **Model Forecasting:**
   - Use the trained model to generate future price predictions for Tesla.
   - Forecast for a period of 6-12 months and include confidence intervals.

2. **Forecast Analysis:**
   - Visualize the forecasted data alongside historical data.
   - Analyze trends, volatility, and potential market risks.
   - Identify market opportunities and risks based on predicted trends.

### Task 4: Optimize Portfolio Based on Forecast

1. **Portfolio Construction:**
   - Use a sample portfolio with three assets: TSLA, BND, and SPY.
   - Compute annual returns and daily compounded returns for each asset.

2. **Portfolio Optimization:**
   - Compute the covariance matrix to understand the relationship between asset returns.
   - Optimize portfolio weights to maximize the Sharpe ratio (risk-adjusted return).
   - Adjust asset allocation based on forecasted market trends.

3. **Performance Metrics:**
   - Analyze portfolio performance through metrics like average return, volatility, and Sharpe ratio.
   - Visualize portfolio performance and risk-return analysis.

## Skills and Techniques

- **Time Series Forecasting Methods:** ARIMA, SARIMA, LSTM.
- **Data Analysis Tools:** Python, YFinance, Pandas, Numpy, Matplotlib, Statsmodels, Keras/TensorFlow.
- **Financial Metrics:** Sharpe Ratio, VaR, Covariance Matrix, Portfolio Optimization.
- **Machine Learning Optimization:** Grid Search, Auto ARIMA, Hyperparameter Tuning.
- **Risk Management:** Volatility Analysis, Risk-Adjusted Return Evaluation.

## Data Sources

- **Tesla (TSLA):** High-growth, high-risk stock in the consumer discretionary sector.
- **Vanguard Total Bond Market ETF (BND):** Bond ETF tracking U.S. investment-grade bonds.
- **S&P 500 ETF (SPY):** ETF tracking the S&P 500 Index for diversified market exposure.

The data will cover the period from January 1, 2015, to October 31, 2024.

## Setup and Requirements

### Prerequisites

To run the project, you need the following libraries installed:

- Python 3.8 or higher
- Pandas
- Numpy
- Matplotlib
- YFinance
- Statsmodels
- Pmdarima
- Keras/TensorFlow (for LSTM)

You can install the necessary libraries using pip:

```bash
pip install pandas numpy matplotlib yfinance statsmodels pmdarima keras tensorflow

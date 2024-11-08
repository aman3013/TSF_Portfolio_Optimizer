import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose

# Stop warnings
import warnings
warnings.filterwarnings('ignore')

sys.path.append('../scripts')


def download_data(assets, start_date, end_date, data_folder="data"):
    """Download historical stock data from Yahoo Finance."""
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    for asset in assets:
        data = yf.download(asset, start=start_date, end=end_date)
        file_path = os.path.join(data_folder, f"{asset}_data.csv")
        data.to_csv(file_path)  # Save each asset's data to CSV in 'data' folder
        print(f"{asset} data saved to {file_path}")


def clean_data(file_paths, columns_order):
    """Load and clean data for each asset."""
    for asset, file_path in file_paths.items():
        df = pd.read_csv(file_path)
        df.rename(columns={'Price': 'Date'}, inplace=True)
        df = df.drop([0, 1]).reset_index(drop=True)
        df = df[columns_order]
        
        cleaned_path = f"../data/cleaned_{asset}_data.csv"
        df.to_csv(cleaned_path, index=False)
        print(f"Cleaned data saved to {cleaned_path}")


def preprocess_and_explore_data(file_path, output_path):
    """Preprocess the dataset and perform exploratory analysis."""
    # Load dataset
    df = pd.read_csv(file_path)
    print(f"\nProcessing {file_path}")
    print("Initial Data Sample:") 
    print(df.head())

    # Handle missing values by interpolation (suitable for time series)
    df.interpolate(method='linear', inplace=True)
    print("\nMissing Values After Handling:")
    print(df.isnull().sum())

    # Convert 'Date' to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Scale numerical columns
    cols_to_scale = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
    scaler = MinMaxScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    # Save preprocessed data
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")

    # Plot histograms and box plots
    df[cols_to_scale].hist(bins=30, figsize=(12, 8))
    plt.suptitle(f"Histograms of Numerical Columns in {file_path}")
    plt.show()
    
    df[cols_to_scale].plot(kind='box', subplots=True, layout=(2, 3), figsize=(12, 8), title="Box Plots of Numerical Columns")
    plt.suptitle(f"Box Plots of Numerical Columns in {file_path}")
    plt.show()


def plot_eda(data_dict):
    """Visualize exploratory data analysis results."""
    # EDA: Visualize the Closing Price over Time
    plt.figure(figsize=(14, 6))
    for asset, df in data_dict.items():
        plt.plot(df.index, df['Close'], label=f'{asset} Close')
    plt.title("Closing Prices Over Time")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.show()

    # EDA 2: Calculate and Plot the Daily Percentage Change to Observe Volatility
    plt.figure(figsize=(14, 6))
    for asset, df in data_dict.items():
        df['Daily Return'] = df['Close'].pct_change()
        plt.plot(df.index, df['Daily Return'], label=f'{asset} Daily Return')
    plt.title("Daily Percentage Change (Volatility)")
    plt.xlabel("Date")
    plt.ylabel("Daily Return")
    plt.legend()
    plt.show()


def decompose_time_series(data_dict):
    """Decompose time series data into trend, seasonality, and residuals."""
    fig, axes = plt.subplots(len(data_dict), 4, figsize=(16, 12), sharex=True)
    fig.suptitle("Time Series Decomposition", fontsize=16)

    for i, (asset, df) in enumerate(data_dict.items()):
        # Decompose the time series
        decomposition = seasonal_decompose(df['Close'], model='multiplicative', period=252)
        
        # Plot decomposition
        axes[i, 0].set_title(f"{asset} Observed")
        decomposition.observed.plot(ax=axes[i, 0], legend=False)
        axes[i, 1].set_title(f"{asset} Trend")
        decomposition.trend.plot(ax=axes[i, 1], legend=False)
        axes[i, 2].set_title(f"{asset} Seasonal")
        decomposition.seasonal.plot(ax=axes[i, 2], legend=False)
        axes[i, 3].set_title(f"{asset} Residual")
        decomposition.resid.plot(ax=axes[i, 3], legend=False)

    plt.tight_layout()
    plt.show()


def risk_analysis(data_dict):
    """Perform risk analysis on the stock data."""
    risk_metrics = {}

    for asset, df in data_dict.items():
        df['Daily Return'] = df['Close'].pct_change()

        # Volatility (Standard Deviation of Daily Returns)
        volatility = df['Daily Return'].std() * np.sqrt(252)  # Annualized volatility
        # Value at Risk (VaR) - 1 day 95% confidence
        var_95 = df['Daily Return'].quantile(0.05)
        # Sharpe Ratio
        risk_free_rate = 0.02  # Example risk-free rate (2% per year)
        sharpe_ratio = (df['Daily Return'].mean() - risk_free_rate / 252) / df['Daily Return'].std() * np.sqrt(252)

        risk_metrics[asset] = {
            'Volatility': volatility,
            '1-Day VaR (95%)': var_95,
            'Sharpe Ratio': sharpe_ratio
        }
    
    # Display the risk metrics
    risk_metrics_df = pd.DataFrame(risk_metrics).T
    print("\nRisk Analysis Metrics:")
    print(risk_metrics_df)


def load_and_prepare_data(file_paths):
    """Load cleaned data, preprocess, and set index for time series analysis."""
    data_dict = {}
    for asset, file_path in file_paths.items():
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        data_dict[asset] = df
    return data_dict


# --- Main Execution ---
if __name__ == "__main__":
    # Define the assets and date range
    assets = ["TSLA", "BND", "SPY"]
    start_date = "2015-01-01"
    end_date = "2024-10-31"
    
    # Download and save data
    download_data(assets, start_date, end_date)
    
    # Define the file paths for cleaning
    file_paths = {
        'TSLA': '../data/TSLA_data.csv',
        'BND': '../data/BND_data.csv',
        'SPY': '../data/SPY_data.csv'
    }
    columns_order = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
    
    # Clean and save data
    clean_data(file_paths, columns_order)

    # Define preprocessed output paths
    output_paths = {
        'TSLA': '../data/preprocessed_TSLA_data.csv',
        'BND': '../data/preprocessed_BND_data.csv',
        'SPY': '../data/preprocessed_SPY_data.csv'
    }

    # Preprocess data and perform EDA
    for asset, input_file in file_paths.items():
        preprocess_and_explore_data(input_file, output_paths[asset])

    # Load the preprocessed data
    data_dict = load_and_prepare_data(output_paths)
    
    # Plot EDA results
    plot_eda(data_dict)
    
    # Decompose the time series
    decompose_time_series(data_dict)

    # Perform risk analysis
    risk_analysis(data_dict)

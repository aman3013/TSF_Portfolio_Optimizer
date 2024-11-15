import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from scipy.stats import norm

# Set page configuration
st.set_page_config(page_title="Time Series Forecasting for Portfolio Management Optimization", 
                   page_icon="📈", layout="wide")

# Title
st.title("Time Series Forecasting for Portfolio Management Optimization")

# Introduction and Objective
st.header("Introduction")
st.write("""
This dashboard leverages time series forecasting techniques to assist in optimizing portfolio management. 
By analyzing historical and forecasted data for assets such as Tesla (TSLA), Vanguard Total Bond Market ETF (BND), and SPDR S&P 500 ETF Trust (SPY), 
the tool provides insights into asset performance, risk, and diversification strategies.
""")

st.header("Objective")
st.write("""
The primary goal of this application is to:
1. Provide actionable insights into portfolio performance metrics, including returns, volatility, and Sharpe ratio.
2. Allow users to simulate portfolio allocations interactively.
3. Visualize the efficient frontier and help identify optimal portfolio strategies.
4. Analyze asset correlations and forecasted price movements to assist in decision-making.
""")

# Helper function to load data
@st.cache_data
def load_data(file_path):
    """Safely load a CSV file into a Pandas DataFrame."""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading {file_path}: {str(e)}")
        return None

# Load forecasted data
tsla_forecast = load_data('../data/forecasted_tsla_renamed.csv')
bnd_forecast = load_data('../data/forecasted_bnd_renamed.csv')
spy_forecast = load_data('../data/forecasted_spy_renamed.csv')

# Stop if any dataset fails to load
if tsla_forecast is None or bnd_forecast is None or spy_forecast is None:
    st.stop()

# Identify the price column (Forecast is the only relevant column)
def get_price_column(df, label):
    """Directly return the Forecast column."""
    if 'Forecast' in df.columns:
        return 'Forecast'
    else:
        st.error(f"'Forecast' column not found in {label} data.")
        st.stop()

# Extract price columns
tsla_price_col = get_price_column(tsla_forecast, 'TSLA')
bnd_price_col = get_price_column(bnd_forecast, 'BND')
spy_price_col = get_price_column(spy_forecast, 'SPY')

# Ensure 'Date' column is in datetime format
for df, label in zip([tsla_forecast, bnd_forecast, spy_forecast], ['TSLA', 'BND', 'SPY']):
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    else:
        st.error(f"'Date' column not found in {label} data.")
        st.stop()

# Merge dataframes on 'Date'
combined_forecast = tsla_forecast[['Date', tsla_price_col]].merge(
    bnd_forecast[['Date', bnd_price_col]], on='Date'
).merge(spy_forecast[['Date', spy_price_col]], on='Date')
combined_forecast.columns = ['Date', 'TSLA', 'BND', 'SPY']

# Calculate daily returns
returns = combined_forecast[['TSLA', 'BND', 'SPY']].pct_change()

# Portfolio metrics function
def calculate_portfolio_metrics(weights):
    """Calculate portfolio return, volatility, Sharpe ratio, and VaR."""
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe_ratio = portfolio_return / portfolio_volatility
    portfolio_returns = np.sum(returns * weights, axis=1)
    var_95 = norm.ppf(0.05, portfolio_returns.mean(), portfolio_returns.std())
    return portfolio_return, portfolio_volatility, sharpe_ratio, var_95

# Sidebar for user input
st.sidebar.header("Portfolio Weights")
weight_tsla = st.sidebar.slider("TSLA Weight", 0.0, 1.0, 0.33, 0.01)
weight_bnd = st.sidebar.slider("BND Weight", 0.0, 1.0, 0.33, 0.01)
weight_spy = st.sidebar.slider("SPY Weight", 0.0, 1.0, 0.34, 0.01)

# Normalize weights
weights = np.array([weight_tsla, weight_bnd, weight_spy])
weights /= weights.sum()

# Portfolio metrics
portfolio_return, portfolio_volatility, sharpe_ratio, var_95 = calculate_portfolio_metrics(weights)

# Portfolio metrics display
st.title("Portfolio Performance Dashboard")
st.header("Portfolio Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Annual Return", f"{portfolio_return:.2%}")
col2.metric("Annual Volatility", f"{portfolio_volatility:.2%}")
col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
col4.metric("VaR (95%)", f"{var_95:.2%}")

# Forecasted prices
st.header("Forecasted Asset Prices")
fig = go.Figure()
for asset in ['TSLA', 'BND', 'SPY']:
    fig.add_trace(go.Scatter(x=combined_forecast['Date'], y=combined_forecast[asset], name=asset))
fig.update_layout(title="Forecasted Asset Prices", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig)

# Cumulative returns
st.header("Cumulative Returns")
cumulative_returns = (1 + returns).cumprod()
fig = go.Figure()
for asset in cumulative_returns.columns:
    fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns[asset], name=asset))
fig.update_layout(title="Cumulative Returns", xaxis_title="Date", yaxis_title="Cumulative Return")
st.plotly_chart(fig)

# Correlation heatmap
st.header("Asset Correlation")
corr_matrix = returns.corr()
fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale="RdBu_r")
fig.update_layout(title="Correlation Heatmap")
st.plotly_chart(fig)

# Rolling volatility
st.header("Rolling Volatility")
window = st.slider("Rolling Window (days)", 5, 100, 20)
rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
fig = go.Figure()
for asset in rolling_vol.columns:
    fig.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol[asset], name=asset))
fig.update_layout(title=f"{window}-Day Rolling Volatility", xaxis_title="Date", yaxis_title="Annualized Volatility")
st.plotly_chart(fig)

# Efficient frontier
st.header("Efficient Frontier")
def efficient_frontier(returns, num_portfolios=1000):
    results = np.zeros((3, num_portfolios))  # Store volatility, return, Sharpe ratio
    weights_record = []
    for i in range(num_portfolios):  # Use 'i' as the loop index
        weights = np.random.random(3)
        weights /= np.sum(weights)
        portfolio_return, portfolio_volatility, sharpe_ratio, var_95 = calculate_portfolio_metrics(weights)
        results[0, i] = portfolio_volatility
        results[1, i] = portfolio_return
        results[2, i] = sharpe_ratio
        weights_record.append(weights)
    return results, weights_record

results, weights_record = efficient_frontier(returns)
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=results[0, :], y=results[1, :],
    mode='markers',
    marker=dict(color=results[2, :], colorscale='Viridis', showscale=True),
    name='Efficient Frontier'
))
fig.add_trace(go.Scatter(
    x=[portfolio_volatility], y=[portfolio_return],
    mode='markers', marker=dict(color='red', size=15),
    name='Current Portfolio'
))
fig.update_layout(title="Efficient Frontier", xaxis_title="Volatility", yaxis_title="Return")
st.plotly_chart(fig)

# Asset allocation pie chart
st.header("Current Asset Allocation")
fig = go.Figure(data=[go.Pie(labels=['TSLA', 'BND', 'SPY'], values=weights)])
fig.update_layout(title="Asset Allocation")
st.plotly_chart(fig)

# Download button
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(combined_forecast)
st.download_button("Download Forecast Data", data=csv, file_name="forecast_data.csv")

# Feedback section
st.header("Feedback")
feedback = st.text_area("Please provide your feedback:")
if st.button("Submit Feedback"):
    st.success("Thank you for your feedback!")
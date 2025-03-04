"""
Dashboard - Interactive visualization for the natural gas trading system.

This module implements a Streamlit-based dashboard that provides:
1. Data exploration and visualization
2. Strategy monitoring and performance metrics
3. Portfolio position tracking
4. Risk analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys

# Add project root to path
project_root = str(Path(__file__).parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

# Import modules from the trading system
from src.utils.logger import get_logger
from src.data_ingestion.cme_fetcher import fetch_futures_data
from src.data_ingestion.noaa_fetcher import fetch_noaa_data
from src.data_processing.feature_engineering import calculate_technical_indicators
from src.strategies.alpha_model import create_default_models
from src.strategies.risk_model import create_default_risk_model

# Configure logging
logger = get_logger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Natural Gas Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define paths
DATA_DIR = Path(project_root) / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INTERIM_DATA_DIR = DATA_DIR / "interim"

# Cache data loading to improve performance
@st.cache_data(ttl=3600)
def load_futures_data(symbol='NG=F', start_date=None, end_date=None):
    """Load futures data for display."""
    try:
        # Try to load from processed data first
        processed_file = PROCESSED_DATA_DIR / f"{symbol.replace('=', '_')}_daily.csv"
        if processed_file.exists():
            data = pd.read_csv(processed_file, index_col=0, parse_dates=True)
            if start_date:
                data = data[data.index >= start_date]
            if end_date:
                data = data[data.index <= end_date]
            return data
        
        # Otherwise fetch from source
        else:
            # Default to last 2 years if no dates provided
            if not start_date:
                start_date = (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
                
            data = fetch_futures_data(ticker=symbol, start_date=start_date, end_date=end_date)
            return data
            
    except Exception as e:
        logger.error(f"Error loading futures data: {e}")
        # Return dummy data if real data not available
        dates = pd.date_range(start='2020-01-01', end='2023-01-01', freq='D')
        return pd.DataFrame({
            'open': np.linspace(2.0, 3.0, len(dates)) + np.random.normal(0, 0.1, len(dates)),
            'high': np.linspace(2.1, 3.1, len(dates)) + np.random.normal(0, 0.1, len(dates)),
            'low': np.linspace(1.9, 2.9, len(dates)) + np.random.normal(0, 0.1, len(dates)),
            'close': np.linspace(2.0, 3.0, len(dates)) + np.random.normal(0, 0.1, len(dates)),
            'volume': np.random.randint(100000, 500000, len(dates))
        }, index=dates)

@st.cache_data(ttl=3600)
def load_weather_data(station_id='USW00094728', start_date=None, end_date=None):
    """Load weather data for display."""
    try:
        # Try to load from processed data first
        processed_file = PROCESSED_DATA_DIR / f"weather_{station_id}.csv"
        if processed_file.exists():
            data = pd.read_csv(processed_file, index_col=0, parse_dates=True)
            if start_date:
                data = data[data.index >= start_date]
            if end_date:
                data = data[data.index <= end_date]
            return data
        
        # Otherwise fetch from source
        else:
            # Default to last 2 years if no dates provided
            if not start_date:
                start_date = (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            try:
                # Try to fetch data using NOAA API
                data = fetch_noaa_data(
                    location_id=station_id,
                    start_date=start_date,
                    end_date=end_date,
                    data_types=['TMAX', 'TMIN', 'TAVG', 'PRCP']
                )
                
                # If successful, rename columns to match expected format
                if not data.empty:
                    if 'TAVG' in data.columns:
                        data['temperature'] = data['TAVG']
                    if 'PRCP' in data.columns:
                        data['precipitation'] = data['PRCP']
                    return data
            except Exception as e:
                logger.warning(f"Could not fetch NOAA data: {e}. Using synthetic data instead.")
            
            # If fetching fails, generate synthetic data
            logger.info("Generating synthetic weather data for dashboard demo")
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            # Create seasonal temperature pattern
            temps = 60 + 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + np.random.normal(0, 5, len(dates))
            return pd.DataFrame({
                'temperature': temps,
                'precipitation': np.random.exponential(0.1, len(dates)),
                'wind_speed': np.random.normal(8, 3, len(dates))
            }, index=dates)
            
    except Exception as e:
        logger.error(f"Error loading weather data: {e}")
        # Return dummy data if real data not available
        dates = pd.date_range(start='2020-01-01', end='2023-01-01', freq='D')
        return pd.DataFrame({
            'temperature': 60 + 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + np.random.normal(0, 5, len(dates)),
            'precipitation': np.random.exponential(0.1, len(dates)),
            'wind_speed': np.random.normal(8, 3, len(dates))
        }, index=dates)

@st.cache_data(ttl=3600)
def load_portfolio_data():
    """Load portfolio position and performance data."""
    try:
        # Try to load from interim data
        position_file = INTERIM_DATA_DIR / "target_positions.csv"
        if position_file.exists():
            return pd.read_csv(position_file, index_col=0, parse_dates=True)
        else:
            # Generate example portfolio data if file doesn't exist
            dates = pd.date_range(start='2022-01-01', end='2023-01-01', freq='D')
            ng_pos = 0.5 + 0.5 * np.sin(2 * np.pi * np.arange(len(dates)) / 180)
            cl_pos = -0.3 + 0.3 * np.cos(2 * np.pi * np.arange(len(dates)) / 90)
            
            return pd.DataFrame({
                'NG_target': ng_pos,
                'CL_target': cl_pos
            }, index=dates)
    except Exception as e:
        logger.error(f"Error loading portfolio data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_backtest_results():
    """Load backtest performance results."""
    try:
        # Try to load from interim data
        backtest_file = INTERIM_DATA_DIR / "backtest_results.csv"
        if backtest_file.exists():
            return pd.read_csv(backtest_file, index_col=0, parse_dates=True)
        else:
            # Generate example backtest data if file doesn't exist
            dates = pd.date_range(start='2022-01-01', end='2023-01-01', freq='D')
            
            # Create cumulative return series with some realistic properties
            daily_returns = np.random.normal(0.0005, 0.01, len(dates))
            daily_returns[::7] = daily_returns[::7] - 0.01  # Add some regular drawdowns
            cum_returns = (1 + daily_returns).cumprod()
            
            # Add some strategy-specific metrics
            return pd.DataFrame({
                'strategy_return': daily_returns,
                'cumulative_return': cum_returns,
                'benchmark_return': (1 + np.random.normal(0.0003, 0.012, len(dates))).cumprod(),
                'position_size': 0.5 + 0.5 * np.sin(2 * np.pi * np.arange(len(dates)) / 180),
                'volatility': np.random.normal(0.1, 0.02, len(dates)),
                'drawdown': -np.random.exponential(0.02, len(dates))
            }, index=dates)
    except Exception as e:
        logger.error(f"Error loading backtest results: {e}")
        return pd.DataFrame()

def create_price_chart(data):
    """Create an interactive price chart with volume."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, row_heights=[0.7, 0.3])
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        name='Price'
    ), row=1, col=1)
    
    # Add volume chart
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['volume'],
        name='Volume',
        marker_color='rgba(0, 0, 255, 0.5)'
    ), row=2, col=1)
    
    # Add moving averages if they exist in the data
    if 'MA_20' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MA_20'],
            name='20-day MA',
            line=dict(color='blue')
        ), row=1, col=1)
    
    if 'MA_50' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MA_50'],
            name='50-day MA',
            line=dict(color='red')
        ), row=1, col=1)
    
    # Update layout
    fig.update_layout(
        title='Natural Gas Futures Price',
        yaxis_title='Price',
        yaxis2_title='Volume',
        xaxis_rangeslider_visible=False,
        height=600,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def create_performance_chart(data):
    """Create performance metrics visualization."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.7, 0.3])
    
    # Add cumulative return
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['cumulative_return'],
        name='Strategy Returns',
        line=dict(color='blue')
    ), row=1, col=1)
    
    if 'benchmark_return' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['benchmark_return'],
            name='Benchmark Returns',
            line=dict(color='grey', dash='dash')
        ), row=1, col=1)
    
    # Add drawdown chart
    if 'drawdown' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['drawdown'],
            name='Drawdown',
            line=dict(color='red'),
            fill='tozeroy'
        ), row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title='Strategy Performance',
        yaxis_title='Cumulative Return',
        yaxis2_title='Drawdown',
        height=500,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def create_position_chart(data):
    """Create position size visualization."""
    fig = go.Figure()
    
    # Add each asset position
    for col in data.columns:
        if 'target' in col.lower():
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[col],
                name=col.replace('_target', ''),
                line=dict(width=2)
            ))
    
    # Update layout
    fig.update_layout(
        title='Portfolio Positions',
        yaxis_title='Position Size',
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def calculate_performance_metrics(data):
    """Calculate key performance metrics."""
    if 'strategy_return' not in data.columns or len(data) < 2:
        return {}
    
    # Calculate basic metrics
    returns = data['strategy_return']
    cum_returns = data['cumulative_return']
    
    metrics = {
        'Total Return': f"{(cum_returns.iloc[-1] - 1) * 100:.2f}%",
        'Annualized Return': f"{(((cum_returns.iloc[-1]) ** (252/len(returns)) - 1) * 100):.2f}%",
        'Volatility': f"{returns.std() * np.sqrt(252) * 100:.2f}%",
        'Sharpe Ratio': f"{(returns.mean() / returns.std()) * np.sqrt(252):.2f}",
        'Max Drawdown': f"{data.get('drawdown', pd.Series(0)).min() * 100:.2f}%",
        'Win Rate': f"{(returns > 0).sum() / len(returns) * 100:.2f}%"
    }
    
    if 'benchmark_return' in data.columns:
        benchmark_return = data['benchmark_return'].pct_change().dropna()
        metrics['Alpha'] = f"{((returns.mean() - benchmark_return.mean()) * 252) * 100:.2f}%"
        
        # Calculate beta if we have sufficient data
        if len(returns) > 5 and len(benchmark_return) > 5:
            cov = np.cov(returns, benchmark_return)[0, 1]
            var = np.var(benchmark_return)
            beta = cov / var if var != 0 else 0
            metrics['Beta'] = f"{beta:.2f}"
    
    return metrics

def main():
    """Main function to run the dashboard."""
    # Sidebar for navigation and controls
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Data Explorer", "Strategy Monitor", "Portfolio Tracker", "Risk Analysis"]
    )
    
    st.sidebar.title("Settings")
    
    # Date range selector
    start_date = st.sidebar.date_input(
        "Start Date",
        datetime.now() - timedelta(days=365)
    )
    end_date = st.sidebar.date_input(
        "End Date",
        datetime.now()
    )
    
    # Convert to string format for the data loaders
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    # Load data
    futures_data = load_futures_data(start_date=start_date_str, end_date=end_date_str)
    weather_data = load_weather_data(start_date=start_date_str, end_date=end_date_str)
    portfolio_data = load_portfolio_data()
    backtest_data = load_backtest_results()
    
    # Filter data by date range
    futures_data = futures_data[(futures_data.index >= start_date_str) & (futures_data.index <= end_date_str)]
    weather_data = weather_data[(weather_data.index >= start_date_str) & (weather_data.index <= end_date_str)]
    
    if len(portfolio_data) > 0:
        portfolio_data = portfolio_data[(portfolio_data.index >= start_date_str) & (portfolio_data.index <= end_date_str)]
    
    if len(backtest_data) > 0:
        backtest_data = backtest_data[(backtest_data.index >= start_date_str) & (backtest_data.index <= end_date_str)]
    
    # Add technical indicators if needed
    if 'MA_20' not in futures_data.columns and 'close' in futures_data.columns:
        futures_data = calculate_technical_indicators(futures_data)
    
    # Data Explorer Page
    if page == "Data Explorer":
        st.title("Natural Gas Trading - Data Explorer")
        
        tab1, tab2 = st.tabs(["Price Data", "Weather Data"])
        
        with tab1:
            st.subheader("Natural Gas Futures Prices")
            st.plotly_chart(create_price_chart(futures_data), use_container_width=True)
            
            st.subheader("Price Data Table")
            # Apply formatting only to numeric columns
            formatter = {
                col: "{:.2f}" for col in futures_data.select_dtypes(include=['float', 'int']).columns
            }
            st.dataframe(futures_data.tail(10).style.format(formatter))
            
            # Download link for the data
            csv = futures_data.to_csv().encode('utf-8')
            st.download_button(
                label="Download Price Data as CSV",
                data=csv,
                file_name='ng_futures_data.csv',
                mime='text/csv',
            )
        
        with tab2:
            st.subheader("Weather Data")
            
            # Plot temperature data
            if 'temperature' in weather_data.columns:
                fig = px.line(weather_data, x=weather_data.index, y='temperature', title='Temperature Over Time')
                st.plotly_chart(fig, use_container_width=True)
            
            # Plot additional weather metrics if available
            weather_cols = [col for col in weather_data.columns if col != 'temperature']
            if weather_cols:
                fig = px.line(weather_data, x=weather_data.index, y=weather_cols, title='Weather Metrics')
                st.plotly_chart(fig, use_container_width=True)
            
            # Display data table
            st.subheader("Weather Data Table")
            weather_formatter = {
                col: "{:.2f}" for col in weather_data.select_dtypes(include=['float', 'int']).columns
            }
            st.dataframe(weather_data.tail(10).style.format(weather_formatter))
            
            # Download link for the data
            csv = weather_data.to_csv().encode('utf-8')
            st.download_button(
                label="Download Weather Data as CSV",
                data=csv,
                file_name='weather_data.csv',
                mime='text/csv',
            )
    
    # Strategy Monitor Page
    elif page == "Strategy Monitor":
        st.title("Natural Gas Trading - Strategy Monitor")
        
        if len(backtest_data) > 0:
            # Display performance chart
            st.subheader("Strategy Performance")
            st.plotly_chart(create_performance_chart(backtest_data), use_container_width=True)
            
            # Display key metrics
            st.subheader("Performance Metrics")
            metrics = calculate_performance_metrics(backtest_data)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Return", metrics.get('Total Return', 'N/A'))
                st.metric("Win Rate", metrics.get('Win Rate', 'N/A'))
            with col2:
                st.metric("Annualized Return", metrics.get('Annualized Return', 'N/A'))
                st.metric("Alpha", metrics.get('Alpha', 'N/A'))
            with col3:
                st.metric("Sharpe Ratio", metrics.get('Sharpe Ratio', 'N/A'))
                st.metric("Beta", metrics.get('Beta', 'N/A'))
            with col4:
                st.metric("Volatility", metrics.get('Volatility', 'N/A'))
                st.metric("Max Drawdown", metrics.get('Max Drawdown', 'N/A'))
            
            # Display recent signals if available
            if 'signal' in backtest_data.columns:
                st.subheader("Recent Trading Signals")
                signal_data = backtest_data[['signal']].tail(10)
                st.dataframe(signal_data)
        else:
            st.info("No strategy data available. Please run a backtest first.")
            
            # Add a button to simulate running a backtest
            if st.button("Run Sample Backtest"):
                st.session_state['run_backtest'] = True
                st.rerun()
    
    # Portfolio Tracker Page
    elif page == "Portfolio Tracker":
        st.title("Natural Gas Trading - Portfolio Tracker")
        
        if len(portfolio_data) > 0:
            # Display position chart
            st.plotly_chart(create_position_chart(portfolio_data), use_container_width=True)
            
            # Display current positions
            st.subheader("Current Positions")
            current_positions = portfolio_data.iloc[-1].to_dict()
            
            position_df = pd.DataFrame({
                'Asset': [key.replace('_target', '') for key in current_positions.keys()],
                'Position': list(current_positions.values())
            })
            
            st.dataframe(position_df.style.format({"Position": "{:.2f}"}))
            
            # Add some position metrics
            total_exposure = abs(position_df['Position']).sum()
            net_exposure = position_df['Position'].sum()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Exposure", f"{total_exposure:.2f}")
            with col2:
                st.metric("Net Exposure", f"{net_exposure:.2f}")
            with col3:
                st.metric("# of Assets", f"{len(position_df)}")
            
            # Display position history
            st.subheader("Position History")
            portfolio_formatter = {
                col: "{:.4f}" for col in portfolio_data.select_dtypes(include=['float', 'int']).columns
            }
            st.dataframe(portfolio_data.tail(10).style.format(portfolio_formatter))
            
            # Download link for position data
            csv = portfolio_data.to_csv().encode('utf-8')
            st.download_button(
                label="Download Position Data as CSV",
                data=csv,
                file_name='portfolio_positions.csv',
                mime='text/csv',
            )
        else:
            st.info("No portfolio data available.")
    
    # Risk Analysis Page
    elif page == "Risk Analysis":
        st.title("Natural Gas Trading - Risk Analysis")
        
        if len(backtest_data) > 0 and len(portfolio_data) > 0:
            # Display drawdown chart
            if 'drawdown' in backtest_data.columns:
                st.subheader("Drawdown Analysis")
                fig = px.area(backtest_data, x=backtest_data.index, y='drawdown', 
                             title='Historical Drawdowns', color_discrete_sequence=['red'])
                fig.update_layout(yaxis_title='Drawdown')
                st.plotly_chart(fig, use_container_width=True)
            
            # Display volatility chart
            if 'volatility' in backtest_data.columns:
                st.subheader("Volatility Analysis")
                fig = px.line(backtest_data, x=backtest_data.index, y='volatility',
                             title='Historical Volatility', color_discrete_sequence=['orange'])
                fig.update_layout(yaxis_title='Volatility')
                st.plotly_chart(fig, use_container_width=True)
            
            # Display risk metrics
            st.subheader("Risk Metrics")
            
            # Calculate risk metrics
            if 'strategy_return' in backtest_data.columns:
                returns = backtest_data['strategy_return']
                vol = returns.std() * np.sqrt(252)
                var_95 = returns.quantile(0.05) * np.sqrt(1)
                var_99 = returns.quantile(0.01) * np.sqrt(1)
                
                cvar_95 = returns[returns <= var_95].mean()
                cvar_99 = returns[returns <= var_99].mean()
                
                max_dd = backtest_data.get('drawdown', pd.Series(0)).min()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Annualized Volatility", f"{vol*100:.2f}%")
                    st.metric("Max Drawdown", f"{max_dd*100:.2f}%")
                with col2:
                    st.metric("Value at Risk (95%)", f"{var_95*100:.2f}%")
                    st.metric("Expected Shortfall (95%)", f"{cvar_95*100:.2f}%")
                with col3:
                    st.metric("Value at Risk (99%)", f"{var_99*100:.2f}%")
                    st.metric("Expected Shortfall (99%)", f"{cvar_99*100:.2f}%")
            
            # Display position concentration
            st.subheader("Position Concentration")
            latest_positions = portfolio_data.iloc[-1]
            
            # Create a pie chart of positions
            fig = px.pie(
                values=latest_positions.abs(),
                names=latest_positions.index,
                title='Position Concentration (Absolute Values)',
                hole=0.3,
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display correlation matrix if multiple assets
            if len(latest_positions) > 1:
                st.subheader("Asset Correlation Matrix")
                
                # Try to calculate correlations between assets if possible
                if len(portfolio_data.columns) > 1:
                    corr_matrix = portfolio_data.corr()
                    fig = px.imshow(
                        corr_matrix,
                        color_continuous_scale='RdBu_r',
                        title='Correlation Matrix',
                        text_auto='.2f'
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No risk data available. Please run a backtest first.")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        **Natural Gas Trading System**  
        Version: 1.0.0  
        © 2023
        """
    )

if __name__ == "__main__":
    main() 
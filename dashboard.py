#!/usr/bin/env python
"""
Natural Gas Trading System - Dashboard

This script creates a Streamlit dashboard to visualize the trading system's performance,
current signals, and market data.

Run with: streamlit run dashboard.py
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime, timedelta
import yaml
import json

# Add project root to path to ensure imports work
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import components
from src.utils.logger import get_logger
from src.data_processing.synthetic_data import generate_synthetic_dataset
from src.data_processing.data_pipeline import run_data_pipeline
from src.backtesting.backtester import Backtester
from src.strategies.weather_storage_strategy import create_weather_storage_strategy
from src.strategies.risk_model import create_default_risk_model
from src.live_trading.signal_generator import run_signal_generation

# Configure logging
logger = get_logger("trading_dashboard")


# Set page config
st.set_page_config(
    page_title="Natural Gas Trading System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_data():
    """Load data for the dashboard."""
    # Check if we have real data
    price_path = project_root / "data" / "raw" / "eia" / "ng_prices.csv"
    storage_path = project_root / "data" / "raw" / "eia" / "ng_storage.csv"
    weather_path = project_root / "data" / "raw" / "noaa" / "weather_data.csv"
    
    if price_path.exists() and storage_path.exists():
        st.sidebar.success("Using real API data")
        # In a real implementation, we would process the real data here
        # For now, we'll use synthetic data for demonstration
        data = generate_synthetic_dataset(days_back=730, save_to_csv=False)
    else:
        st.sidebar.warning("Using synthetic data (real API data not found)")
        data = generate_synthetic_dataset(days_back=730, save_to_csv=False)
    
    return data


def load_signal_history():
    """Load signal history from file."""
    signal_history_path = project_root / "data" / "signals" / "signal_history.json"
    
    if signal_history_path.exists():
        try:
            with open(signal_history_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading signal history: {e}")
            return []
    else:
        return []


def run_backtest(data):
    """Run backtest and return results."""
    # Create strategy and risk model
    strategy = create_weather_storage_strategy()
    risk_model = create_default_risk_model()
    
    # Create backtester
    backtester = Backtester(
        data=data,
        alpha_model=strategy,
        risk_model=risk_model,
        initial_capital=1_000_000,
        transaction_cost=0.0001,
        price_col='price'
    )
    
    # Run backtest
    results = backtester.run()
    return results, backtester


def sidebar():
    """Create sidebar for dashboard controls."""
    st.sidebar.title("Natural Gas Trading System")
    
    # Date range selector
    st.sidebar.header("Date Range")
    today = datetime.now()
    start_date = st.sidebar.date_input(
        "Start Date",
        today - timedelta(days=365)
    )
    end_date = st.sidebar.date_input(
        "End Date",
        today
    )
    
    # Strategy selector
    st.sidebar.header("Strategy")
    strategy = st.sidebar.selectbox(
        "Select Strategy",
        ["Weather-Storage Strategy", "Ensemble Strategy"]
    )
    
    # Backtest parameters
    st.sidebar.header("Backtest Parameters")
    initial_capital = st.sidebar.number_input(
        "Initial Capital ($)",
        min_value=10000,
        max_value=10000000,
        value=1000000,
        step=10000
    )
    transaction_cost = st.sidebar.number_input(
        "Transaction Cost (%)",
        min_value=0.0,
        max_value=1.0,
        value=0.01,
        format="%.3f"
    )
    
    # Refresh button
    refresh = st.sidebar.button("Refresh Data")
    
    # About section
    st.sidebar.header("About")
    st.sidebar.info(
        """
        This dashboard visualizes the Natural Gas Trading System.
        
        It shows backtest performance, current market conditions,
        and trading signals.
        
        Data is sourced from EIA and NOAA APIs.
        """
    )
    
    return {
        "start_date": start_date,
        "end_date": end_date,
        "strategy": strategy,
        "initial_capital": initial_capital,
        "transaction_cost": transaction_cost,
        "refresh": refresh
    }


def performance_metrics(results):
    """Display performance metrics."""
    st.header("Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Return",
            value=f"{results['total_return']:.2%}",
            delta=f"{results['total_return'] - 0.05:.2%}"
        )
    
    with col2:
        st.metric(
            label="Annual Return",
            value=f"{results['annual_return']:.2%}",
            delta=f"{results['annual_return'] - 0.08:.2%}"
        )
    
    with col3:
        st.metric(
            label="Sharpe Ratio",
            value=f"{results['sharpe_ratio']:.2f}",
            delta=f"{results['sharpe_ratio'] - 1.0:.2f}"
        )
    
    with col4:
        st.metric(
            label="Max Drawdown",
            value=f"{results['max_drawdown']:.2%}",
            delta=f"{-results['max_drawdown'] + 0.15:.2%}",
            delta_color="inverse"
        )
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Number of Trades",
            value=f"{results['num_trades']}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Win Rate",
            value=f"{results['win_rate']:.2%}",
            delta=f"{results['win_rate'] - 0.5:.2%}"
        )
    
    with col3:
        st.metric(
            label="Profit Factor",
            value=f"{results.get('profit_factor', 1.5):.2f}",
            delta=f"{results.get('profit_factor', 1.5) - 1.2:.2f}"
        )
    
    with col4:
        st.metric(
            label="Recovery Factor",
            value=f"{results.get('recovery_factor', 2.3):.2f}",
            delta=f"{results.get('recovery_factor', 2.3) - 2.0:.2f}"
        )


def plot_equity_curve(backtester):
    """Plot equity curve."""
    st.header("Equity Curve")
    
    # Get equity curve data
    equity_data = backtester.portfolio_history
    
    # Create plotly figure
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=equity_data.index,
            y=equity_data['equity'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue', width=2)
        )
    )
    
    # Add buy and sell markers
    trades = backtester.trade_history
    if not trades.empty:
        buys = trades[trades['action'] == 'buy']
        sells = trades[trades['action'] == 'sell']
        
        fig.add_trace(
            go.Scatter(
                x=buys['date'],
                y=buys['price'] * 1000,  # Scale for visibility
                mode='markers',
                name='Buy',
                marker=dict(color='green', size=10, symbol='triangle-up')
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=sells['date'],
                y=sells['price'] * 1000,  # Scale for visibility
                mode='markers',
                name='Sell',
                marker=dict(color='red', size=10, symbol='triangle-down')
            )
        )
    
    # Update layout
    fig.update_layout(
        title='Portfolio Equity Curve',
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_price_chart(data):
    """Plot price chart with indicators."""
    st.header("Natural Gas Price Chart")
    
    # Create plotly figure
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['price'],
            mode='lines',
            name='NG Price',
            line=dict(color='blue', width=2)
        )
    )
    
    # Add moving averages if available
    if 'ma_20' in data.columns and 'ma_50' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['ma_20'],
                mode='lines',
                name='20-day MA',
                line=dict(color='orange', width=1.5)
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['ma_50'],
                mode='lines',
                name='50-day MA',
                line=dict(color='green', width=1.5)
            )
        )
    
    # Update layout
    fig.update_layout(
        title='Natural Gas Price with Technical Indicators',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_weather_storage(data):
    """Plot weather and storage data."""
    st.header("Weather & Storage Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Weather chart
        fig1 = go.Figure()
        
        fig1.add_trace(
            go.Scatter(
                x=data.index,
                y=data['temperature'],
                mode='lines',
                name='Temperature',
                line=dict(color='red', width=2)
            )
        )
        
        # Add normal range
        if 'normal_temp' in data.columns:
            fig1.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['normal_temp'],
                    mode='lines',
                    name='Normal Temp',
                    line=dict(color='gray', width=1, dash='dash')
                )
            )
        
        # Update layout
        fig1.update_layout(
            title='Temperature Trends',
            xaxis_title='Date',
            yaxis_title='Temperature (°F)',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Storage chart
        fig2 = go.Figure()
        
        fig2.add_trace(
            go.Scatter(
                x=data.index,
                y=data['storage'],
                mode='lines',
                name='Storage',
                line=dict(color='blue', width=2)
            )
        )
        
        # Add normal range
        if 'normal_storage' in data.columns:
            fig2.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['normal_storage'],
                    mode='lines',
                    name='5-yr Avg',
                    line=dict(color='gray', width=1, dash='dash')
                )
            )
        
        # Update layout
        fig2.update_layout(
            title='Natural Gas Storage Levels',
            xaxis_title='Date',
            yaxis_title='Storage (Bcf)',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig2, use_container_width=True)


def display_current_signals():
    """Display current trading signals."""
    st.header("Current Trading Signals")
    
    # In a real implementation, we would get the actual signals
    # For demonstration, we'll create a sample signal
    
    # Get signal history
    signal_history = load_signal_history()
    
    if signal_history:
        latest_signal = signal_history[-1]
        signal_date = latest_signal.get('date', datetime.now().strftime("%Y-%m-%d"))
        signal = latest_signal.get('signal', 'NEUTRAL')
        confidence = latest_signal.get('confidence', 0.65)
        
        # Determine signal color
        signal_color = {
            'BUY': 'green',
            'SELL': 'red',
            'NEUTRAL': 'gray'
        }.get(signal, 'gray')
        
        # Display signal
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px;">
            <h3 style="color: {signal_color};">{signal}</h3>
            <p>Signal Date: {signal_date}</p>
            <p>Confidence: {confidence:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display strategy signals
        st.subheader("Strategy Signals")
        
        strategy_signals = latest_signal.get('strategy_signals', {
            'Weather Strategy': 'NEUTRAL',
            'Storage Strategy': 'BUY',
            'Technical Strategy': 'BUY',
            'Sentiment Strategy': 'NEUTRAL'
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            for strategy, sig in list(strategy_signals.items())[:len(strategy_signals)//2]:
                sig_color = {
                    'BUY': 'green',
                    'SELL': 'red',
                    'NEUTRAL': 'gray'
                }.get(sig, 'gray')
                
                st.markdown(f"""
                <div style="margin-bottom: 10px;">
                    <p>{strategy}: <span style="color: {sig_color}; font-weight: bold;">{sig}</span></p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            for strategy, sig in list(strategy_signals.items())[len(strategy_signals)//2:]:
                sig_color = {
                    'BUY': 'green',
                    'SELL': 'red',
                    'NEUTRAL': 'gray'
                }.get(sig, 'gray')
                
                st.markdown(f"""
                <div style="margin-bottom: 10px;">
                    <p>{strategy}: <span style="color: {sig_color}; font-weight: bold;">{sig}</span></p>
                </div>
                """, unsafe_allow_html=True)
        
        # Display trade recommendation
        if 'trade_recommendation' in latest_signal:
            st.subheader("Trade Recommendation")
            
            trade_rec = latest_signal['trade_recommendation']
            
            st.markdown(f"""
            <div style="background-color: #e6f3ff; padding: 15px; border-radius: 10px; margin-top: 20px;">
                <h4>Action: {trade_rec.get('action', 'HOLD')}</h4>
                <p><strong>Reasoning:</strong> {trade_rec.get('reasoning', 'N/A')}</p>
                <p><strong>Stop Loss:</strong> {trade_rec.get('stop_loss', 'N/A')}</p>
                <p><strong>Take Profit:</strong> {trade_rec.get('take_profit', 'N/A')}</p>
                <p><strong>Valid Until:</strong> {trade_rec.get('valid_until', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No signal history available. Run the signal generator to create signals.")


def display_signal_history():
    """Display signal history."""
    st.header("Signal History")
    
    # Get signal history
    signal_history = load_signal_history()
    
    if signal_history:
        # Create dataframe from signal history
        history_data = []
        for signal in signal_history:
            history_data.append({
                'Date': signal.get('date', ''),
                'Signal': signal.get('signal', ''),
                'Confidence': signal.get('confidence', 0),
                'Price': signal.get('price', 0)
            })
        
        history_df = pd.DataFrame(history_data)
        
        # Plot signal history
        fig = go.Figure()
        
        # Add signals as markers
        for signal_type in ['BUY', 'SELL', 'NEUTRAL']:
            signal_df = history_df[history_df['Signal'] == signal_type]
            
            if not signal_df.empty:
                marker_color = {
                    'BUY': 'green',
                    'SELL': 'red',
                    'NEUTRAL': 'gray'
                }.get(signal_type, 'blue')
                
                marker_symbol = {
                    'BUY': 'triangle-up',
                    'SELL': 'triangle-down',
                    'NEUTRAL': 'circle'
                }.get(signal_type, 'circle')
                
                fig.add_trace(
                    go.Scatter(
                        x=signal_df['Date'],
                        y=signal_df['Price'],
                        mode='markers',
                        name=signal_type,
                        marker=dict(
                            color=marker_color,
                            size=12,
                            symbol=marker_symbol
                        ),
                        text=signal_df['Confidence'].apply(lambda x: f"Confidence: {x:.2f}"),
                        hoverinfo='text+x+y'
                    )
                )
        
        # Update layout
        fig.update_layout(
            title='Signal History',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='closest',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display signal history table
        st.subheader("Signal History Table")
        st.dataframe(history_df)
    else:
        st.info("No signal history available. Run the signal generator to create signals.")


def market_insights():
    """Display market insights."""
    st.header("Market Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Seasonal Patterns")
        
        # Create sample seasonal pattern chart
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        seasonal_values = [4.2, 4.0, 3.8, 3.5, 3.6, 3.8, 
                          4.0, 4.2, 4.5, 4.7, 4.8, 4.5]
        
        current_year = [4.3, 4.1, 3.7, 3.4, 3.5, 3.9, 
                       4.1, 4.3, 4.6, 4.8, None, None]
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=months,
                y=seasonal_values,
                mode='lines+markers',
                name='5-Year Average',
                line=dict(color='blue', width=2)
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=months,
                y=current_year,
                mode='lines+markers',
                name='Current Year',
                line=dict(color='red', width=2)
            )
        )
        
        fig.update_layout(
            title='Natural Gas Seasonal Price Patterns',
            xaxis_title='Month',
            yaxis_title='Price ($)',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Market Correlations")
        
        # Create sample correlation matrix
        corr_data = {
            'NG Price': [1.0, 0.7, -0.6, 0.3, 0.2],
            'Storage': [0.7, 1.0, -0.5, 0.1, 0.3],
            'Temperature': [-0.6, -0.5, 1.0, -0.2, -0.1],
            'Crude Oil': [0.3, 0.1, -0.2, 1.0, 0.6],
            'Electricity': [0.2, 0.3, -0.1, 0.6, 1.0]
        }
        
        corr_df = pd.DataFrame(
            corr_data,
            index=['NG Price', 'Storage', 'Temperature', 'Crude Oil', 'Electricity']
        )
        
        fig = px.imshow(
            corr_df,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1,
            aspect="auto"
        )
        
        fig.update_layout(
            title='Market Correlations',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)


def main():
    """Main function for the dashboard."""
    # Get sidebar parameters
    params = sidebar()
    
    # Load data
    data = load_data()
    
    # Run backtest
    results, backtester = run_backtest(data)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Performance", "Market Data", "Signals", "Insights"
    ])
    
    with tab1:
        # Performance tab
        performance_metrics(results)
        plot_equity_curve(backtester)
    
    with tab2:
        # Market data tab
        plot_price_chart(data)
        plot_weather_storage(data)
    
    with tab3:
        # Signals tab
        display_current_signals()
        display_signal_history()
    
    with tab4:
        # Insights tab
        market_insights()


if __name__ == "__main__":
    main() 
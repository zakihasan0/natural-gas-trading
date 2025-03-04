#!/usr/bin/env python
"""
Natural Gas Trading System - Example Run

This script demonstrates the full workflow of the natural gas trading system,
from data fetching to signal generation and backtest visualization.

It runs in demonstration mode, which can use either synthetic data or real API data.
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta

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
logger = get_logger("trading_system_example")


def run_with_synthetic_data(args):
    """
    Run the trading system with synthetic data.
    
    Args:
        args: Command line arguments
    """
    logger.info("Starting example run with synthetic data")
    
    # Generate synthetic data
    logger.info("Generating synthetic data...")
    data = generate_synthetic_dataset(days_back=730, save_to_csv=True)
    
    # Run backtest
    run_backtest(data, args)
    
    # Generate latest signal
    run_latest_signal(data, args)


def run_with_real_data(args):
    """
    Run the trading system with real API data.
    
    Args:
        args: Command line arguments
    """
    logger.info("Starting example run with real API data")
    
    # Fetch data using data pipeline
    logger.info("Fetching real data from APIs...")
    data = run_data_pipeline(days_back=730)
    
    if data.empty:
        logger.error("Failed to fetch real data. Please check API credentials.")
        logger.info("Falling back to synthetic data...")
        data = generate_synthetic_dataset(days_back=730, save_to_csv=True)
    
    # Run backtest
    run_backtest(data, args)
    
    # Generate latest signal
    run_latest_signal(data, args)


def run_backtest(data, args):
    """
    Run a backtest on the provided data.
    
    Args:
        data: DataFrame with historical data
        args: Command line arguments
    """
    logger.info("Running backtest...")
    
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
    
    # Print performance summary
    print("\nBacktest Performance Summary:")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Annual Return: {results['annual_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Number of Trades: {results['num_trades']}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    
    # Plot results
    if args.plot:
        backtester.plot_results()


def run_latest_signal(data, args):
    """
    Generate the latest trading signal.
    
    Args:
        data: DataFrame with historical data
        args: Command line arguments
    """
    logger.info("Generating latest trading signal...")
    
    # Run signal generation
    signal_info = run_signal_generation(save_history=True)
    
    # Print signal information
    print("\nLatest Signal Information:")
    print(f"Signal: {signal_info['signal']} (confidence: {signal_info['confidence']:.2f})")
    print(f"Latest data date: {signal_info['latest_data_date']}")
    print("\nStrategy Signals:")
    for strategy, signal in signal_info["strategy_signals"].items():
        print(f"  {strategy}: {signal}")
    
    # Print trade recommendation
    if "trade_recommendation" in signal_info:
        trade_rec = signal_info["trade_recommendation"]
        print("\nTrade Recommendation:")
        print(f"Action: {trade_rec['action']}")
        print(f"Reasoning: {trade_rec['reasoning']}")
        if trade_rec['stop_loss']:
            print(f"Stop Loss: {trade_rec['stop_loss']}")
        if trade_rec['take_profit']:
            print(f"Take Profit: {trade_rec['take_profit']}")
        print(f"Valid Until: {trade_rec['valid_until']}")


def main():
    """Main entry point for the example."""
    parser = argparse.ArgumentParser(description='Natural Gas Trading System Example Run')
    
    # Example arguments
    parser.add_argument('--data', choices=['synthetic', 'real'], default='synthetic',
                       help='Data source to use')
    parser.add_argument('--plot', action='store_true',
                       help='Plot backtest results')
    
    args = parser.parse_args()
    
    if args.data == 'synthetic':
        run_with_synthetic_data(args)
    else:
        run_with_real_data(args)
    
    logger.info("Example run completed")


if __name__ == "__main__":
    main() 
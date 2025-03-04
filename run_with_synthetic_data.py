#!/usr/bin/env python
"""
Run Trading System with Synthetic Data

This script demonstrates how to run the natural gas trading system
using synthetic data instead of real API data. This is useful for
testing the system when API access is not available.

Usage:
    python run_with_synthetic_data.py --strategy weather --years 2
"""

import argparse
import sys
import os
import pandas as pd
import logging
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
from src.backtesting.backtester import Backtester, MultiStrategyBacktester
from src.strategies.alpha_model import create_default_models
from src.strategies.risk_model import create_default_risk_model
from src.strategies.weather_storage_strategy import create_weather_storage_strategy
from src.data_processing.feature_engineering import calculate_technical_indicators

# Configure logging
logger = get_logger("synthetic_trading_system")


def process_synthetic_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Process synthetic data to create all necessary features.
    
    Args:
        data: Raw synthetic data
        
    Returns:
        Processed data with all features
    """
    logger.info(f"Processing synthetic data with {len(data)} records")
    
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Calculate technical indicators if not already present
    if 'rsi_14' not in df.columns and 'price' in df.columns:
        tech_df = calculate_technical_indicators(df, price_col='price')
        for col in tech_df.columns:
            if col not in df.columns:
                df[col] = tech_df[col]
    
    # Make sure we have weather deviation features
    if 'weather_TAVG' in df.columns and 'weather_deviation' not in df.columns:
        # Calculate seasonal averages based on day of year
        df['dayofyear'] = df.index.dayofyear
        avg_temp = df.groupby('dayofyear')['weather_TAVG'].transform('mean')
        df['weather_deviation'] = df['weather_TAVG'] - avg_temp
        df.drop(columns=['dayofyear'], inplace=True)
    
    # Make sure we have storage deviation features
    if 'storage' in df.columns and 'storage_deviation_pct' not in df.columns:
        # Calculate seasonal averages based on week of year
        df['weekofyear'] = df.index.isocalendar().week
        avg_storage = df.groupby('weekofyear')['storage'].transform('mean')
        df['storage_deviation'] = df['storage'] - avg_storage
        df['storage_deviation_pct'] = (df['storage_deviation'] / avg_storage) * 100
        df.drop(columns=['weekofyear'], inplace=True)
    
    # Drop any rows with missing values
    df.dropna(inplace=True)
    
    logger.info(f"Processed data: {len(df)} records with {len(df.columns)} features")
    return df


def run_backtest_with_synthetic_data(args):
    """
    Run backtest with synthetic data.
    
    Args:
        args: Command line arguments
    """
    logger.info(f"Starting backtest with synthetic data (years: {args.years}, strategy: {args.strategy})")
    
    # Generate synthetic data
    days = int(args.years * 365)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    # Check if synthetic data already exists
    synthetic_data_path = Path(project_root) / 'data' / 'synthetic' / 'synthetic_data.csv'
    
    if synthetic_data_path.exists() and not args.regenerate:
        logger.info(f"Loading existing synthetic data from {synthetic_data_path}")
        data = pd.read_csv(synthetic_data_path, index_col=0, parse_dates=True)
        
        # Filter to desired date range
        data = data[(data.index >= start_date) & (data.index <= end_date)]
        
        logger.info(f"Loaded synthetic data with {len(data)} records from {data.index[0]} to {data.index[-1]}")
    else:
        logger.info(f"Generating new synthetic data from {start_date} to {end_date}")
        data = generate_synthetic_dataset(start_date=start_date, end_date=end_date, save_to_csv=True)
    
    # Process the data
    processed_data = process_synthetic_data(data)
    
    if processed_data.empty:
        logger.error("No data available for backtesting")
        return
    
    if args.strategy == 'weather':
        # Run weather strategy backtest
        strategy = create_weather_storage_strategy()
        risk_model = create_default_risk_model()
        
        backtester = Backtester(
            data=processed_data,
            alpha_model=strategy,
            risk_model=risk_model,
            initial_capital=args.capital,
            transaction_cost=args.cost,
            price_col='price'
        )
        
        results = backtester.run()
        backtester.plot_results()
        
        # Print summary
        print(f"\nWeather-Storage Strategy Performance Summary:")
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Annual Return: {results['annual_return']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"Number of Trades: {results['num_trades']}")
        print(f"Win Rate: {results['win_rate']:.2%}")
        
    elif args.strategy == 'ensemble':
        # Run ensemble backtest with multiple strategies
        strategies = {}
        
        # Add default strategies
        default_models = create_default_models()
        default_risk = create_default_risk_model()
        
        for name, model in default_models.items():
            strategies[name] = (model, default_risk)
        
        # Add weather strategy
        weather_strategy = create_weather_storage_strategy()
        strategies['WeatherStorage'] = (weather_strategy, default_risk)
        
        # Run multi-strategy backtest
        multi_backtester = MultiStrategyBacktester(
            data=processed_data,
            strategies=strategies,
            initial_capital=args.capital,
            transaction_cost=args.cost
        )
        
        results = multi_backtester.run()
        
        # Plot results
        multi_backtester.plot_equity_curves()
        
        # Compare metrics
        comparison = multi_backtester.compare_metrics()
        print("\nStrategy Comparison:")
        print(comparison)
        
    elif args.strategy == 'all':
        # Run individual backtests for all strategies
        strategies = {}
        
        # Add default strategies
        default_models = create_default_models()
        default_risk = create_default_risk_model()
        
        for name, model in default_models.items():
            strategies[name] = (model, default_risk)
        
        # Add weather strategy
        weather_strategy = create_weather_storage_strategy()
        strategies['WeatherStorage'] = (weather_strategy, default_risk)
        
        # Run backtest for each strategy
        for name, (strategy, risk_model) in strategies.items():
            logger.info(f"Running backtest for strategy: {name}")
            
            backtester = Backtester(
                data=processed_data,
                alpha_model=strategy,
                risk_model=risk_model,
                initial_capital=args.capital,
                transaction_cost=args.cost,
                price_col='price'
            )
            
            results = backtester.run()
            
            # Print summary
            print(f"\n{name} Strategy Performance Summary:")
            print(f"Total Return: {results['total_return']:.2%}")
            print(f"Annual Return: {results['annual_return']:.2%}")
            print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {results['max_drawdown']:.2%}")
            print(f"Number of Trades: {results['num_trades']}")
            print(f"Win Rate: {results['win_rate']:.2%}")
            
            # Plot results if requested
            if args.plot:
                backtester.plot_results()
    
    else:
        logger.error(f"Unknown strategy: {args.strategy}")
        return
    
    logger.info("Backtest with synthetic data completed")


def main():
    """Main entry point for the synthetic data trading system."""
    parser = argparse.ArgumentParser(description='Natural Gas Trading System with Synthetic Data')
    
    # Backtest arguments
    parser.add_argument('--years', type=float, default=2,
                       help='Years of synthetic data for backtesting')
    parser.add_argument('--strategy', choices=['weather', 'ensemble', 'all'], default='weather',
                       help='Strategy to backtest')
    parser.add_argument('--capital', type=float, default=1000000,
                       help='Initial capital for backtesting')
    parser.add_argument('--cost', type=float, default=0.0001,
                       help='Transaction cost as fraction of trade value')
    parser.add_argument('--plot', action='store_true',
                       help='Plot backtest results for individual strategies')
    parser.add_argument('--regenerate', action='store_true',
                       help='Regenerate synthetic data even if it already exists')
    
    args = parser.parse_args()
    
    # Run the backtest with synthetic data
    run_backtest_with_synthetic_data(args)


if __name__ == "__main__":
    main() 
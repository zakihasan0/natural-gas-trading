#!/usr/bin/env python
"""
Natural Gas Trading System - Main Runner Script

This script serves as the entry point for running the trading system,
with options for backtesting or live trading.

Usage:
    python run_trading_system.py --mode backtest --years 2
    python run_trading_system.py --mode live --interval 60
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
from src.data_processing.data_pipeline import run_data_pipeline
from src.backtesting.backtester import Backtester, MultiStrategyBacktester
from src.strategies.alpha_model import create_default_models
from src.strategies.risk_model import create_default_risk_model
from src.strategies.weather_storage_strategy import create_weather_storage_strategy
from src.live_trading.signal_generator import start_signal_service, run_signal_generation

# Configure logging
logger = get_logger("trading_system")


def run_backtest(args):
    """
    Run backtest mode.
    
    Args:
        args: Command line arguments
    """
    logger.info(f"Starting backtest mode (years: {args.years}, strategy: {args.strategy})")
    
    # Get historical data
    days = int(args.years * 365)
    data = run_data_pipeline(days_back=days)
    
    if data.empty:
        logger.error("No data available for backtesting")
        sys.exit(1)
    
    logger.info(f"Loaded {len(data)} data points for backtesting")
    
    if args.strategy == 'weather':
        # Run weather strategy backtest
        strategy = create_weather_storage_strategy()
        risk_model = create_default_risk_model()
        
        backtester = Backtester(
            data=data,
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
            data=data,
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
                data=data,
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
        sys.exit(1)
    
    logger.info("Backtest completed")


def run_live(args):
    """
    Run live trading mode.
    
    Args:
        args: Command line arguments
    """
    logger.info(f"Starting live trading mode (interval: {args.interval} minutes)")
    
    if args.once:
        # Run signal generation once
        signal_info = run_signal_generation()
        
        # Print signal information
        print("\nSignal Information:")
        print(f"Signal: {signal_info['signal']} (confidence: {signal_info['confidence']:.2f})")
        print(f"Latest data date: {signal_info['latest_data_date']}")
        print("\nStrategy Signals:")
        for strategy, signal in signal_info["strategy_signals"].items():
            print(f"  {strategy}: {signal}")
        
        # Print trade recommendation if available
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
    else:
        # Start signal service with specified interval
        start_signal_service(interval_minutes=args.interval)


def main():
    """Main entry point for the trading system."""
    parser = argparse.ArgumentParser(description='Natural Gas Trading System')
    parser.add_argument('--mode', choices=['backtest', 'live'], required=True,
                       help='Trading mode: backtest or live')
    
    # Backtest arguments
    parser.add_argument('--years', type=float, default=2,
                       help='Years of historical data for backtesting')
    parser.add_argument('--strategy', choices=['weather', 'ensemble', 'all'], default='weather',
                       help='Strategy to backtest')
    parser.add_argument('--capital', type=float, default=1000000,
                       help='Initial capital for backtesting')
    parser.add_argument('--cost', type=float, default=0.0001,
                       help='Transaction cost as fraction of trade value')
    parser.add_argument('--plot', action='store_true',
                       help='Plot backtest results for individual strategies')
    
    # Live trading arguments
    parser.add_argument('--interval', type=int, default=60,
                       help='Interval in minutes between signal generations')
    parser.add_argument('--once', action='store_true',
                       help='Run signal generation once instead of continuously')
    
    args = parser.parse_args()
    
    # Run the selected mode
    if args.mode == 'backtest':
        run_backtest(args)
    elif args.mode == 'live':
        run_live(args)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
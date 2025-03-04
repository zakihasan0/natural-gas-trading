#!/usr/bin/env python
"""
Scheduler for Trading System

This script schedules the trading system to run at regular intervals,
such as daily signal generation and execution.

Usage:
    python schedule_trading_system.py --interval daily --time 17:00
"""

import argparse
import sys
import os
import time
import schedule
import logging
from pathlib import Path
from datetime import datetime, timedelta
import subprocess

# Add project root to path to ensure imports work
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import utility logger
from src.utils.logger import get_logger

# Configure logging
logger = get_logger("trading_system_scheduler")


def run_trading_system(mode='live', args=None):
    """
    Run the trading system with specified mode and arguments.
    
    Args:
        mode: Trading mode ('live' or 'backtest')
        args: Additional command-line arguments
    """
    if args is None:
        args = []
    
    # Build command
    cmd = [sys.executable, str(project_root / 'run_trading_system.py'), f'--mode={mode}']
    cmd.extend(args)
    
    # Log command
    logger.info(f"Running trading system: {' '.join(cmd)}")
    
    try:
        # Run command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Log output
        if result.stdout:
            for line in result.stdout.splitlines():
                logger.info(f"[Output] {line}")
        
        # Log errors
        if result.stderr:
            for line in result.stderr.splitlines():
                logger.error(f"[Error] {line}")
        
        # Log status
        if result.returncode == 0:
            logger.info("Trading system run completed successfully")
        else:
            logger.error(f"Trading system run failed with code {result.returncode}")
    
    except Exception as e:
        logger.error(f"Error running trading system: {e}")


def run_live_signal_generation():
    """Run live signal generation."""
    logger.info("Running scheduled live signal generation")
    run_trading_system(mode='live', args=['--once'])


def run_daily_backtest_update():
    """Run daily backtest update."""
    logger.info("Running scheduled backtest update")
    run_trading_system(mode='backtest', args=['--strategy=ensemble', '--years=1'])


def is_market_open():
    """
    Check if the natural gas market is currently open.
    
    Returns:
        bool: True if market is open, False otherwise
    """
    # Get current time in US Eastern Time (market time)
    # This is a simplified version - a real implementation would handle holidays
    now = datetime.now()  # Ideally should be converted to Eastern Time
    
    # Check if it's a weekday (0 = Monday, 4 = Friday)
    if now.weekday() > 4:
        return False
    
    # Check if it's between 9:00 and 14:30 Eastern Time
    # This is a simplification - actual natural gas futures trading hours may differ
    hour = now.hour
    minute = now.minute
    
    market_open = hour >= 9 and (hour < 14 or (hour == 14 and minute <= 30))
    
    return market_open


def schedule_tasks(args):
    """
    Schedule trading system tasks based on command line arguments.
    
    Args:
        args: Command line arguments
    """
    # Log initialization
    logger.info(f"Initializing trading system scheduler (interval: {args.interval})")
    
    if args.interval == 'daily':
        # Schedule once daily at specified time
        logger.info(f"Scheduling daily signal generation at {args.time}")
        schedule.every().day.at(args.time).do(run_live_signal_generation)
        
        # Schedule daily backtest update (if enabled)
        if args.backtest_update:
            logger.info(f"Scheduling daily backtest update at {args.backtest_time}")
            schedule.every().day.at(args.backtest_time).do(run_daily_backtest_update)
    
    elif args.interval == 'hourly':
        # Schedule hourly
        logger.info(f"Scheduling hourly signal generation")
        schedule.every().hour.do(run_live_signal_generation)
    
    elif args.interval == 'market':
        # Schedule during market hours
        logger.info(f"Scheduling signal generation every {args.market_interval} minutes during market hours")
        
        def market_hour_job():
            if is_market_open():
                run_live_signal_generation()
            else:
                logger.info("Market closed, skipping signal generation")
        
        # Schedule during market hours
        schedule.every(args.market_interval).minutes.do(market_hour_job)
    
    # Run once at startup if requested
    if args.run_at_start:
        logger.info("Running signal generation at startup")
        run_live_signal_generation()
    
    # Infinite loop to run scheduled tasks
    logger.info("Starting scheduler loop")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
    except Exception as e:
        logger.error(f"Error in scheduler: {e}")


def main():
    """Main entry point for the trading system scheduler."""
    parser = argparse.ArgumentParser(description='Natural Gas Trading System Scheduler')
    
    # Scheduler arguments
    parser.add_argument('--interval', choices=['daily', 'hourly', 'market'], default='daily',
                       help='Scheduling interval')
    parser.add_argument('--time', type=str, default='17:00',
                       help='Time for daily runs (HH:MM)')
    parser.add_argument('--market-interval', type=int, default=60,
                       help='Interval in minutes for market-hour runs')
    parser.add_argument('--run-at-start', action='store_true',
                       help='Run the system once at startup')
    parser.add_argument('--backtest-update', action='store_true',
                       help='Run daily backtest update')
    parser.add_argument('--backtest-time', type=str, default='23:00',
                       help='Time for daily backtest update (HH:MM)')
    
    args = parser.parse_args()
    
    # Start scheduler
    schedule_tasks(args)


if __name__ == "__main__":
    main() 
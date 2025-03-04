"""
Signal Generator - Process data and generate live trading signals.

This module processes recent market data and generates trading signals
using the configured strategy models. It runs on a schedule and can
be used to automate trading decisions or provide signal alerts.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import os
import logging
import json
from datetime import datetime, timedelta
import time
import yaml

# Import utility logger
from src.utils.logger import get_logger

# Import pipeline components
from src.data_processing.data_pipeline import run_data_pipeline
from src.strategies.alpha_model import create_default_models
from src.strategies.risk_model import create_default_risk_model
from src.strategies.weather_storage_strategy import create_weather_storage_strategy

# Configure logging
logger = get_logger(__name__)


def load_config() -> Dict:
    """
    Load configuration from the config file.
    
    Returns:
        Dict: Configuration dictionary
    """
    try:
        config_path = Path(__file__).parents[2] / 'config' / 'config.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}


def get_latest_data(days_back: int = 120) -> pd.DataFrame:
    """
    Get the latest data for signal generation.
    
    Args:
        days_back: Number of days of historical data to include
        
    Returns:
        DataFrame with processed market data
    """
    logger.info(f"Fetching latest data for signal generation (last {days_back} days)")
    
    try:
        # Use the data pipeline to get processed data
        data = run_data_pipeline(days_back=days_back)
        logger.info(f"Retrieved {len(data)} rows of latest data")
        return data
    except Exception as e:
        logger.error(f"Error getting latest data: {e}")
        return pd.DataFrame()


def generate_signals(
    data: pd.DataFrame,
    strategy_weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Generate trading signals from the latest data.
    
    Args:
        data: DataFrame with processed market data
        strategy_weights: Dict mapping strategy names to weights
        
    Returns:
        Dict with signal information
    """
    if data.empty:
        logger.error("Cannot generate signals: No data provided")
        return {"signal": 0, "confidence": 0, "timestamp": datetime.now().isoformat()}
    
    # Get latest date in the data
    latest_date = data.index[-1]
    logger.info(f"Generating signals for {latest_date}")
    
    # Load strategies
    strategies = {}
    
    # Add default strategies from alpha model
    default_strategies = create_default_models()
    strategies.update(default_strategies)
    
    # Add specialized strategies
    strategies["WeatherStorage"] = create_weather_storage_strategy()
    
    # Default weights if not provided
    if strategy_weights is None:
        strategy_weights = {
            "Momentum": 0.2,
            "MeanReversion": 0.2,
            "Fundamental": 0.3,
            "WeatherStorage": 0.3
        }
    
    # Normalize weights to sum to 1
    total_weight = sum(strategy_weights.values())
    if total_weight != 1.0:
        strategy_weights = {k: v / total_weight for k, v in strategy_weights.items()}
    
    # Generate signals for each strategy
    signals = {}
    for name, strategy in strategies.items():
        if name in strategy_weights:
            try:
                # Generate signal for the strategy
                signal = strategy.generate_signals(data)
                
                # Get the latest signal
                latest_signal = signal.iloc[-1] if not signal.empty else 0
                
                # Store the signal
                signals[name] = latest_signal
                logger.info(f"Strategy {name}: Signal = {latest_signal}")
            except Exception as e:
                logger.error(f"Error generating signal for {name}: {e}")
                signals[name] = 0
    
    # Calculate weighted signal
    weighted_signal = 0
    for name, weight in strategy_weights.items():
        if name in signals:
            weighted_signal += signals[name] * weight
    
    # Calculate signal confidence as absolute value of weighted signal
    confidence = abs(weighted_signal)
    
    # Determine final signal (-1, 0, 1)
    if weighted_signal > 0.3:
        final_signal = 1
    elif weighted_signal < -0.3:
        final_signal = -1
    else:
        final_signal = 0
    
    # Create results dictionary
    signal_info = {
        "signal": final_signal,
        "raw_signal": weighted_signal,
        "confidence": confidence,
        "strategy_signals": signals,
        "strategy_weights": strategy_weights,
        "timestamp": datetime.now().isoformat(),
        "latest_data_date": latest_date.isoformat()
    }
    
    logger.info(f"Final signal: {final_signal} (confidence: {confidence:.2f})")
    return signal_info


def generate_trade_recommendation(signal_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a trading recommendation based on the signal.
    
    Args:
        signal_info: Dict with signal information
        
    Returns:
        Dict with trading recommendation
    """
    trade_rec = {
        "timestamp": datetime.now().isoformat(),
        "signal": signal_info["signal"],
        "confidence": signal_info["confidence"],
        "action": "HOLD",
        "reasoning": "",
        "stop_loss": None,
        "take_profit": None
    }
    
    # Determine action based on signal
    if signal_info["signal"] == 1:
        trade_rec["action"] = "BUY"
        trade_rec["reasoning"] = "Strong bullish signal based on weather and storage fundamentals."
        
        # Set stop loss and take profit levels (placeholder - would need current price data)
        trade_rec["stop_loss"] = "2% below entry"
        trade_rec["take_profit"] = "5% above entry"
    
    elif signal_info["signal"] == -1:
        trade_rec["action"] = "SELL"
        trade_rec["reasoning"] = "Strong bearish signal based on weather and storage fundamentals."
        
        # Set stop loss and take profit levels (placeholder - would need current price data)
        trade_rec["stop_loss"] = "2% above entry"
        trade_rec["take_profit"] = "5% below entry"
    
    else:
        trade_rec["action"] = "HOLD"
        trade_rec["reasoning"] = "No strong signal detected. Remain neutral."
    
    # Add time validity
    trade_rec["valid_until"] = (datetime.now() + timedelta(days=1)).isoformat()
    
    logger.info(f"Trade recommendation: {trade_rec['action']} (confidence: {trade_rec['confidence']:.2f})")
    return trade_rec


def save_signal_history(signal_info: Dict[str, Any]) -> None:
    """
    Save signal history to a file for tracking.
    
    Args:
        signal_info: Dict with signal information
    """
    try:
        # Create directory if it doesn't exist
        data_dir = Path(__file__).parents[2] / 'data' / 'signals'
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Signal history file path
        signal_file = data_dir / 'signal_history.csv'
        
        # Prepare data for the CSV
        signal_data = {
            "timestamp": datetime.now().isoformat(),
            "signal": signal_info["signal"],
            "confidence": signal_info["confidence"],
            "raw_signal": signal_info["raw_signal"],
            "latest_data_date": signal_info["latest_data_date"]
        }
        
        # Add strategy signals
        for strategy, signal in signal_info["strategy_signals"].items():
            signal_data[f"strategy_{strategy}"] = signal
        
        # Create DataFrame
        signal_df = pd.DataFrame([signal_data])
        
        # Append to file if it exists, otherwise create new file
        if os.path.exists(signal_file):
            signal_df.to_csv(signal_file, mode='a', header=False, index=False)
        else:
            signal_df.to_csv(signal_file, index=False)
        
        logger.info(f"Saved signal to history file: {signal_file}")
    except Exception as e:
        logger.error(f"Error saving signal history: {e}")


def run_signal_generation(
    save_history: bool = True,
    generate_recommendation: bool = True
) -> Dict[str, Any]:
    """
    Run the complete signal generation process.
    
    Args:
        save_history: Whether to save signal history to file
        generate_recommendation: Whether to generate a trade recommendation
        
    Returns:
        Dict with signal information and recommendation
    """
    logger.info("Starting signal generation process")
    
    # Get latest data
    data = get_latest_data()
    
    if data.empty:
        logger.error("No data available for signal generation")
        return {
            "error": "No data available",
            "timestamp": datetime.now().isoformat()
        }
    
    # Generate signals
    signal_info = generate_signals(data)
    
    # Save signal history if requested
    if save_history:
        save_signal_history(signal_info)
    
    # Generate trade recommendation if requested
    if generate_recommendation:
        trade_rec = generate_trade_recommendation(signal_info)
        signal_info["trade_recommendation"] = trade_rec
    
    logger.info("Signal generation process completed")
    return signal_info


def start_signal_service(
    interval_minutes: int = 60,
    run_once: bool = False
) -> None:
    """
    Start the signal generation service to run periodically.
    
    Args:
        interval_minutes: Time interval between signal generations in minutes
        run_once: Whether to run only once or continuously
    """
    logger.info(f"Starting signal service (interval: {interval_minutes} minutes)")
    
    try:
        while True:
            # Run signal generation
            signal_info = run_signal_generation()
            
            # Log signal
            signal = signal_info.get("signal", 0)
            confidence = signal_info.get("confidence", 0)
            logger.info(f"Signal generated: {signal} (confidence: {confidence:.2f})")
            
            # Exit if run_once is True
            if run_once:
                logger.info("Run once mode: Exiting")
                break
            
            # Sleep until next interval
            logger.info(f"Sleeping for {interval_minutes} minutes")
            time.sleep(interval_minutes * 60)
    
    except KeyboardInterrupt:
        logger.info("Signal service stopped by user")
    except Exception as e:
        logger.error(f"Error in signal service: {e}")


if __name__ == "__main__":
    # Run the signal generation once
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
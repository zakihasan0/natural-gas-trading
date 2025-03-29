"""
Weather Storage Strategy - A specialized natural gas trading strategy that combines
weather data with storage levels to generate trading signals.

This strategy looks for:
1. Temperature anomalies (colder/warmer than normal)
2. Storage deviations from seasonal norms
3. Price momentum aligned with fundamentals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import os
import logging
from datetime import datetime

# Import utility logger
from src.utils.logger import get_logger

# Import base alpha model
from src.strategies.alpha_model import AlphaModel

# Configure logging
logger = get_logger(__name__)


class WeatherStorageStrategy(AlphaModel):
    """
    Strategy that combines weather deviation and storage levels to generate trading signals.
    """
    
    def __init__(
        self,
        temp_threshold: float = 1.5,  # More sensitive to temperature changes
        storage_threshold: float = 5.0,  # More sensitive to storage changes
        lookback_period: int = 10  # Longer lookback for trend analysis
    ):
        """
        Initialize the strategy parameters.
        
        Args:
            temp_threshold: Temperature deviation threshold (in degrees F)
            storage_threshold: Storage deviation threshold (in percentage)
            lookback_period: Days to look back for signal calculation
        """
        super().__init__(name="WeatherStorage")
        self.temp_threshold = temp_threshold
        self.storage_threshold = storage_threshold
        self.lookback_period = lookback_period
        
        logger.info(f"Initialized WeatherStorageStrategy with temp_threshold={temp_threshold}, "
                  f"storage_threshold={storage_threshold}, lookback_period={lookback_period}")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on weather and storage data.
        
        Args:
            data: DataFrame with weather and storage data
            
        Returns:
            Series with trading signals (-1 to 1)
        """
        signals = pd.Series(0.0, index=data.index)
        
        try:
            # Calculate temperature deviation from seasonal norm
            if 'weather_TAVG' in data.columns:
                temp_ma = data['weather_TAVG'].rolling(window=self.lookback_period).mean()
                temp_std = data['weather_TAVG'].rolling(window=self.lookback_period).std()
                temp_zscore = (data['weather_TAVG'] - temp_ma) / temp_std.replace(0, 1)
                
                # Temperature signals
                temp_signals = np.zeros_like(signals)
                temp_signals[temp_zscore > self.temp_threshold] = -1.0  # Higher temp -> lower demand -> short
                temp_signals[temp_zscore < -self.temp_threshold] = 1.0  # Lower temp -> higher demand -> long
                
                # Scale signals by deviation magnitude
                temp_signals = temp_signals * (abs(temp_zscore) / (2 * self.temp_threshold)).clip(0, 1)
            else:
                temp_signals = np.zeros_like(signals)
            
            # Calculate storage deviation from seasonal norm
            if 'storage' in data.columns:
                storage_ma = data['storage'].rolling(window=self.lookback_period).mean()
                storage_std = data['storage'].rolling(window=self.lookback_period).std()
                storage_zscore = (data['storage'] - storage_ma) / storage_std.replace(0, 1)
                
                # Storage signals
                storage_signals = np.zeros_like(signals)
                storage_signals[storage_zscore > self.storage_threshold] = -1.0  # Higher storage -> lower prices -> short
                storage_signals[storage_zscore < -self.storage_threshold] = 1.0  # Lower storage -> higher prices -> long
                
                # Scale signals by deviation magnitude
                storage_signals = storage_signals * (abs(storage_zscore) / (2 * self.storage_threshold)).clip(0, 1)
            else:
                storage_signals = np.zeros_like(signals)
            
            # Combine signals with weights
            # Give more weight to storage signals as they are more fundamental
            signals = 0.4 * temp_signals + 0.6 * storage_signals
            
            # Add momentum component
            if 'price' in data.columns:
                # Calculate price momentum
                returns = data['price'].pct_change()
                momentum = returns.rolling(window=self.lookback_period).mean()
                momentum_signal = np.sign(momentum) * (abs(momentum) / abs(momentum).mean()).clip(0, 1)
                
                # Combine with momentum (30% weight)
                signals = 0.7 * signals + 0.3 * momentum_signal
            
            # Smooth signals
            signals = signals.rolling(window=3).mean()
            
            # Ensure signals are between -1 and 1
            signals = signals.clip(-1, 1)
            
            # Fill any NaN values with 0
            signals = signals.fillna(0)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return signals


def create_weather_storage_strategy() -> WeatherStorageStrategy:
    """
    Create and configure a weather-storage strategy.
    
    Returns:
        Configured WeatherStorageStrategy instance
    """
    return WeatherStorageStrategy(
        temp_threshold=1.5,
        storage_threshold=5.0,
        lookback_period=10
    )


if __name__ == "__main__":
    from src.data_processing.data_pipeline import run_data_pipeline
    from src.strategies.risk_model import VolatilityBasedRisk
    from src.backtesting.backtester import Backtester
    import matplotlib.pyplot as plt
    
    # Get data for the past 2 years
    data = run_data_pipeline(days_back=730)
    
    # Create strategy and risk model
    strategy = create_weather_storage_strategy()
    risk_model = VolatilityBasedRisk(
        target_volatility=0.15,
        volatility_lookback=20,
        max_position_size=1.0
    )
    
    # Run backtest
    backtester = Backtester(
        data=data,
        alpha_model=strategy,
        risk_model=risk_model,
        initial_capital=1_000_000,
        transaction_cost=0.0001,
        price_col='price'
    )
    
    results = backtester.run()
    
    # Plot results
    backtester.plot_results()
    
    # Print summary
    print(f"\nWeather-Storage Strategy Performance Summary:")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Annual Return: {results['annual_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Number of Trades: {results['num_trades']}")
    print(f"Win Rate: {results['win_rate']:.2%}") 
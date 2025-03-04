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
        temp_threshold: float = 2.0,
        storage_threshold: float = 10.0,
        lookback_period: int = 5,
        holding_period: int = 3,
        price_momentum_weight: float = 0.3,
        weather_weight: float = 0.4,
        storage_weight: float = 0.3,
        use_hdd_cdd: bool = True,
        price_col: str = 'price',
        weather_col: str = 'weather_TAVG'
    ):
        """
        Initialize the strategy parameters.
        
        Args:
            temp_threshold: Temperature deviation threshold (in degrees F)
            storage_threshold: Storage deviation threshold (in percentage)
            lookback_period: Days to look back for signal calculation
            holding_period: Days to hold the position
            price_momentum_weight: Weight for price momentum signal
            weather_weight: Weight for weather signal
            storage_weight: Weight for storage signal
            use_hdd_cdd: Whether to use heating/cooling degree days
            price_col: Column name for price data
            weather_col: Column name for temperature data
        """
        super().__init__(name="WeatherStorage")
        self.temp_threshold = temp_threshold
        self.storage_threshold = storage_threshold
        self.lookback_period = lookback_period
        self.holding_period = holding_period
        self.price_momentum_weight = price_momentum_weight
        self.weather_weight = weather_weight
        self.storage_weight = storage_weight
        self.use_hdd_cdd = use_hdd_cdd
        self.price_col = price_col
        self.weather_col = weather_col
        
        logger.info(f"Initialized WeatherStorageStrategy with temp_threshold={temp_threshold}, "
                  f"storage_threshold={storage_threshold}, lookback_period={lookback_period}")
    
    def _calculate_seasonal_norms(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate seasonal norms for temperature and storage.
        
        Args:
            df: DataFrame with historical data
            
        Returns:
            DataFrame with seasonal norms
        """
        # Add month and day columns for grouping
        temp_df = df.copy()
        temp_df['month'] = temp_df.index.month
        temp_df['day'] = temp_df.index.day
        
        # Calculate 5-year average for each day of the year
        if self.weather_col in temp_df.columns:
            weather_avg = temp_df.groupby(['month', 'day'])[self.weather_col].mean()
            temp_df = temp_df.join(weather_avg.rename('weather_avg'), on=['month', 'day'])
            temp_df['weather_deviation'] = temp_df[self.weather_col] - temp_df['weather_avg']
        
        # Calculate storage seasonal norms
        if 'storage' in temp_df.columns:
            # Add week of year for storage (reported weekly)
            temp_df['week'] = temp_df.index.isocalendar().week
            storage_avg = temp_df.groupby('week')['storage'].mean()
            temp_df = temp_df.join(storage_avg.rename('storage_avg'), on='week')
            temp_df['storage_deviation'] = temp_df['storage'] - temp_df['storage_avg']
            temp_df['storage_deviation_pct'] = (temp_df['storage_deviation'] / temp_df['storage_avg']) * 100
        
        # Drop temporary columns
        cols_to_drop = ['month', 'day', 'week']
        for col in cols_to_drop:
            if col in temp_df.columns:
                temp_df = temp_df.drop(columns=[col])
        
        return temp_df
    
    def _calculate_hdd_cdd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Heating Degree Days (HDD) and Cooling Degree Days (CDD).
        
        Args:
            df: DataFrame with temperature data
            
        Returns:
            DataFrame with HDD and CDD columns
        """
        temp_df = df.copy()
        
        # Calculate HDD and CDD if temperature column exists
        if self.weather_col in temp_df.columns:
            base_temp = 65.0  # Standard base temperature (Fahrenheit)
            temp_df['HDD'] = np.maximum(base_temp - temp_df[self.weather_col], 0)
            temp_df['CDD'] = np.maximum(temp_df[self.weather_col] - base_temp, 0)
            
            # Calculate rolling sum of HDD and CDD
            temp_df['HDD_sum'] = temp_df['HDD'].rolling(window=self.lookback_period).sum()
            temp_df['CDD_sum'] = temp_df['CDD'].rolling(window=self.lookback_period).sum()
        
        return temp_df
    
    def _generate_weather_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signal based on weather data.
        
        Args:
            df: DataFrame with weather data
            
        Returns:
            Series with weather signals
        """
        if 'weather_deviation' in df.columns:
            # Calculate rolling average of temperature deviation
            weather_ma = df['weather_deviation'].rolling(window=self.lookback_period).mean()
            
            # Generate signal based on temperature deviation
            weather_signal = pd.Series(0, index=df.index)
            weather_signal[weather_ma < -self.temp_threshold] = 1  # Colder than normal - bullish
            weather_signal[weather_ma > self.temp_threshold] = -1  # Warmer than normal - bearish
            
            return weather_signal
        
        # If using HDD/CDD
        elif self.use_hdd_cdd and 'HDD_sum' in df.columns and 'CDD_sum' in df.columns:
            # Generate signal based on HDD (winter) and CDD (summer)
            weather_signal = pd.Series(0, index=df.index)
            
            # Current month to determine season
            month = df.index.month
            
            # Winter months (November to March) - focus on HDD
            winter_mask = month.isin([11, 12, 1, 2, 3])
            hdd_ma = df['HDD_sum'].rolling(window=self.lookback_period).mean()
            hdd_std = df['HDD_sum'].rolling(window=30).std()
            
            # Summer months (June to September) - focus on CDD
            summer_mask = month.isin([6, 7, 8, 9])
            cdd_ma = df['CDD_sum'].rolling(window=self.lookback_period).mean()
            cdd_std = df['CDD_sum'].rolling(window=30).std()
            
            # Higher HDD in winter is bullish (colder = more heating demand)
            if not hdd_std.empty and hdd_std.mean() > 0:
                weather_signal[winter_mask & (hdd_ma > hdd_std)] = 1
                weather_signal[winter_mask & (hdd_ma < -hdd_std)] = -1
            
            # Higher CDD in summer is bullish (hotter = more cooling demand, electricity gen)
            if not cdd_std.empty and cdd_std.mean() > 0:
                weather_signal[summer_mask & (cdd_ma > cdd_std)] = 1
                weather_signal[summer_mask & (cdd_ma < -cdd_std)] = -1
            
            return weather_signal
        
        # Fallback if no weather features available
        return pd.Series(0, index=df.index)
    
    def _generate_storage_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signal based on storage data.
        
        Args:
            df: DataFrame with storage data
            
        Returns:
            Series with storage signals
        """
        if 'storage_deviation_pct' in df.columns:
            # Calculate rolling average of storage deviation
            storage_ma = df['storage_deviation_pct'].rolling(window=4).mean()  # 4 weeks = ~1 month
            
            # Generate signal based on storage deviation
            storage_signal = pd.Series(0, index=df.index)
            storage_signal[storage_ma < -self.storage_threshold] = 1  # Lower than normal storage - bullish
            storage_signal[storage_ma > self.storage_threshold] = -1  # Higher than normal storage - bearish
            
            return storage_signal
        
        # Fallback if no storage features available
        return pd.Series(0, index=df.index)
    
    def _generate_price_momentum_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signal based on price momentum.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Series with price momentum signals
        """
        if self.price_col in df.columns:
            # Calculate returns
            returns = df[self.price_col].pct_change()
            
            # Calculate short-term and medium-term momentum
            short_momentum = returns.rolling(window=5).mean()
            medium_momentum = returns.rolling(window=20).mean()
            
            # Generate signal based on momentum
            momentum_signal = pd.Series(0, index=df.index)
            momentum_signal[(short_momentum > 0) & (medium_momentum > 0)] = 1  # Uptrend
            momentum_signal[(short_momentum < 0) & (medium_momentum < 0)] = -1  # Downtrend
            
            return momentum_signal
        
        # Fallback if no price column available
        return pd.Series(0, index=df.index)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on combined features.
        
        Args:
            data: DataFrame with temperature, storage, and price features
            
        Returns:
            Series with signals (-1, 0, 1)
        """
        df = data.copy()
        
        # Check for required columns
        required_columns = [self.price_col]
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"Missing required column: {col}")
                return pd.Series(0, index=df.index)
        
        # Calculate seasonal norms
        df = self._calculate_seasonal_norms(df)
        
        # Calculate HDD and CDD if using them
        if self.use_hdd_cdd:
            df = self._calculate_hdd_cdd(df)
        
        # Generate individual signals
        weather_signal = self._generate_weather_signal(df)
        storage_signal = self._generate_storage_signal(df)
        price_signal = self._generate_price_momentum_signal(df)
        
        # Combine signals with weights
        combined_signal = (
            self.weather_weight * weather_signal +
            self.storage_weight * storage_signal +
            self.price_momentum_weight * price_signal
        )
        
        # Apply threshold to get final signal
        final_signal = pd.Series(0, index=df.index)
        final_signal[combined_signal > 0.2] = 1  # Bullish threshold
        final_signal[combined_signal < -0.2] = -1  # Bearish threshold
        
        # Apply holding period logic
        position = self._apply_holding_period(final_signal)
        
        return position
    
    def _apply_holding_period(self, signal: pd.Series) -> pd.Series:
        """
        Apply minimum holding period to avoid overtrading.
        
        Args:
            signal: Series with raw signals
            
        Returns:
            Series with signals adjusted for holding period
        """
        position = signal.copy()
        current_position = 0
        days_in_position = 0
        
        for i in range(len(position)):
            # Get current day's signal
            current_signal = position.iloc[i]
            
            # Check if we need to take a new position
            if current_signal != 0 and current_signal != current_position:
                if days_in_position >= self.holding_period or current_position == 0:
                    # Take new position
                    current_position = current_signal
                    days_in_position = 1
                else:
                    # Stay in current position
                    position.iloc[i] = current_position
                    days_in_position += 1
            elif current_position != 0:
                # Stay in current position
                position.iloc[i] = current_position
                days_in_position += 1
                
                # If holding period elapsed, look for exit
                if days_in_position >= self.holding_period and current_signal == 0:
                    current_position = 0
                    days_in_position = 0
        
        return position


def create_weather_storage_strategy() -> WeatherStorageStrategy:
    """
    Create and configure a weather-storage strategy.
    
    Returns:
        Configured WeatherStorageStrategy instance
    """
    return WeatherStorageStrategy(
        temp_threshold=2.5,
        storage_threshold=8.0,
        lookback_period=5,
        holding_period=3,
        price_momentum_weight=0.3,
        weather_weight=0.4,
        storage_weight=0.3,
        use_hdd_cdd=True
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
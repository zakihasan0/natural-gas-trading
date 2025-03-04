"""
Synthetic Data Generator - Creates realistic synthetic data for testing the trading system.

This module generates synthetic data for natural gas prices, weather, and storage
with realistic seasonal patterns and correlations for testing the trading system
when real API data is not available.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import os
import logging
from datetime import datetime, timedelta

# Import utility logger
from src.utils.logger import get_logger

# Configure logging
logger = get_logger(__name__)


def generate_synthetic_prices(
    start_date: str,
    end_date: str,
    base_price: float = 3.0,
    volatility: float = 0.02,
    trend: float = 0.0001,
    seasonal_factor: float = 0.5
) -> pd.DataFrame:
    """
    Generate synthetic natural gas price data with realistic patterns.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        base_price: Base price level
        volatility: Daily price volatility
        trend: Long-term trend factor
        seasonal_factor: Strength of seasonal pattern
        
    Returns:
        DataFrame with synthetic price data
    """
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date)
    
    # Initialize price series with random walk
    np.random.seed(42)  # For reproducibility
    n_days = len(date_range)
    
    # Random component (random walk)
    random_changes = np.random.normal(0, volatility, n_days)
    
    # Trend component
    trend_component = np.arange(n_days) * trend
    
    # Seasonal component (higher in winter, lower in summer)
    # Convert date to position in year (0-1)
    seasonal_pos = np.array([d.dayofyear / 366 for d in date_range])
    # Winter peak around day 15 (January 15th)
    winter_peak_pos = 15 / 366
    # Calculate distance from winter peak, considering circular nature of year
    distance_from_peak = np.minimum(
        np.abs(seasonal_pos - winter_peak_pos),
        np.abs(seasonal_pos - winter_peak_pos - 1)
    )
    # Convert to seasonal factor (higher in winter, lower in summer)
    seasonal_component = seasonal_factor * (0.5 - distance_from_peak)
    
    # Combine components
    prices = base_price * (1 + np.cumsum(random_changes) + trend_component + seasonal_component)
    
    # Ensure no negative prices
    prices = np.maximum(prices, 0.1)
    
    # Create DataFrame
    price_df = pd.DataFrame({
        'date': date_range,
        'price': prices
    })
    
    # Set date as index
    price_df = price_df.set_index('date')
    
    logger.info(f"Generated synthetic price data: {len(price_df)} records from {start_date} to {end_date}")
    return price_df


def generate_synthetic_weather(
    start_date: str,
    end_date: str,
    base_temp: float = 65.0,
    temp_range: float = 30.0,
    volatility: float = 3.0,
    location: str = "Synthetic"
) -> pd.DataFrame:
    """
    Generate synthetic weather data with realistic seasonal patterns.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        base_temp: Base temperature (Fahrenheit)
        temp_range: Annual temperature range
        volatility: Daily temperature volatility
        location: Synthetic location name
        
    Returns:
        DataFrame with synthetic weather data
    """
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date)
    
    # Initialize with seasonal pattern (sinusoidal)
    np.random.seed(43)  # Different seed from prices
    n_days = len(date_range)
    
    # Position in year (0-2π)
    year_pos = np.array([2 * np.pi * d.dayofyear / 366 for d in date_range])
    
    # Seasonal pattern (coolest in January/February, warmest in July/August)
    # Phase shift to align with Northern Hemisphere seasons
    phase_shift = np.pi  # Shift so that summer is peaks around day 200 (July)
    seasonal_component = -np.cos(year_pos + phase_shift)  # -1 to 1
    
    # Scale to desired range and add base temperature
    temps = base_temp + temp_range * seasonal_component
    
    # Add random daily fluctuations
    daily_fluctuation = np.random.normal(0, volatility, n_days)
    temps += daily_fluctuation
    
    # Calculate other weather variables
    # Temperature variations
    temp_max = temps + np.random.uniform(2, 8, n_days)
    temp_min = temps - np.random.uniform(2, 8, n_days)
    
    # Precipitation (more in warmer months for typical temperate climate)
    # Base probability affected by temperature
    precip_prob = 0.2 + 0.3 * (seasonal_component + 1) / 2
    precip = np.zeros(n_days)
    for i in range(n_days):
        if np.random.random() < precip_prob[i]:
            # Precipitation amount when it occurs
            precip[i] = np.random.exponential(0.5)
    
    # Snow (only when cold)
    snow = np.zeros(n_days)
    for i in range(n_days):
        if temps[i] < 32 and np.random.random() < 0.3:
            snow[i] = np.random.exponential(2)
    
    # Wind speed
    wind = np.random.gamma(2, 2, n_days)
    
    # Create DataFrame
    weather_df = pd.DataFrame({
        'date': date_range,
        'weather_TAVG': temps,
        'weather_TMAX': temp_max,
        'weather_TMIN': temp_min,
        'weather_PRCP': precip,
        'weather_SNOW': snow,
        'weather_AWND': wind
    })
    
    # Set date as index
    weather_df = weather_df.set_index('date')
    
    logger.info(f"Generated synthetic weather data: {len(weather_df)} records from {start_date} to {end_date}")
    return weather_df


def generate_synthetic_storage(
    start_date: str,
    end_date: str,
    base_storage: float = 2000.0,
    storage_range: float = 1500.0,
    noise_level: float = 50.0
) -> pd.DataFrame:
    """
    Generate synthetic natural gas storage data with realistic seasonal patterns.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        base_storage: Base storage level (Bcf)
        storage_range: Annual storage range
        noise_level: Random noise in storage reporting
        
    Returns:
        DataFrame with synthetic storage data
    """
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date)
    
    # Initialize with seasonal pattern (sinusoidal)
    np.random.seed(44)  # Different seed
    n_days = len(date_range)
    
    # Position in year (0-2π)
    year_pos = np.array([2 * np.pi * d.dayofyear / 366 for d in date_range])
    
    # Seasonal pattern (lowest in March, highest in November)
    # Phase shift to align with typical natural gas storage cycle
    phase_shift = 3 * np.pi / 2  # Shift for natural gas storage cycle
    seasonal_component = -np.cos(year_pos + phase_shift)  # -1 to 1
    
    # Scale to desired range and add base storage
    storage = base_storage + storage_range * seasonal_component
    
    # Add random noise
    noise = np.random.normal(0, noise_level, n_days)
    storage += noise
    
    # Ensure no negative storage
    storage = np.maximum(storage, 10)
    
    # Storage is typically reported weekly (Thursday)
    # Create mask for Thursdays
    is_thursday = [d.weekday() == 3 for d in date_range]
    
    # Create storage DataFrame with weekly values
    storage_df = pd.DataFrame({
        'date': date_range,
        'storage': np.nan
    })
    
    # Only set storage values for Thursdays
    storage_df.loc[is_thursday, 'storage'] = storage[is_thursday]
    
    # Set date as index
    storage_df = storage_df.set_index('date')
    
    # Forward fill to simulate weekly reporting
    storage_df = storage_df.ffill()
    
    logger.info(f"Generated synthetic storage data: {len(storage_df)} records from {start_date} to {end_date}")
    return storage_df


def generate_synthetic_dataset(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    days_back: int = 730,
    save_to_csv: bool = True
) -> pd.DataFrame:
    """
    Generate a complete synthetic dataset with price, weather, and storage data.
    
    Args:
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
        days_back: Number of days to look back if start_date is not specified
        save_to_csv: Whether to save the synthetic dataset to CSV
        
    Returns:
        DataFrame with combined synthetic data
    """
    # Set default dates if not specified
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    logger.info(f"Generating synthetic dataset from {start_date} to {end_date}")
    
    # Generate individual components
    price_df = generate_synthetic_prices(start_date, end_date)
    weather_df = generate_synthetic_weather(start_date, end_date)
    storage_df = generate_synthetic_storage(start_date, end_date)
    
    # Create master dataframe
    master_df = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date))
    
    # Add components
    master_df['price'] = price_df['price']
    
    for col in weather_df.columns:
        master_df[col] = weather_df[col]
    
    master_df['storage'] = storage_df['storage']
    
    # Calculate correlated features
    
    # 1. Heating and Cooling Degree Days
    if 'weather_TAVG' in master_df.columns:
        master_df['HDD'] = np.maximum(65 - master_df['weather_TAVG'], 0)  # Heating Degree Days
        master_df['CDD'] = np.maximum(master_df['weather_TAVG'] - 65, 0)  # Cooling Degree Days
    
    # 2. Add some momentum features
    if 'price' in master_df.columns:
        # Simple moving averages
        master_df['price_sma5'] = master_df['price'].rolling(window=5).mean()
        master_df['price_sma20'] = master_df['price'].rolling(window=20).mean()
        
        # Relative strength
        master_df['price_rs'] = master_df['price_sma5'] / master_df['price_sma20']
    
    # 3. Add storage change
    if 'storage' in master_df.columns:
        master_df['storage_change'] = master_df['storage'].diff(5)  # Weekly change
    
    # Drop rows with NaN values
    master_df = master_df.dropna()
    
    logger.info(f"Generated synthetic dataset with {len(master_df)} records and {len(master_df.columns)} columns")
    
    if save_to_csv:
        try:
            # Create directory if it doesn't exist
            data_dir = Path(__file__).parents[2] / 'data' / 'synthetic'
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Save to CSV
            output_path = data_dir / 'synthetic_data.csv'
            master_df.to_csv(output_path)
            logger.info(f"Saved synthetic dataset to {output_path}")
            
            # Also save individual components for reference
            price_df.to_csv(data_dir / 'synthetic_prices.csv')
            weather_df.to_csv(data_dir / 'synthetic_weather.csv')
            storage_df.to_csv(data_dir / 'synthetic_storage.csv')
        except Exception as e:
            logger.error(f"Error saving synthetic dataset: {e}")
    
    return master_df


if __name__ == "__main__":
    # Generate synthetic data for the past 2 years
    synthetic_data = generate_synthetic_dataset(days_back=730)
    print(f"Generated synthetic dataset with {len(synthetic_data)} records")
    print(f"Columns: {', '.join(synthetic_data.columns)}")
    
    # Show a sample of the data
    print("\nSample data:")
    print(synthetic_data.head()) 
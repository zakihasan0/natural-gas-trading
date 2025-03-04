"""
Data Pipeline - Fetch, process, and merge data from different sources for the natural gas trading system.

This module provides functionality to:
1. Fetch data from NOAA, EIA, and other sources
2. Clean and preprocess the data
3. Merge data from different sources into a unified dataset
4. Generate features for the trading model
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any
import yaml

# Import utility logger
from src.utils.logger import get_logger

# Import data fetchers
from src.data_ingestion.noaa_fetcher import fetch_noaa_data
from src.data_ingestion.eia_fetcher import fetch_natural_gas_prices, fetch_natural_gas_storage

# Import feature engineering
from src.data_processing.feature_engineering import (
    calculate_technical_indicators,
    integrate_fundamental_data,
    create_seasonal_features,
    create_lag_features,
    normalize_features
)

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


def fetch_weather_data(
    start_date: str,
    end_date: str,
    location_id: Optional[str] = None,
    config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Fetch weather data from NOAA API.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        location_id: Station ID for weather data
        config: Configuration dictionary (optional)
        
    Returns:
        DataFrame with weather data
    """
    if config is None:
        config = load_config()
    
    # Get default location_id from config if not specified
    if location_id is None and 'noaa' in config and 'default_location_id' in config['noaa']:
        location_id = config['noaa']['default_location_id']
    
    # Fallback to Central Park, NY if still not specified
    if location_id is None:
        location_id = 'GHCND:USW00094728'  # Central Park, NY
        
    # Get data types to fetch
    data_types = ['TMAX', 'TMIN', 'PRCP', 'SNOW', 'AWND']
    if 'noaa' in config and 'data_types' in config['noaa']:
        data_types = config['noaa']['data_types']
    
    logger.info(f"Fetching weather data for {location_id} from {start_date} to {end_date}")
    try:
        weather_data = fetch_noaa_data(
            dataset_id='GHCND',
            location_id=location_id,
            start_date=start_date,
            end_date=end_date,
            data_types=data_types
        )
        
        # Process the data to have date as index and columns for each data type
        if not weather_data.empty:
            # Pivot the data to have data types as columns
            weather_df = weather_data.pivot(index='date', columns='datatype', values='value')
            
            # Convert index to datetime
            weather_df.index = pd.to_datetime(weather_df.index)
            
            # Sort by date
            weather_df = weather_df.sort_index()
            
            # Make sure all expected columns exist
            for dt in data_types:
                if dt not in weather_df.columns:
                    weather_df[dt] = np.nan
            
            # Calculate average temperature
            if 'TMAX' in weather_df.columns and 'TMIN' in weather_df.columns:
                weather_df['TAVG'] = (weather_df['TMAX'] + weather_df['TMIN']) / 2
            
            logger.info(f"Successfully processed weather data: {len(weather_df)} records")
            return weather_df
        else:
            logger.warning("No weather data found")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching weather data: {e}")
        return pd.DataFrame()


def fetch_price_data(
    start_date: str,
    end_date: str,
    config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Fetch natural gas price data from EIA API.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        config: Configuration dictionary (optional)
        
    Returns:
        DataFrame with price data
    """
    if config is None:
        config = load_config()
    
    logger.info(f"Fetching natural gas price data from {start_date} to {end_date}")
    try:
        price_data = fetch_natural_gas_prices(
            start_date=start_date,
            end_date=end_date
        )
        
        if not price_data.empty:
            # Process the data
            price_df = price_data.copy()
            
            # Convert value column to numeric
            price_df['value'] = pd.to_numeric(price_df['value'], errors='coerce')
            
            # Filter for Henry Hub prices if available
            henry_hub_data = price_df[price_df['series-description'].str.contains('Henry Hub', case=False, na=False)]
            
            if not henry_hub_data.empty:
                # Use only Henry Hub data
                price_df = henry_hub_data
            
            # Create a clean dataframe with date as index and price as value
            clean_df = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date))
            
            # Merge price data
            price_df['date'] = pd.to_datetime(price_df['date'])
            price_df = price_df.set_index('date')
            
            # Select relevant columns
            if 'value' in price_df.columns:
                clean_df['price'] = price_df['value']
            
            # Forward fill missing values for up to 5 days
            clean_df = clean_df.ffill(limit=5)
            
            logger.info(f"Successfully processed price data: {len(clean_df[~clean_df['price'].isna()])} records")
            return clean_df
        else:
            logger.warning("No price data found")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching price data: {e}")
        return pd.DataFrame()


def fetch_storage_data(
    start_date: str,
    end_date: str,
    config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Fetch natural gas storage data from EIA API.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        config: Configuration dictionary (optional)
        
    Returns:
        DataFrame with storage data
    """
    if config is None:
        config = load_config()
    
    logger.info(f"Fetching natural gas storage data from {start_date} to {end_date}")
    try:
        storage_data = fetch_natural_gas_storage(
            start_date=start_date,
            end_date=end_date
        )
        
        if not storage_data.empty:
            # Process the data
            storage_df = storage_data.copy()
            
            # Convert value column to numeric
            storage_df['value'] = pd.to_numeric(storage_df['value'], errors='coerce')
            
            # Filter for Lower 48 States if available
            lower_48_data = storage_df[storage_df['area-name'].str.contains('Lower 48', case=False, na=False)]
            
            if not lower_48_data.empty:
                # Use only Lower 48 data
                storage_df = lower_48_data
            
            # Create a clean dataframe with date as index and storage as value
            clean_df = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date))
            
            # Merge storage data
            storage_df['date'] = pd.to_datetime(storage_df['date'])
            storage_df = storage_df.set_index('date')
            
            # Select relevant columns
            if 'value' in storage_df.columns:
                clean_df['storage'] = storage_df['value']
            
            # Forward fill missing values - storage is reported weekly
            clean_df = clean_df.ffill(limit=7)
            
            logger.info(f"Successfully processed storage data: {len(clean_df[~clean_df['storage'].isna()])} records")
            return clean_df
        else:
            logger.warning("No storage data found")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching storage data: {e}")
        return pd.DataFrame()


def create_master_dataset(
    start_date: str,
    end_date: str,
    config: Optional[Dict] = None,
    save_to_csv: bool = True
) -> pd.DataFrame:
    """
    Create a master dataset by fetching and merging data from different sources.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        config: Configuration dictionary (optional)
        save_to_csv: Whether to save the master dataset to CSV
        
    Returns:
        DataFrame with merged data
    """
    if config is None:
        config = load_config()
    
    logger.info(f"Creating master dataset from {start_date} to {end_date}")
    
    # Fetch data from different sources
    weather_df = fetch_weather_data(start_date, end_date, config=config)
    price_df = fetch_price_data(start_date, end_date, config=config)
    storage_df = fetch_storage_data(start_date, end_date, config=config)
    
    # Create a master dataframe with date as index
    master_df = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date))
    
    # Merge data from different sources
    if not weather_df.empty:
        # Add weather data
        for col in weather_df.columns:
            master_df[f'weather_{col}'] = weather_df[col]
    
    if not price_df.empty:
        # Add price data
        master_df['price'] = price_df['price']
    
    if not storage_df.empty:
        # Add storage data
        master_df['storage'] = storage_df['storage']
    
    # Fill missing values
    # For weather data, use interpolation for up to 3 days
    weather_cols = [col for col in master_df.columns if col.startswith('weather_')]
    if weather_cols:
        master_df[weather_cols] = master_df[weather_cols].interpolate(method='time', limit=3)
    
    # For price data, forward fill for up to 5 days
    if 'price' in master_df.columns:
        master_df['price'] = master_df['price'].ffill(limit=5)
    
    # For storage data, forward fill for up to 7 days (weekly data)
    if 'storage' in master_df.columns:
        master_df['storage'] = master_df['storage'].ffill(limit=7)
    
    # Drop rows with missing price data
    if 'price' in master_df.columns:
        master_df = master_df.dropna(subset=['price'])
    
    logger.info(f"Created master dataset with {len(master_df)} records and {len(master_df.columns)} features")
    
    if save_to_csv:
        try:
            # Create directory if it doesn't exist
            data_dir = Path(__file__).parents[2] / 'data' / 'processed'
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Save to CSV
            output_path = data_dir / 'master_dataset.csv'
            master_df.to_csv(output_path)
            logger.info(f"Saved master dataset to {output_path}")
        except Exception as e:
            logger.error(f"Error saving master dataset: {e}")
    
    return master_df


def process_features(
    master_df: pd.DataFrame,
    config: Optional[Dict] = None,
    save_to_csv: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Process features for the trading model.
    
    Args:
        master_df: Master dataset
        config: Configuration dictionary (optional)
        save_to_csv: Whether to save the processed features to CSV
        
    Returns:
        Tuple of (processed_df, normalization_params)
    """
    if config is None:
        config = load_config()
    
    logger.info(f"Processing features for {len(master_df)} records")
    
    # Create a copy of the master dataframe
    df = master_df.copy()
    
    # Calculate technical indicators
    if 'price' in df.columns:
        tech_df = calculate_technical_indicators(df, price_col='price')
        for col in tech_df.columns:
            if col not in df.columns:
                df[col] = tech_df[col]
    
    # Create seasonal features
    seasonal_df = create_seasonal_features(df)
    for col in seasonal_df.columns:
        if col not in df.columns:
            df[col] = seasonal_df[col]
    
    # Create lag features for price
    if 'price' in df.columns:
        lag_periods = [1, 2, 3, 5, 10, 20]
        lag_df = create_lag_features(df, columns=['price'], lag_periods=lag_periods)
        for col in lag_df.columns:
            if col not in df.columns:
                df[col] = lag_df[col]
    
    # Create lag features for storage
    if 'storage' in df.columns:
        lag_periods = [1, 2, 4, 8]  # Storage is weekly
        lag_df = create_lag_features(df, columns=['storage'], lag_periods=lag_periods)
        for col in lag_df.columns:
            if col not in df.columns:
                df[col] = lag_df[col]
    
    # Create weather features
    weather_cols = [col for col in df.columns if col.startswith('weather_')]
    if weather_cols:
        # Calculate heating and cooling degree days
        if 'weather_TAVG' in df.columns:
            df['HDD'] = np.maximum(65 - df['weather_TAVG'], 0)  # Heating Degree Days
            df['CDD'] = np.maximum(df['weather_TAVG'] - 65, 0)  # Cooling Degree Days
        
        # Create lag features for weather
        lag_periods = [1, 2, 3, 7]
        lag_df = create_lag_features(df, columns=['HDD', 'CDD'], lag_periods=lag_periods)
        for col in lag_df.columns:
            if col not in df.columns:
                df[col] = lag_df[col]
    
    # Create target variable (next day return)
    if 'price' in df.columns:
        df['target_next_day_return'] = df['price'].pct_change(1).shift(-1)
    
    # Drop rows with NaN values
    df = df.dropna()
    
    # Normalize features
    exclude_cols = ['target_next_day_return']
    normalized_df, norm_params = normalize_features(df, method='zscore', exclude_cols=exclude_cols)
    
    logger.info(f"Processed features: {len(normalized_df)} records and {len(normalized_df.columns)} features")
    
    if save_to_csv:
        try:
            # Create directory if it doesn't exist
            data_dir = Path(__file__).parents[2] / 'data' / 'processed'
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Save to CSV
            output_path = data_dir / 'processed_features.csv'
            normalized_df.to_csv(output_path)
            
            # Save normalization parameters
            import json
            norm_params_path = data_dir / 'normalization_params.json'
            
            # Convert any numpy types to Python native types for JSON serialization
            clean_params = {}
            for k, v in norm_params.items():
                if isinstance(v, dict):
                    clean_params[k] = {
                        key: float(val) if hasattr(val, 'item') else val
                        for key, val in v.items()
                    }
                else:
                    clean_params[k] = float(v) if hasattr(v, 'item') else v
            
            with open(norm_params_path, 'w') as f:
                json.dump(clean_params, f)
            
            logger.info(f"Saved processed features to {output_path}")
        except Exception as e:
            logger.error(f"Error saving processed features: {e}")
    
    return normalized_df, norm_params


def run_data_pipeline(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    days_back: int = 365,
    config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Run the complete data pipeline.
    
    Args:
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
        days_back: Number of days to look back if start_date is not specified
        config: Configuration dictionary (optional)
        
    Returns:
        DataFrame with processed features
    """
    # Set default dates if not specified
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    logger.info(f"Running data pipeline from {start_date} to {end_date}")
    
    # Load config
    if config is None:
        config = load_config()
    
    # Create master dataset
    master_df = create_master_dataset(start_date, end_date, config, save_to_csv=True)
    
    # Process features
    processed_df, _ = process_features(master_df, config, save_to_csv=True)
    
    return processed_df


if __name__ == "__main__":
    # Run the data pipeline for the past year
    processed_data = run_data_pipeline()
    print(f"Data pipeline completed: {len(processed_data)} records processed") 
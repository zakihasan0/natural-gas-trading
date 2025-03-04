"""
Feature Engineering Module - Process raw data and create features for the trading model.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import os
import logging
from datetime import datetime, timedelta

# Import utility logger
from src.utils.logger import get_logger

# Configure logging
logger = get_logger(__name__)


def calculate_technical_indicators(
    price_df: pd.DataFrame,
    price_col: str = 'close',
    volume_col: Optional[str] = 'volume'
) -> pd.DataFrame:
    """
    Calculate technical indicators from price data.
    
    Args:
        price_df: DataFrame with price data
        price_col: Column name for price data
        volume_col: Column name for volume data. If None, volume indicators are skipped.
    
    Returns:
        DataFrame with added technical indicators
    """
    logger.info("Calculating technical indicators")
    
    # Create a copy to avoid modifying the original
    df = price_df.copy()
    
    # Ensure the DataFrame is sorted by date
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()
    else:
        logger.warning("Index is not a DatetimeIndex, sorting may not work as expected")
        df = df.sort_index()
    
    # Calculate returns
    df['returns'] = df[price_col].pct_change()
    df['log_returns'] = np.log(df[price_col] / df[price_col].shift(1))
    
    # Calculate rolling statistics
    for window in [5, 10, 20, 60]:
        # Moving averages
        df[f'ma_{window}'] = df[price_col].rolling(window=window).mean()
        
        # Exponential moving averages
        df[f'ema_{window}'] = df[price_col].ewm(span=window, adjust=False).mean()
        
        # Volatility (standard deviation of returns)
        df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()
        
        # Rolling max and min
        df[f'max_{window}'] = df[price_col].rolling(window=window).max()
        df[f'min_{window}'] = df[price_col].rolling(window=window).min()
        
        # Z-score (how many standard deviations from the mean)
        rolling_mean = df[price_col].rolling(window=window).mean()
        rolling_std = df[price_col].rolling(window=window).std()
        df[f'zscore_{window}'] = (df[price_col] - rolling_mean) / rolling_std
    
    # Calculate MACD (Moving Average Convergence Divergence)
    ema_12 = df[price_col].ewm(span=12, adjust=False).mean()
    ema_26 = df[price_col].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Calculate RSI (Relative Strength Index)
    delta = df[price_col].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain_14 = gain.rolling(window=14).mean()
    avg_loss_14 = loss.rolling(window=14).mean()
    
    rs_14 = avg_gain_14 / avg_loss_14
    df['rsi_14'] = 100 - (100 / (1 + rs_14))
    
    # Bollinger Bands
    for window in [20]:
        rolling_mean = df[price_col].rolling(window=window).mean()
        rolling_std = df[price_col].rolling(window=window).std()
        
        df[f'bb_middle_{window}'] = rolling_mean
        df[f'bb_upper_{window}'] = rolling_mean + (rolling_std * 2)
        df[f'bb_lower_{window}'] = rolling_mean - (rolling_std * 2)
        df[f'bb_width_{window}'] = (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']) / df[f'bb_middle_{window}']
        df[f'bb_pct_{window}'] = (df[price_col] - df[f'bb_lower_{window}']) / (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}'])
    
    # Rate of Change
    for period in [1, 5, 10, 20]:
        df[f'roc_{period}'] = df[price_col].pct_change(periods=period) * 100
    
    # Volume indicators (if volume data is available)
    if volume_col and volume_col in df.columns:
        # Volume moving average
        df['volume_ma_5'] = df[volume_col].rolling(window=5).mean()
        df['volume_ma_10'] = df[volume_col].rolling(window=10).mean()
        
        # Price-volume trend
        df['pvt'] = df['returns'] * df[volume_col]
        df['pvt_cumulative'] = df['pvt'].cumsum()
        
        # On-balance volume (OBV)
        df['obv_signal'] = np.where(df[price_col] > df[price_col].shift(1), 1, 
                                   np.where(df[price_col] < df[price_col].shift(1), -1, 0))
        df['obv'] = (df[volume_col] * df['obv_signal']).cumsum()
    
    # Momentum
    for period in [3, 6, 12]:
        df[f'momentum_{period}'] = df[price_col].diff(period)
    
    # Clean up NaN values at the beginning of the series
    df = df.fillna(method='backfill').fillna(0)
    
    logger.info(f"Created {len(df.columns) - len(price_df.columns)} technical indicators")
    return df


def integrate_fundamental_data(
    price_df: pd.DataFrame,
    storage_df: Optional[pd.DataFrame] = None,
    weather_df: Optional[pd.DataFrame] = None,
    production_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Integrate fundamental data with price data.
    
    Args:
        price_df: DataFrame with price data (index should be DatetimeIndex)
        storage_df: DataFrame with storage data
        weather_df: DataFrame with weather data
        production_df: DataFrame with production data
    
    Returns:
        DataFrame with integrated data
    """
    logger.info("Integrating fundamental data")
    
    # Create a copy to avoid modifying the original
    df = price_df.copy()
    
    # Ensure all DataFrames have DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("Price DataFrame index is not DatetimeIndex, converting...")
        df.index = pd.to_datetime(df.index)
    
    # Add storage data
    if storage_df is not None:
        if not isinstance(storage_df.index, pd.DatetimeIndex):
            storage_df.index = pd.to_datetime(storage_df.index)
        
        # Reindex storage data to match price data frequency
        # For storage, we forward-fill missing values since storage reports are weekly
        storage_reindexed = storage_df.reindex(df.index, method='ffill')
        
        # Add columns to main DataFrame
        for col in storage_df.columns:
            df[f'storage_{col}'] = storage_reindexed[col]
        
        # Calculate 5-year average for the same week of the year
        if 'value' in storage_df.columns:
            # Extract week of the year
            storage_df['week_of_year'] = storage_df.index.isocalendar().week
            
            # Group by week of the year and calculate average
            week_avg = storage_df.groupby('week_of_year')['value'].mean()
            
            # Map back to original index
            df['week_of_year'] = df.index.isocalendar().week
            df['storage_5yr_avg'] = df['week_of_year'].map(week_avg)
            
            # Calculate deviation from 5-year average
            df['storage_deviation'] = df['storage_value'] - df['storage_5yr_avg']
            df['storage_deviation_pct'] = df['storage_deviation'] / df['storage_5yr_avg'] * 100
            
            # Drop intermediate column
            df = df.drop('week_of_year', axis=1)
    
    # Add weather data
    if weather_df is not None:
        if not isinstance(weather_df.index, pd.DatetimeIndex):
            weather_df.index = pd.to_datetime(weather_df.index)
        
        # Reindex weather data to match price data frequency
        weather_reindexed = weather_df.reindex(df.index, method='ffill')
        
        # Add columns to main DataFrame
        for col in weather_df.columns:
            df[f'weather_{col}'] = weather_reindexed[col]
        
        # Create additional weather indicators
        if all(col in weather_df.columns for col in ['HTDD', 'CLDD']):
            # Calculate cumulative HDDs and CDDs over the last 7, 14, and 30 days
            for window in [7, 14, 30]:
                df[f'weather_HTDD_cum_{window}d'] = df['weather_HTDD'].rolling(window=window).sum()
                df[f'weather_CLDD_cum_{window}d'] = df['weather_CLDD'].rolling(window=window).sum()
            
            # Calculate HDD and CDD anomalies if we have multi-year data
            if len(weather_df) > 365:
                # Extract day of the year
                weather_df['doy'] = weather_df.index.dayofyear
                
                # Group by day of the year and calculate average
                doy_avg_hdd = weather_df.groupby('doy')['HTDD'].mean()
                doy_avg_cdd = weather_df.groupby('doy')['CLDD'].mean()
                
                # Map back to original index
                df['doy'] = df.index.dayofyear
                df['weather_HTDD_normal'] = df['doy'].map(doy_avg_hdd)
                df['weather_CLDD_normal'] = df['doy'].map(doy_avg_cdd)
                
                # Calculate anomalies
                df['weather_HTDD_anomaly'] = df['weather_HTDD'] - df['weather_HTDD_normal']
                df['weather_CLDD_anomaly'] = df['weather_CLDD'] - df['weather_CLDD_normal']
                
                # Drop intermediate column
                df = df.drop('doy', axis=1)
    
    # Add production data
    if production_df is not None:
        if not isinstance(production_df.index, pd.DatetimeIndex):
            production_df.index = pd.to_datetime(production_df.index)
        
        # Reindex production data to match price data frequency
        # For production, we forward-fill missing values since production reports are monthly
        production_reindexed = production_df.reindex(df.index, method='ffill')
        
        # Add columns to main DataFrame
        for col in production_df.columns:
            df[f'production_{col}'] = production_reindexed[col]
        
        # Calculate year-over-year changes
        if 'value' in production_df.columns:
            # Calculate 12-month (YoY) change
            df['production_yoy_change'] = df['production_value'].pct_change(periods=252)  # Approx 1 year
            df['production_yoy_change_pct'] = df['production_yoy_change'] / df['production_value'].shift(252) * 100
    
    # Clean up NaN values
    df = df.fillna(method='backfill').fillna(0)
    
    logger.info(f"Integrated fundamental data, final DataFrame has {len(df.columns)} columns")
    return df


def create_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create seasonal features based on the date index.
    
    Args:
        df: DataFrame with DatetimeIndex
    
    Returns:
        DataFrame with added seasonal features
    """
    logger.info("Creating seasonal features")
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    if not isinstance(result_df.index, pd.DatetimeIndex):
        logger.warning("DataFrame index is not DatetimeIndex, converting...")
        result_df.index = pd.to_datetime(result_df.index)
    
    # Extract date components
    result_df['month'] = result_df.index.month
    result_df['day_of_week'] = result_df.index.dayofweek
    result_df['day_of_year'] = result_df.index.dayofyear
    result_df['week_of_year'] = result_df.index.isocalendar().week
    result_df['quarter'] = result_df.index.quarter
    
    # Create cyclic encoding for seasonal components
    # This prevents the model from seeing Dec 31 and Jan 1 as very different
    
    # Month of the year (1-12) -> cyclic encoding
    result_df['month_sin'] = np.sin(2 * np.pi * result_df['month'] / 12)
    result_df['month_cos'] = np.cos(2 * np.pi * result_df['month'] / 12)
    
    # Day of the week (0-6) -> cyclic encoding
    result_df['day_of_week_sin'] = np.sin(2 * np.pi * result_df['day_of_week'] / 7)
    result_df['day_of_week_cos'] = np.cos(2 * np.pi * result_df['day_of_week'] / 7)
    
    # Day of the year (1-365/366) -> cyclic encoding
    result_df['day_of_year_sin'] = np.sin(2 * np.pi * result_df['day_of_year'] / 365)
    result_df['day_of_year_cos'] = np.cos(2 * np.pi * result_df['day_of_year'] / 365)
    
    # Week of the year (1-52/53) -> cyclic encoding
    result_df['week_of_year_sin'] = np.sin(2 * np.pi * result_df['week_of_year'] / 52)
    result_df['week_of_year_cos'] = np.cos(2 * np.pi * result_df['week_of_year'] / 52)
    
    # Heating/cooling season indicators
    # Northern hemisphere heating season is roughly Oct-Mar (months 10-12, 1-3)
    result_df['heating_season'] = result_df['month'].isin([1, 2, 3, 10, 11, 12]).astype(int)
    # Northern hemisphere cooling season is roughly May-Sep (months 5-9)
    result_df['cooling_season'] = result_df['month'].isin([5, 6, 7, 8, 9]).astype(int)
    
    # Drop the original date columns (keep cyclic versions)
    result_df = result_df.drop(['month', 'day_of_week', 'day_of_year', 'week_of_year'], axis=1)
    
    logger.info(f"Created {len(result_df.columns) - len(df.columns)} seasonal features")
    return result_df


def create_lag_features(
    df: pd.DataFrame,
    columns: List[str],
    lag_periods: List[int]
) -> pd.DataFrame:
    """
    Create lagged features for specified columns.
    
    Args:
        df: DataFrame to add lag features to
        columns: List of column names to create lags for
        lag_periods: List of lag periods to create
    
    Returns:
        DataFrame with added lag features
    """
    logger.info(f"Creating lag features for {len(columns)} columns with {len(lag_periods)} lag periods")
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    for col in columns:
        if col not in result_df.columns:
            logger.warning(f"Column {col} not found in DataFrame, skipping")
            continue
        
        for lag in lag_periods:
            result_df[f'{col}_lag_{lag}'] = result_df[col].shift(lag)
    
    logger.info(f"Created {len(result_df.columns) - len(df.columns)} lag features")
    return result_df


def create_rolling_features(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int],
    functions: Dict[str, callable]
) -> pd.DataFrame:
    """
    Create rolling window aggregations for specified columns.
    
    Args:
        df: DataFrame to add rolling features to
        columns: List of column names to create rolling features for
        windows: List of rolling window sizes
        functions: Dictionary mapping function names to functions
    
    Returns:
        DataFrame with added rolling features
    """
    logger.info(f"Creating rolling features for {len(columns)} columns with {len(windows)} windows")
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    for col in columns:
        if col not in result_df.columns:
            logger.warning(f"Column {col} not found in DataFrame, skipping")
            continue
        
        for window in windows:
            for func_name, func in functions.items():
                result_df[f'{col}_{func_name}_{window}'] = result_df[col].rolling(window=window).apply(func)
    
    logger.info(f"Created {len(result_df.columns) - len(df.columns)} rolling features")
    return result_df


def normalize_features(
    df: pd.DataFrame,
    method: str = 'zscore',
    exclude_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Normalize features using the specified method.
    
    Args:
        df: DataFrame with features to normalize
        method: Normalization method ('zscore', 'minmax', or 'robust')
        exclude_cols: List of columns to exclude from normalization
    
    Returns:
        Tuple of normalized DataFrame and dictionary with normalization parameters
    """
    logger.info(f"Normalizing features using {method} method")
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Default to empty list if exclude_cols is None
    exclude_cols = exclude_cols or []
    
    # Dictionary to store normalization parameters for inverse transform
    norm_params = {}
    
    # Columns to normalize (exclude specified columns)
    norm_cols = [col for col in result_df.columns if col not in exclude_cols]
    
    for col in norm_cols:
        if method == 'zscore':
            # Z-score normalization (mean=0, std=1)
            mean = result_df[col].mean()
            std = result_df[col].std()
            
            # Skip if std is 0 or NaN
            if std == 0 or pd.isna(std):
                logger.warning(f"Skipping normalization for {col} (std=0 or NaN)")
                continue
            
            result_df[col] = (result_df[col] - mean) / std
            norm_params[col] = {'method': 'zscore', 'mean': mean, 'std': std}
            
        elif method == 'minmax':
            # Min-max normalization (range: 0-1)
            min_val = result_df[col].min()
            max_val = result_df[col].max()
            
            # Skip if min equals max
            if min_val == max_val:
                logger.warning(f"Skipping normalization for {col} (min=max)")
                continue
            
            result_df[col] = (result_df[col] - min_val) / (max_val - min_val)
            norm_params[col] = {'method': 'minmax', 'min': min_val, 'max': max_val}
            
        elif method == 'robust':
            # Robust scaling using quantiles
            q1 = result_df[col].quantile(0.25)
            q3 = result_df[col].quantile(0.75)
            iqr = q3 - q1
            
            # Skip if IQR is 0
            if iqr == 0:
                logger.warning(f"Skipping normalization for {col} (IQR=0)")
                continue
            
            result_df[col] = (result_df[col] - q1) / iqr
            norm_params[col] = {'method': 'robust', 'q1': q1, 'q3': q3, 'iqr': iqr}
            
        else:
            logger.warning(f"Unknown normalization method: {method}, skipping")
    
    logger.info(f"Normalized {len(norm_params)} features")
    return result_df, norm_params


def denormalize_features(
    df: pd.DataFrame,
    norm_params: Dict
) -> pd.DataFrame:
    """
    Denormalize features using the parameters from normalize_features.
    
    Args:
        df: Normalized DataFrame
        norm_params: Dictionary with normalization parameters
    
    Returns:
        Denormalized DataFrame
    """
    logger.info("Denormalizing features")
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    for col, params in norm_params.items():
        if col not in result_df.columns:
            logger.warning(f"Column {col} not found in DataFrame, skipping")
            continue
        
        method = params.get('method')
        
        if method == 'zscore':
            mean = params['mean']
            std = params['std']
            result_df[col] = result_df[col] * std + mean
            
        elif method == 'minmax':
            min_val = params['min']
            max_val = params['max']
            result_df[col] = result_df[col] * (max_val - min_val) + min_val
            
        elif method == 'robust':
            q1 = params['q1']
            iqr = params['iqr']
            result_df[col] = result_df[col] * iqr + q1
            
        else:
            logger.warning(f"Unknown normalization method: {method}, skipping")
    
    logger.info(f"Denormalized {len(norm_params)} features")
    return result_df


def create_target_variable(
    df: pd.DataFrame,
    price_col: str = 'close',
    horizon: int = 1,
    target_type: str = 'returns'
) -> pd.DataFrame:
    """
    Create target variable for supervised learning.
    
    Args:
        df: DataFrame with price data
        price_col: Column name for price data
        horizon: Forecast horizon (number of periods ahead)
        target_type: Type of target ('returns', 'direction', or 'price')
    
    Returns:
        DataFrame with added target variable
    """
    logger.info(f"Creating {target_type} target variable with horizon {horizon}")
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    if target_type == 'returns':
        # Future returns (percentage change)
        result_df[f'target_{horizon}'] = result_df[price_col].pct_change(periods=horizon).shift(-horizon)
        
    elif target_type == 'direction':
        # Binary classification target (price direction)
        future_price = result_df[price_col].shift(-horizon)
        result_df[f'target_{horizon}'] = (future_price > result_df[price_col]).astype(int)
        
    elif target_type == 'price':
        # Raw future price
        result_df[f'target_{horizon}'] = result_df[price_col].shift(-horizon)
        
    else:
        logger.warning(f"Unknown target type: {target_type}, defaulting to returns")
        result_df[f'target_{horizon}'] = result_df[price_col].pct_change(periods=horizon).shift(-horizon)
    
    logger.info(f"Created target variable 'target_{horizon}'")
    return result_df


def prepare_features_for_training(
    raw_data_dir: Path,
    processed_data_dir: Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    target_horizon: int = 1,
    target_type: str = 'returns'
) -> Tuple[pd.DataFrame, Dict]:
    """
    Prepare features for model training by loading raw data and applying feature engineering.
    
    Args:
        raw_data_dir: Directory containing raw data files
        processed_data_dir: Directory to save processed data
        start_date: Start date for filtering data
        end_date: End date for filtering data
        target_horizon: Forecast horizon for target variable
        target_type: Type of target variable
    
    Returns:
        Tuple of processed DataFrame and normalization parameters
    """
    logger.info("Preparing features for model training")
    
    # Create directories if they don't exist
    os.makedirs(processed_data_dir, exist_ok=True)
    
    # Load price data (continuous futures contract)
    price_file = raw_data_dir / 'cme' / 'ng_continuous.csv'
    if not price_file.exists():
        logger.error(f"Price data file not found: {price_file}")
        raise FileNotFoundError(f"Price data file not found: {price_file}")
    
    price_df = pd.read_csv(price_file, index_col=0, parse_dates=True)
    logger.info(f"Loaded price data with {len(price_df)} rows")
    
    # Load storage data if available
    storage_df = None
    storage_file = raw_data_dir / 'eia' / 'NG.NW2_EPG0_SAX_R48_BCF.W.csv'
    if storage_file.exists():
        storage_df = pd.read_csv(storage_file, index_col=0, parse_dates=True)
        logger.info(f"Loaded storage data with {len(storage_df)} rows")
    
    # Load weather data if available
    weather_df = None
    weather_file = raw_data_dir / 'noaa' / 'noaa_aggregated.csv'
    if weather_file.exists():
        weather_df = pd.read_csv(weather_file, index_col=0, parse_dates=True)
        logger.info(f"Loaded weather data with {len(weather_df)} rows")
    
    # Load production data if available
    production_df = None
    production_file = raw_data_dir / 'eia' / 'NG.N9070US2.M.csv'
    if production_file.exists():
        production_df = pd.read_csv(production_file, index_col=0, parse_dates=True)
        logger.info(f"Loaded production data with {len(production_df)} rows")
    
    # Filter by date range if specified
    if start_date:
        start_date = pd.to_datetime(start_date)
        price_df = price_df[price_df.index >= start_date]
        if storage_df is not None:
            storage_df = storage_df[storage_df.index >= start_date]
        if weather_df is not None:
            weather_df = weather_df[weather_df.index >= start_date]
        if production_df is not None:
            production_df = production_df[production_df.index >= start_date]
    
    if end_date:
        end_date = pd.to_datetime(end_date)
        price_df = price_df[price_df.index <= end_date]
        if storage_df is not None:
            storage_df = storage_df[storage_df.index <= end_date]
        if weather_df is not None:
            weather_df = weather_df[weather_df.index <= end_date]
        if production_df is not None:
            production_df = production_df[production_df.index <= end_date]
    
    # Step 1: Calculate technical indicators
    df = calculate_technical_indicators(price_df)
    
    # Step 2: Integrate fundamental data
    df = integrate_fundamental_data(
        df, 
        storage_df=storage_df,
        weather_df=weather_df,
        production_df=production_df
    )
    
    # Step 3: Create seasonal features
    df = create_seasonal_features(df)
    
    # Step 4: Create lag features for important variables
    important_cols = ['close', 'volume', 'returns', 'rsi_14', 'macd']
    # Add fundamental columns if available
    if 'storage_value' in df.columns:
        important_cols.append('storage_value')
    if 'weather_HTDD' in df.columns:
        important_cols.append('weather_HTDD')
    
    df = create_lag_features(
        df,
        columns=important_cols,
        lag_periods=[1, 2, 3, 5, 10]
    )
    
    # Step 5: Create rolling features for select columns
    rolling_functions = {
        'mean': np.mean,
        'std': np.std,
        'min': np.min,
        'max': np.max
    }
    
    df = create_rolling_features(
        df,
        columns=['returns', 'volume'],
        windows=[5, 10, 20],
        functions=rolling_functions
    )
    
    # Step 6: Create target variable
    df = create_target_variable(
        df,
        price_col='close',
        horizon=target_horizon,
        target_type=target_type
    )
    
    # Step 7: Clean and normalize features
    # Drop rows with NaN in target
    df = df.dropna(subset=[f'target_{target_horizon}'])
    
    # Exclude these columns from normalization
    exclude_from_norm = [
        'open', 'high', 'low', 'close', 'volume',  # Raw price data
        'ticker', 'is_continuous',                 # Metadata
        f'target_{target_horizon}'                 # Target variable (normalize separately)
    ]
    
    # Normalize features
    normalized_df, norm_params = normalize_features(
        df,
        method='zscore',
        exclude_cols=exclude_from_norm
    )
    
    # Save processed data
    output_file = processed_data_dir / 'processed_features.csv'
    normalized_df.to_csv(output_file)
    logger.info(f"Saved processed features to {output_file}")
    
    # Save normalization parameters
    params_file = processed_data_dir / 'normalization_params.csv'
    params_df = pd.DataFrame([
        {
            'column': col,
            'method': params['method'],
            **{k: v for k, v in params.items() if k != 'method'}
        }
        for col, params in norm_params.items()
    ])
    params_df.to_csv(params_file, index=False)
    logger.info(f"Saved normalization parameters to {params_file}")
    
    return normalized_df, norm_params


if __name__ == "__main__":
    # Example usage
    try:
        # Set paths
        base_dir = Path(__file__).parents[2]
        raw_data_dir = base_dir / 'data' / 'raw'
        processed_data_dir = base_dir / 'data' / 'processed'
        
        # Ensure directories exist
        os.makedirs(processed_data_dir, exist_ok=True)
        
        # Set date range (last 5 years)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = f"{datetime.now().year - 5}-01-01"
        
        logger.info(f"Preparing features from {start_date} to {end_date}")
        
        # Process features for different horizons and target types
        for horizon in [1, 5, 20]:
            for target_type in ['returns', 'direction']:
                # Create specific output directory
                target_dir = processed_data_dir / f"horizon_{horizon}_{target_type}"
                os.makedirs(target_dir, exist_ok=True)
                
                logger.info(f"Processing features for horizon={horizon}, target_type={target_type}")
                
                df, norm_params = prepare_features_for_training(
                    raw_data_dir=raw_data_dir,
                    processed_data_dir=target_dir,
                    start_date=start_date,
                    end_date=end_date,
                    target_horizon=horizon,
                    target_type=target_type
                )
                
                logger.info(f"Successfully processed features with shape {df.shape}")
        
        logger.info("Feature engineering completed successfully")
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        raise 
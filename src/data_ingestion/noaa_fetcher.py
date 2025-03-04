"""
NOAA Weather Data Fetcher - Module for retrieving weather data from the NOAA API.
"""

import os
import requests
import pandas as pd
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta

# Import utility logger
from src.utils.logger import get_logger

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
        logger.error(f"Error loading configuration: {e}")
        raise


def load_credentials() -> Dict:
    """
    Load API credentials from the credentials file.
    
    Returns:
        Dict: Credentials dictionary
    """
    try:
        credentials_path = Path(__file__).parents[2] / 'config' / 'credentials.yaml'
        with open(credentials_path, 'r') as f:
            credentials = yaml.safe_load(f)
        return credentials
    except Exception as e:
        logger.error(f"Error loading credentials: {e}")
        raise


def fetch_noaa_data(
    dataset_id: Optional[str] = None,
    location_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    data_types: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch weather data from the NOAA API.
    
    Args:
        dataset_id: NOAA dataset ID. If None, uses the default from config.
        location_id: NOAA location ID. If None, uses the default from config.
        start_date: Start date in 'YYYY-MM-DD' format. If None, fetches all available data.
        end_date: End date in 'YYYY-MM-DD' format. If None, fetches up to current date.
        data_types: List of data types to fetch (e.g., 'TMAX', 'TMIN', 'TAVG')
        save_path: Path to save the data. If None, doesn't save.
    
    Returns:
        DataFrame: Pandas DataFrame with the NOAA weather data
    """
    logger.info(f"Fetching NOAA data for location {location_id}")
    
    # Load configuration and credentials
    config = load_config()
    credentials = load_credentials()
    
    # Get API key
    api_token = credentials['noaa']['token']
    
    # Use default values if not provided
    if dataset_id is None:
        dataset_id = config['data_sources']['noaa']['datasets']['daily_summaries']
    
    if data_types is None:
        data_types = ['TMAX', 'TMIN', 'TAVG', 'HTDD', 'CLDD']
    
    # Set default date range if not provided
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    if start_date is None:
        start_date_dt = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=365)
        start_date = start_date_dt.strftime('%Y-%m-%d')
    
    # Construct API URL
    base_url = config['data_sources']['noaa']['base_url']
    url = f"{base_url}data"
    
    # Prepare headers with API token
    headers = {
        'token': api_token
    }
    
    # Prepare parameters
    params = {
        'datasetid': dataset_id,
        'startdate': start_date,
        'enddate': end_date,
        'limit': 1000,
        'includemetadata': False,
        'units': 'standard'
    }
    
    # Add location if provided
    if location_id:
        params['locationid'] = location_id
    
    # Add data types if provided
    if data_types:
        params['datatypeid'] = ','.join(data_types)
    
    try:
        # Make API request
        logger.info(f"Sending request to NOAA API: {url} with params {params}")
        all_data = []
        
        # Handle pagination
        offset = 0
        while True:
            params['offset'] = offset
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'results' in data:
                results = data['results']
                if not results:
                    break
                
                all_data.extend(results)
                offset += len(results)
                
                if len(results) < params['limit']:
                    break
            else:
                break
        
        # Transform the data into a pandas DataFrame
        df = pd.DataFrame(all_data)
        
        if df.empty:
            logger.warning(f"No data returned from NOAA API for location {location_id}")
            return pd.DataFrame()
        
        # Process the DataFrame
        logger.info(f"Processing {len(df)} records from NOAA API")
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Pivot the DataFrame to have data types as columns
        pivot_df = df.pivot_table(index=['date', 'station'], 
                                 columns='datatype', 
                                 values='value')
        
        # Reset the index to move date back to a column
        pivot_df.reset_index(level='station', inplace=True)
        
        # Save the data if a path is provided
        if save_path:
            logger.info(f"Saving NOAA data to {save_path}")
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            pivot_df.to_csv(save_path)
        
        return pivot_df
    
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error when fetching NOAA data: {e}")
        if e.response.status_code == 429:
            logger.error("Rate limit exceeded. Please wait and try again.")
        elif e.response.status_code == 403:
            logger.error("Authentication failed. Check your API key.")
        raise
    except Exception as e:
        logger.error(f"Error fetching NOAA data: {e}")
        raise


def fetch_multiple_locations(
    location_ids: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    data_types: Optional[List[str]] = None,
    save_dir: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Fetch weather data for multiple locations and return as dictionary of DataFrames.
    
    Args:
        location_ids: List of NOAA location IDs
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        data_types: List of data types to fetch
        save_dir: Directory to save the data files
    
    Returns:
        Dict[str, DataFrame]: Dictionary mapping location IDs to DataFrames
    """
    results = {}
    
    for location_id in location_ids:
        logger.info(f"Fetching data for location {location_id}")
        
        save_path = None
        if save_dir:
            location_filename = f"noaa_{location_id.replace(':', '_')}.csv"
            save_path = os.path.join(save_dir, location_filename)
        
        df = fetch_noaa_data(
            location_id=location_id,
            start_date=start_date,
            end_date=end_date,
            data_types=data_types,
            save_path=save_path
        )
        
        results[location_id] = df
    
    return results


def calculate_degree_days(
    temperature_df: pd.DataFrame,
    base_temp: float = 65.0,
    temp_col: str = 'TAVG'
) -> pd.DataFrame:
    """
    Calculate heating and cooling degree days from temperature data.
    
    Args:
        temperature_df: DataFrame with temperature data
        base_temp: Base temperature for degree day calculation (°F)
        temp_col: Column name with temperature data
    
    Returns:
        DataFrame with added HDD and CDD columns
    """
    df = temperature_df.copy()
    
    # Calculate HDD and CDD if not already present
    if 'HTDD' not in df.columns:
        df['HTDD'] = df[temp_col].apply(lambda x: max(0, base_temp - x))
    
    if 'CLDD' not in df.columns:
        df['CLDD'] = df[temp_col].apply(lambda x: max(0, x - base_temp))
    
    return df


def aggregate_weather_data(
    location_dfs: Dict[str, pd.DataFrame],
    weight_map: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Aggregate weather data from multiple locations using weighted average.
    
    Args:
        location_dfs: Dictionary mapping location IDs to DataFrames
        weight_map: Dictionary mapping location IDs to weights (must sum to 1.0)
                    If None, equal weights are assigned
    
    Returns:
        DataFrame with weighted average of weather data
    """
    if not location_dfs:
        logger.warning("No location data provided for aggregation")
        return pd.DataFrame()
    
    # Assign equal weights if not provided
    if weight_map is None:
        n_locations = len(location_dfs)
        weight_map = {loc_id: 1.0 / n_locations for loc_id in location_dfs.keys()}
    
    # Check weights sum to 1.0
    weight_sum = sum(weight_map.values())
    if abs(weight_sum - 1.0) > 1e-6:
        logger.warning(f"Weights do not sum to 1.0 (sum={weight_sum}), normalizing")
        weight_map = {k: v / weight_sum for k, v in weight_map.items()}
    
    # Find common columns and dates
    common_cols = set.intersection(*[set(df.columns) for df in location_dfs.values()])
    all_dates = sorted(set.union(*[set(df.index) for df in location_dfs.values()]))
    
    # Create empty aggregated DataFrame
    date_idx = pd.DatetimeIndex(all_dates)
    agg_df = pd.DataFrame(index=date_idx)
    
    # Calculate weighted average for each column
    for col in common_cols:
        agg_df[col] = 0.0
        for loc_id, df in location_dfs.items():
            if loc_id in weight_map and col in df.columns:
                # Reindex to match all dates, filling with NaN
                reindexed = df.reindex(date_idx)
                weight = weight_map[loc_id]
                
                # Apply weight and add to aggregate (treating NaN as 0 for calculation)
                agg_df[col] += reindexed[col].fillna(0) * weight
    
    return agg_df


if __name__ == "__main__":
    # Example usage
    try:
        # Create data directories if they don't exist
        raw_data_dir = Path(__file__).parents[2] / 'data' / 'raw' / 'noaa'
        os.makedirs(raw_data_dir, exist_ok=True)
        
        # Load config to get location IDs
        config = load_config()
        location_ids = config['data_sources']['noaa']['locations']
        
        # Set date range (last 2 years)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date_dt = datetime.now() - timedelta(days=365*2)
        start_date = start_date_dt.strftime('%Y-%m-%d')
        
        logger.info(f"Fetching NOAA data from {start_date} to {end_date}")
        
        # Data types to fetch
        data_types = ['TMAX', 'TMIN', 'TAVG', 'HTDD', 'CLDD']
        
        # Fetch all locations
        data_dict = fetch_multiple_locations(
            location_ids=location_ids,
            start_date=start_date,
            end_date=end_date,
            data_types=data_types,
            save_dir=raw_data_dir
        )
        
        # Aggregate data (optional)
        if data_dict:
            agg_df = aggregate_weather_data(data_dict)
            agg_save_path = raw_data_dir / 'noaa_aggregated.csv'
            agg_df.to_csv(agg_save_path)
            logger.info(f"Aggregated data saved to {agg_save_path}")
        
        logger.info(f"Successfully fetched and processed data for {len(data_dict)} locations")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise 
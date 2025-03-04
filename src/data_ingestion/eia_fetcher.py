"""
EIA Data Fetcher - Module for retrieving natural gas data from the Energy Information Administration API.
"""

import os
import requests
import pandas as pd
from datetime import datetime
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union
import json

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


def fetch_ng_prices(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    frequency: str = 'monthly',
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch natural gas price data from the EIA API v2.
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format. If None, uses default start date.
        end_date: End date in 'YYYY-MM-DD' format. If None, fetches up to current date.
        frequency: Data frequency ('weekly', 'monthly', etc.)
        save_path: Path to save the data. If None, doesn't save.
    
    Returns:
        DataFrame: Pandas DataFrame with the EIA natural gas price data
    """
    logger.info(f"Fetching EIA natural gas price data with frequency {frequency}")
    
    # Load configuration and credentials
    config = load_config()
    credentials = load_credentials()
    
    # Get API key
    api_key = credentials['eia']['api_key']
    
    # Construct API URL
    base_url = config['data_sources']['eia']['base_url']
    url = f"{base_url}{config['data_sources']['eia']['natural_gas']['prices_url']}"
    
    # Prepare parameters
    params = {
        "frequency": frequency,
        "data": ["value"],
        "sort": [{"column": "period", "direction": "desc"}],
        "offset": 0,
        "length": 5000
    }
    
    # Add date parameters if provided
    if start_date:
        params["start"] = start_date
    if end_date:
        params["end"] = end_date
    
    # Prepare headers with API key
    headers = {
        "X-Api-Key": api_key,
        "X-Params": json.dumps(params)
    }
    
    try:
        # Make API request
        logger.info(f"Sending request to EIA API: {url}")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse response
        data = response.json()
        
        # Extract data from response
        if 'response' in data and 'data' in data['response']:
            data_list = data['response']['data']
            df = pd.DataFrame(data_list)
            
            # Rename columns to more descriptive names if needed
            if 'period' in df.columns:
                df.rename(columns={'period': 'date'}, inplace=True)
            
            # Convert date to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Set date as index
            df = df.set_index('date')
            
            # Sort by date
            df = df.sort_index()
            
            # Save the data if path provided
            if save_path:
                logger.info(f"Saving EIA natural gas price data to {save_path}")
                save_dir = os.path.dirname(save_path)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                df.to_csv(save_path)
            
            return df
        else:
            logger.error(f"Unexpected API response format: {data}")
            raise ValueError("Unexpected API response format")
    
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error when fetching EIA data: {e}")
        if e.response.status_code == 429:
            logger.error("Rate limit exceeded. Please wait and try again.")
        elif e.response.status_code == 403:
            logger.error("Authentication failed. Check your API key.")
        raise
    except Exception as e:
        logger.error(f"Error fetching EIA data: {e}")
        raise


def fetch_ng_storage(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    frequency: str = 'weekly',
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch natural gas storage data from the EIA API v2.
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format. If None, uses default start date.
        end_date: End date in 'YYYY-MM-DD' format. If None, fetches up to current date.
        frequency: Data frequency ('weekly', 'monthly', etc.)
        save_path: Path to save the data. If None, doesn't save.
    
    Returns:
        DataFrame: Pandas DataFrame with the EIA natural gas storage data
    """
    logger.info(f"Fetching EIA natural gas storage data with frequency {frequency}")
    
    # Load configuration and credentials
    config = load_config()
    credentials = load_credentials()
    
    # Get API key
    api_key = credentials['eia']['api_key']
    
    # Construct API URL
    base_url = config['data_sources']['eia']['base_url']
    url = f"{base_url}{config['data_sources']['eia']['natural_gas']['storage_url']}"
    
    # Prepare parameters
    params = {
        "frequency": frequency,
        "data": ["value"],
        "sort": [{"column": "period", "direction": "desc"}],
        "offset": 0,
        "length": 5000
    }
    
    # Add date parameters if provided
    if start_date:
        params["start"] = start_date
    if end_date:
        params["end"] = end_date
    
    # Prepare headers with API key
    headers = {
        "X-Api-Key": api_key,
        "X-Params": json.dumps(params)
    }
    
    try:
        # Make API request
        logger.info(f"Sending request to EIA API: {url}")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse response
        data = response.json()
        
        # Extract data from response
        if 'response' in data and 'data' in data['response']:
            data_list = data['response']['data']
            df = pd.DataFrame(data_list)
            
            # Rename columns to more descriptive names if needed
            if 'period' in df.columns:
                df.rename(columns={'period': 'date'}, inplace=True)
            
            # Convert date to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Set date as index
            df = df.set_index('date')
            
            # Sort by date
            df = df.sort_index()
            
            # Save the data if path provided
            if save_path:
                logger.info(f"Saving EIA natural gas storage data to {save_path}")
                save_dir = os.path.dirname(save_path)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                df.to_csv(save_path)
            
            return df
        else:
            logger.error(f"Unexpected API response format: {data}")
            raise ValueError("Unexpected API response format")
    
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error when fetching EIA data: {e}")
        if e.response.status_code == 429:
            logger.error("Rate limit exceeded. Please wait and try again.")
        elif e.response.status_code == 403:
            logger.error("Authentication failed. Check your API key.")
        raise
    except Exception as e:
        logger.error(f"Error fetching EIA data: {e}")
        raise


def fetch_eia_data(
    series_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    frequency: str = 'weekly',
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    [DEPRECATED] Fetch data from the EIA API v1 (to be replaced with v2 API).
    
    Use fetch_ng_prices() or fetch_ng_storage() instead for better integration with
    the EIA API v2 endpoints.
    
    Args:
        series_id: EIA series ID. If None, uses the storage series from config.
        start_date: Start date in 'YYYY-MM-DD' format. If None, fetches all available data.
        end_date: End date in 'YYYY-MM-DD' format. If None, fetches up to current date.
        frequency: Data frequency ('weekly', 'monthly', etc.)
        save_path: Path to save the data. If None, doesn't save.
    
    Returns:
        DataFrame: Pandas DataFrame with the EIA data
    """
    logger.warning("fetch_eia_data is deprecated. Use fetch_ng_prices or fetch_ng_storage instead.")
    logger.info(f"Fetching EIA data for series {series_id}")
    
    # Load configuration and credentials
    config = load_config()
    credentials = load_credentials()
    
    # Get API key
    api_key = credentials['eia']['api_key']
    
    # Use default series ID if not provided
    if series_id is None:
        series_id = config['data_sources']['eia']['series']['ng_storage']
    
    # Construct API URL for v1 API (deprecated)
    base_url = "https://api.eia.gov/series/"
    url = f"{base_url}?api_key={api_key}&series_id={series_id}"
    
    # Add date parameters if provided
    if start_date:
        url += f"&start={start_date}"
    if end_date:
        url += f"&end={end_date}"
    
    try:
        # Make API request
        logger.info(f"Sending request to EIA API v1: {url.replace(api_key, 'API_KEY')}")
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse response
        data = response.json()
        
        # Extract series data
        if 'series' in data and len(data['series']) > 0 and 'data' in data['series'][0]:
            series_data = data['series'][0]['data']
            df = pd.DataFrame(series_data, columns=['date', 'value'])
            
            # Convert date to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Set date as index
            df = df.set_index('date')
            
            # Sort by date
            df = df.sort_index()
            
            # Save data if path provided
            if save_path:
                logger.info(f"Saving EIA data to {save_path}")
                save_dir = os.path.dirname(save_path)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                df.to_csv(save_path)
            
            return df
        else:
            logger.error(f"Unexpected API response format: {data}")
            raise ValueError("Unexpected API response format")
    
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error when fetching EIA data: {e}")
        if e.response.status_code == 429:
            logger.error("Rate limit exceeded. Please wait and try again.")
        elif e.response.status_code == 403:
            logger.error("Authentication failed. Check your API key.")
        raise
    except Exception as e:
        logger.error(f"Error fetching EIA data: {e}")
        raise 
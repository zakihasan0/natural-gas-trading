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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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


def fetch_eia_data(
    series_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    frequency: str = 'weekly',
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch natural gas data from the EIA API.
    
    Args:
        series_id: EIA series ID. If None, uses the storage series from config.
        start_date: Start date in 'YYYY-MM-DD' format. If None, fetches all available data.
        end_date: End date in 'YYYY-MM-DD' format. If None, fetches up to current date.
        frequency: Data frequency ('weekly', 'monthly', etc.)
        save_path: Path to save the data. If None, doesn't save.
    
    Returns:
        DataFrame: Pandas DataFrame with the EIA data
    """
    logger.info(f"Fetching EIA data for series {series_id}")
    
    # Load configuration and credentials
    config = load_config()
    credentials = load_credentials()
    
    # Get API key
    api_key = credentials['api_keys']['eia']['api_key']
    
    # Use default series ID if not provided
    if series_id is None:
        series_id = config['data_sources']['eia']['series']['ng_storage']
    
    # Construct API URL
    base_url = config['data_sources']['eia']['base_url']
    url = f"{base_url}series/?api_key={api_key}&series_id={series_id}"
    
    # Add date parameters if provided
    if start_date:
        url += f"&start={start_date}"
    if end_date:
        url += f"&end={end_date}"
    
    try:
        # Make API request
        logger.info(f"Sending request to EIA API: {url.replace(api_key, 'API_KEY')}")
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse response
        data = response.json()
        
        # Extract series data
        if 'response' in data and 'data' in data['response']:
            series_data = data['response']['data']
            df = pd.DataFrame(series_data, columns=['date', 'value'])
            
            # Convert date to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Set date as index
            df = df.set_index('date')
            
            # Sort by date
            df = df.sort_index()
            
            # Save data if path provided
            if save_path:
                save_dir = Path(save_path).parent
                os.makedirs(save_dir, exist_ok=True)
                df.to_csv(save_path)
                logger.info(f"Data saved to {save_path}")
            
            logger.info(f"Successfully fetched {len(df)} records for series {series_id}")
            return df
        else:
            logger.error(f"Unexpected response format: {data}")
            raise ValueError("Unexpected API response format")
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching EIA data: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


def fetch_multiple_series(
    series_ids: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    save_dir: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Fetch multiple EIA series and return as dictionary of DataFrames.
    
    Args:
        series_ids: List of EIA series IDs
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        save_dir: Directory to save the data files
    
    Returns:
        Dict[str, DataFrame]: Dictionary mapping series IDs to DataFrames
    """
    results = {}
    
    for series_id in series_ids:
        logger.info(f"Fetching series {series_id}")
        
        save_path = None
        if save_dir:
            series_filename = f"{series_id.replace('.', '_')}.csv"
            save_path = os.path.join(save_dir, series_filename)
        
        df = fetch_eia_data(
            series_id=series_id,
            start_date=start_date,
            end_date=end_date,
            save_path=save_path
        )
        
        results[series_id] = df
    
    return results


if __name__ == "__main__":
    # Example usage
    try:
        # Create data directories if they don't exist
        raw_data_dir = Path(__file__).parents[2] / 'data' / 'raw' / 'eia'
        os.makedirs(raw_data_dir, exist_ok=True)
        
        # Load config to get series IDs
        config = load_config()
        series_map = config['data_sources']['eia']['series']
        series_ids = list(series_map.values())
        
        # Set date range (last 5 years)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = f"{int(end_date[:4]) - 5}-{end_date[5:7]}-{end_date[8:10]}"
        
        logger.info(f"Fetching EIA data from {start_date} to {end_date}")
        
        # Fetch all series
        data_dict = fetch_multiple_series(
            series_ids=series_ids,
            start_date=start_date,
            end_date=end_date,
            save_dir=raw_data_dir
        )
        
        logger.info(f"Successfully fetched {len(data_dict)} series")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise 
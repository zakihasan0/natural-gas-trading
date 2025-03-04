"""
Simple test script for the NOAA API.
"""

import os
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = str(Path(__file__).parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from datetime import datetime, timedelta
import logging

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_noaa_api():
    """Test the NOAA API with specific parameters."""
    import requests
    import yaml
    
    # Load credentials
    credentials_path = Path(__file__).parents[2] / 'config' / 'credentials.yaml'
    with open(credentials_path, 'r') as f:
        credentials = yaml.safe_load(f)
    
    # Get NOAA token from credentials
    token = credentials['noaa']['token']
    
    # Set parameters for a historical period (2020)
    dataset_id = "GHCND"
    station_id = "GHCND:USW00094728"  # Central Park, NY
    start_date = "2020-01-01"
    end_date = "2020-01-31"
    
    # Construct API URL
    base_url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
    
    # Prepare headers with API token
    headers = {
        'token': token
    }
    
    # Prepare parameters
    params = {
        'datasetid': dataset_id,
        'stationid': station_id,
        'startdate': start_date,
        'enddate': end_date,
        'limit': 1000,
        'includemetadata': False,
        'units': 'standard'
    }
    
    logger.info(f"Sending request to NOAA API: {base_url}")
    logger.info(f"Parameters: {params}")
    
    # Make API request
    response = requests.get(base_url, headers=headers, params=params)
    
    # Check status code
    if response.status_code != 200:
        logger.error(f"API request failed with status code {response.status_code}")
        logger.error(f"Response: {response.text}")
        return
    
    # Parse response
    data = response.json()
    
    # Check if results exist
    if 'results' not in data or not data['results']:
        logger.error("No results found in the API response")
        return
    
    # Process results
    results = data['results']
    logger.info(f"Found {len(results)} records")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Print sample of the data
    logger.info(f"Data columns: {df.columns.tolist()}")
    logger.info(f"Data sample:\n{df.head()}")
    
    # Save to CSV
    output_path = Path(__file__).parents[2] / 'data' / 'raw' / 'noaa' / 'test_data.csv'
    os.makedirs(output_path.parent, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Data saved to {output_path}")

if __name__ == "__main__":
    test_noaa_api() 
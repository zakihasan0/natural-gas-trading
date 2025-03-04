"""
Example script demonstrating how to use the NOAA and EIA API functions.

This script shows:
1. How to fetch NOAA weather data
2. How to fetch EIA natural gas price data
3. How to fetch EIA natural gas storage data
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
import matplotlib.pyplot as plt

# Import the API functions
from src.data_ingestion.noaa_fetcher import fetch_noaa_data
from src.data_ingestion.eia_fetcher import fetch_ng_prices, fetch_ng_storage

# Import the utility logger, or create a simple logger if that fails
try:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

def main():
    """Run the example API fetches."""
    try:
        # Create data directories if they don't exist
        data_dir = Path(__file__).parents[2] / 'data'
        raw_data_dir = data_dir / 'raw'
        viz_dir = data_dir / 'visualizations'
        os.makedirs(raw_data_dir / 'noaa', exist_ok=True)
        os.makedirs(raw_data_dir / 'eia', exist_ok=True)
        os.makedirs(viz_dir, exist_ok=True)
        
        # Set date range (most recent year of data)
        end_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')  # 30 days ago to avoid issues with recent data
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')  # 1 year of data
        
        logger.info(f"Fetching data from {start_date} to {end_date}")
        
        # 1. Fetch NOAA weather data for Central Park, NY
        logger.info("Fetching NOAA weather data for Central Park, NY")
        weather_data = fetch_noaa_data(
            dataset_id="GHCND",
            location_id="USW00094728",  # Central Park, NY
            start_date=start_date,
            end_date=end_date,
            data_types=["TMAX", "TMIN", "TAVG", "PRCP"],
            save_path=str(raw_data_dir / 'noaa' / 'central_park_ny.csv')
        )
        
        if not weather_data.empty:
            logger.info(f"Successfully fetched {len(weather_data)} weather records")
            # Plot temperature data
            if 'TAVG' in weather_data.columns:
                plt.figure(figsize=(12, 6))
                weather_data['TAVG'].plot(
                    title="Average Temperature - Central Park, NY"
                )
                plt.ylabel('Temperature (°F)')
                plt.savefig(str(viz_dir / 'temperature.png'))
                plt.close()
        
        # 2. Fetch EIA natural gas price data
        logger.info("Fetching EIA natural gas price data")
        price_data = fetch_ng_prices(
            start_date=start_date,
            end_date=end_date,
            frequency='monthly',
            save_path=str(raw_data_dir / 'eia' / 'ng_prices.csv')
        )
        
        if not price_data.empty:
            logger.info(f"Successfully fetched {len(price_data)} price records")
            
            # Examine and prepare the data for plotting
            logger.info(f"EIA price data columns: {price_data.columns.tolist()}")
            logger.info(f"EIA price data types: {price_data.dtypes}")
            
            # Convert value column to numeric if it's not already
            if 'value' in price_data.columns:
                price_data['value'] = pd.to_numeric(price_data['value'], errors='coerce')
                
                # Plot only if we have numeric data
                if not price_data['value'].isna().all():
                    plt.figure(figsize=(12, 6))
                    price_data['value'].plot(
                        title="Natural Gas Prices"
                    )
                    plt.ylabel('Price ($/MMBtu)')
                    plt.savefig(str(viz_dir / 'prices.png'))
                    plt.close()
                else:
                    logger.warning("No numeric values in price data to plot")
            else:
                logger.warning(f"'value' column not found in price data. Available columns: {price_data.columns.tolist()}")
                
                # Try to find the right column to plot
                for col in price_data.columns:
                    if 'price' in col.lower() or 'value' in col.lower() or 'cost' in col.lower():
                        try:
                            numeric_col = pd.to_numeric(price_data[col], errors='coerce')
                            if not numeric_col.isna().all():
                                plt.figure(figsize=(12, 6))
                                numeric_col.plot(
                                    title=f"Natural Gas Prices - {col}"
                                )
                                plt.ylabel('Price ($/MMBtu)')
                                plt.savefig(str(viz_dir / 'prices.png'))
                                plt.close()
                                logger.info(f"Plotted {col} column instead of 'value'")
                                break
                        except:
                            continue
        
        # 3. Fetch EIA natural gas storage data
        logger.info("Fetching EIA natural gas storage data")
        storage_data = fetch_ng_storage(
            start_date=start_date,
            end_date=end_date,
            frequency='weekly',
            save_path=str(raw_data_dir / 'eia' / 'ng_storage.csv')
        )
        
        if not storage_data.empty:
            logger.info(f"Successfully fetched {len(storage_data)} storage records")
            
            # Examine and prepare the data for plotting
            logger.info(f"EIA storage data columns: {storage_data.columns.tolist()}")
            logger.info(f"EIA storage data types: {storage_data.dtypes}")
            
            # Convert value column to numeric if it's not already
            if 'value' in storage_data.columns:
                storage_data['value'] = pd.to_numeric(storage_data['value'], errors='coerce')
                
                # Plot only if we have numeric data
                if not storage_data['value'].isna().all():
                    plt.figure(figsize=(12, 6))
                    storage_data['value'].plot(
                        title="Natural Gas Storage"
                    )
                    plt.ylabel('Storage (Bcf)')
                    plt.savefig(str(viz_dir / 'storage.png'))
                    plt.close()
                else:
                    logger.warning("No numeric values in storage data to plot")
            else:
                logger.warning(f"'value' column not found in storage data. Available columns: {storage_data.columns.tolist()}")
                
                # Try to find the right column to plot
                for col in storage_data.columns:
                    if 'storage' in col.lower() or 'value' in col.lower() or 'volume' in col.lower():
                        try:
                            numeric_col = pd.to_numeric(storage_data[col], errors='coerce')
                            if not numeric_col.isna().all():
                                plt.figure(figsize=(12, 6))
                                numeric_col.plot(
                                    title=f"Natural Gas Storage - {col}"
                                )
                                plt.ylabel('Storage (Bcf)')
                                plt.savefig(str(viz_dir / 'storage.png'))
                                plt.close()
                                logger.info(f"Plotted {col} column instead of 'value'")
                                break
                        except:
                            continue
        
        logger.info("Example completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main() 
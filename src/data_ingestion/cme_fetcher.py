"""
CME Futures Data Fetcher - Module for retrieving natural gas futures data from Yahoo Finance API.

Since direct CME API access requires a paid subscription, this module uses Yahoo Finance as a proxy
to access natural gas futures data. For production systems, a direct CME data feed is recommended.
"""

import os
import pandas as pd
import yfinance as yf
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


def get_ng_futures_ticker(month_code: str, year: int) -> str:
    """
    Generate a ticker symbol for a specific natural gas futures contract.
    
    Args:
        month_code: Single letter month code (F, G, H, J, K, M, N, Q, U, V, X, Z)
        year: 4-digit year
    
    Returns:
        Ticker symbol for the specified contract
    """
    # Month codes for futures contracts
    # F=Jan, G=Feb, H=Mar, J=Apr, K=May, M=Jun, N=Jul, Q=Aug, U=Sep, V=Oct, X=Nov, Z=Dec
    valid_months = {'F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z'}
    
    if month_code not in valid_months:
        raise ValueError(f"Invalid month code: {month_code}. Must be one of {valid_months}")
    
    # Format ticker (YF format for NYMEX NG futures)
    ticker = f"NG{month_code}{str(year)[-2:]}.NYM"
    
    return ticker


def get_active_contract_ticker() -> str:
    """
    Get the ticker for the currently active (front-month) natural gas contract.
    
    Returns:
        Ticker symbol for the active contract
    """
    config = load_config()
    base_symbol = config['data_sources']['cme']['symbol']
    
    # Month codes for futures contracts
    month_codes = {'01': 'F', '02': 'G', '03': 'H', '04': 'J', '05': 'K', '06': 'M',
                   '07': 'N', '08': 'Q', '09': 'U', '10': 'V', '11': 'X', '12': 'Z'}
    
    # Get current date
    today = datetime.now()
    
    # For simplicity, we'll consider the active contract to be:
    # - The next month's contract if we're in the last week of the month
    # - The current month's contract otherwise
    
    # Calculate days left in current month
    days_in_month = (datetime(today.year, today.month % 12 + 1, 1) - timedelta(days=1)).day
    days_left = days_in_month - today.day
    
    # Determine which month's contract is active
    if days_left <= 7:  # Last week of the month
        # Move to next month
        if today.month == 12:
            active_year = today.year + 1
            active_month = 1
        else:
            active_year = today.year
            active_month = today.month + 1
    else:
        active_year = today.year
        active_month = today.month
    
    # Get month code
    month_str = f"{active_month:02d}"
    month_code = month_codes[month_str]
    
    return get_ng_futures_ticker(month_code, active_year)


def fetch_futures_data(
    ticker: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = '1d',
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch natural gas futures data from Yahoo Finance.
    
    Args:
        ticker: Ticker symbol for the futures contract. If None, uses the active contract.
        start_date: Start date in 'YYYY-MM-DD' format. If None, fetches all available data.
        end_date: End date in 'YYYY-MM-DD' format. If None, fetches up to current date.
        interval: Data interval ('1d', '1wk', '1mo')
        save_path: Path to save the data. If None, doesn't save.
    
    Returns:
        DataFrame: Pandas DataFrame with the futures data
    """
    # Use active contract if ticker not specified
    if ticker is None:
        ticker = get_active_contract_ticker()
    
    logger.info(f"Fetching futures data for {ticker}")
    
    try:
        # Create Ticker object
        futures = yf.Ticker(ticker)
        
        # Fetch historical data
        df = futures.history(
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=True
        )
        
        if df.empty:
            logger.warning(f"No data returned for {ticker}")
            return pd.DataFrame()
        
        # Ensure the index is a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Clean up column names
        df.columns = [col.lower() for col in df.columns]
        
        # Add ticker as a column
        df['ticker'] = ticker
        
        # Save data if path provided
        if save_path:
            save_dir = Path(save_path).parent
            os.makedirs(save_dir, exist_ok=True)
            df.to_csv(save_path)
            logger.info(f"Data saved to {save_path}")
        
        logger.info(f"Successfully fetched {len(df)} records for {ticker}")
        return df
    
    except Exception as e:
        logger.error(f"Error fetching futures data for {ticker}: {e}")
        raise


def fetch_futures_curve(
    month_codes: Optional[List[str]] = None,
    year: Optional[int] = None,
    save_dir: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Fetch full futures curve data (multiple contracts).
    
    Args:
        month_codes: List of month codes to fetch. If None, fetches all months.
        year: Year for contracts. If None, uses current year.
        save_dir: Directory to save the data files. If None, doesn't save.
    
    Returns:
        Dict[str, DataFrame]: Dictionary mapping tickers to DataFrames
    """
    # Default to all month codes if not specified
    if month_codes is None:
        month_codes = ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z']
    
    # Use current year if not specified
    if year is None:
        year = datetime.now().year
    
    logger.info(f"Fetching futures curve for {year}")
    
    results = {}
    for month_code in month_codes:
        ticker = get_ng_futures_ticker(month_code, year)
        
        try:
            save_path = None
            if save_dir:
                filename = f"{ticker.replace('.', '_')}.csv"
                save_path = os.path.join(save_dir, filename)
            
            df = fetch_futures_data(
                ticker=ticker,
                save_path=save_path
            )
            
            if not df.empty:
                results[ticker] = df
        
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            # Continue with other contracts even if one fails
    
    return results


def calculate_term_structure(
    futures_dict: Dict[str, pd.DataFrame],
    date: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculate term structure of natural gas futures for a specific date.
    
    Args:
        futures_dict: Dictionary mapping tickers to DataFrames
        date: Date for term structure in 'YYYY-MM-DD' format. If None, uses most recent date.
    
    Returns:
        DataFrame with contract months and prices
    """
    if not futures_dict:
        logger.warning("No futures data provided")
        return pd.DataFrame()
    
    # Convert date string to datetime if provided
    if date is not None:
        date = pd.to_datetime(date)
    
    # Extract closing prices for each contract
    contracts = []
    
    for ticker, df in futures_dict.items():
        if df.empty:
            continue
        
        # Use provided date or most recent date in the data
        if date is None:
            date = df.index.max()
        
        # Skip if date not in dataframe
        if date not in df.index:
            logger.warning(f"Date {date} not in data for {ticker}")
            continue
        
        # Extract price for the date
        price = df.loc[date, 'close']
        
        # Extract month code and year from ticker
        month_code = ticker[2]
        year = '20' + ticker[3:5]
        
        contracts.append({
            'ticker': ticker,
            'month_code': month_code,
            'year': year,
            'price': price,
            'date': date
        })
    
    if not contracts:
        logger.warning(f"No contract data found for date {date}")
        return pd.DataFrame()
    
    # Create DataFrame and sort by date
    term_df = pd.DataFrame(contracts)
    
    # Map month codes to months for sorting
    month_map = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
                 'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}
    
    # Add month number for sorting
    term_df['month_num'] = term_df['month_code'].map(month_map)
    
    # Create contract date for sorting
    term_df['contract_date'] = pd.to_datetime(term_df['year'] + '-' + term_df['month_num'].astype(str) + '-01')
    
    # Sort by contract date
    term_df = term_df.sort_values('contract_date')
    
    return term_df


def create_continuous_contract(
    futures_dict: Dict[str, pd.DataFrame],
    roll_days: int = 7,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Create a continuous futures contract by rolling over near-term contracts.
    
    Args:
        futures_dict: Dictionary mapping tickers to DataFrames
        roll_days: Number of days before expiration to roll to next contract
        output_path: Path to save continuous contract data
    
    Returns:
        DataFrame with continuous contract prices
    """
    if not futures_dict:
        logger.warning("No futures data provided")
        return pd.DataFrame()
    
    # Create empty dataframe for continuous contract
    cont_df = pd.DataFrame()
    
    # Map month codes to months for identifying contract order
    month_map = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
                 'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}
    
    # Parse ticker info
    contract_info = []
    for ticker, df in futures_dict.items():
        if df.empty:
            continue
        
        # Extract month code and year from ticker
        try:
            month_code = ticker[2]
            year = int('20' + ticker[3:5])
            month = month_map[month_code]
            
            # Calculate approximate expiration date (typically 3rd business day before 25th)
            # This is a simplification - actual expiration calendar should be used in production
            expiry_day = 25
            expiry = datetime(year, month, expiry_day)
            
            # Adjust for weekends - move to previous Friday if expiry is on weekend
            if expiry.weekday() >= 5:  # 5=Saturday, 6=Sunday
                expiry = expiry - timedelta(days=(expiry.weekday() - 4))
            
            # Calculate roll date
            roll_date = expiry - timedelta(days=roll_days)
            
            contract_info.append({
                'ticker': ticker,
                'df': df,
                'year': year,
                'month': month,
                'expiry': expiry,
                'roll_date': roll_date
            })
        
        except Exception as e:
            logger.warning(f"Could not parse contract info for {ticker}: {e}")
    
    if not contract_info:
        logger.warning("No valid contracts found")
        return pd.DataFrame()
    
    # Sort contracts by expiry date
    contract_info.sort(key=lambda x: x['expiry'])
    
    # Build continuous series
    for i, contract in enumerate(contract_info):
        df = contract['df']
        ticker = contract['ticker']
        
        if i == len(contract_info) - 1:
            # Last contract - use all available data
            cont_df = pd.concat([cont_df, df])
        else:
            # Use data up to roll date
            next_contract = contract_info[i+1]
            roll_date = contract['roll_date']
            
            # Filter data up to roll date
            mask = df.index <= roll_date
            df_to_use = df[mask]
            
            cont_df = pd.concat([cont_df, df_to_use])
    
    # Remove duplicates in case of overlapping dates
    cont_df = cont_df[~cont_df.index.duplicated(keep='first')]
    
    # Sort by date
    cont_df = cont_df.sort_index()
    
    # Add metadata
    cont_df['is_continuous'] = True
    
    # Save if output path provided
    if output_path:
        save_dir = Path(output_path).parent
        os.makedirs(save_dir, exist_ok=True)
        cont_df.to_csv(output_path)
        logger.info(f"Continuous contract saved to {output_path}")
    
    return cont_df


if __name__ == "__main__":
    # Example usage
    try:
        # Create data directories if they don't exist
        raw_data_dir = Path(__file__).parents[2] / 'data' / 'raw' / 'cme'
        os.makedirs(raw_data_dir, exist_ok=True)
        
        # Set date range (last 5 years)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = f"{datetime.now().year - 5}-01-01"
        
        logger.info(f"Fetching futures data from {start_date} to {end_date}")
        
        # Fetch active contract
        active_ticker = get_active_contract_ticker()
        active_contract_df = fetch_futures_data(
            ticker=active_ticker,
            start_date=start_date,
            end_date=end_date,
            save_path=raw_data_dir / f"{active_ticker.replace('.', '_')}.csv"
        )
        
        # Fetch futures curve data
        current_year = datetime.now().year
        futures_dict = {}
        
        # Fetch next 12 months of contracts
        for year in range(current_year, current_year + 2):
            year_dict = fetch_futures_curve(
                year=year,
                save_dir=raw_data_dir
            )
            futures_dict.update(year_dict)
        
        # Create continuous contract
        cont_df = create_continuous_contract(
            futures_dict,
            output_path=raw_data_dir / "ng_continuous.csv"
        )
        
        # Calculate current term structure
        if futures_dict:
            term_df = calculate_term_structure(futures_dict)
            term_df.to_csv(raw_data_dir / "term_structure.csv")
            logger.info(f"Term structure saved with {len(term_df)} contracts")
        
        logger.info("Futures data fetching completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise 
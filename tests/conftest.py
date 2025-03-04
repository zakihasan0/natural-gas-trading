"""
Conftest.py - Configuration and fixtures for pytest.
"""

import os
import sys
import pytest
import yaml

# Add the src directory to the Python path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def config():
    """Load test configuration."""
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_ng_data():
    """Return sample natural gas price data for testing."""
    import pandas as pd
    import numpy as np
    
    # Create a simple dataframe with random data
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='B')
    prices = np.random.normal(3.0, 0.2, size=len(dates))
    volumes = np.random.randint(1000, 5000, size=len(dates))
    
    df = pd.DataFrame({
        'date': dates,
        'price': prices,
        'volume': volumes
    }).set_index('date')
    
    return df 
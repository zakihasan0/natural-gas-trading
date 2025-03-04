#!/usr/bin/env python
"""
Natural Gas Trading System - Setup Script

This script handles the installation and setup of the Natural Gas Trading System.
It creates necessary directories, installs dependencies, and sets up configuration files.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import argparse
import yaml
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("setup")


def create_directory_structure():
    """Create the necessary directory structure for the project."""
    logger.info("Creating directory structure...")
    
    # Define directories to create
    directories = [
        "config",
        "data/raw/eia",
        "data/raw/noaa",
        "data/processed",
        "data/synthetic",
        "data/signals",
        "logs",
        "models",
        "notebooks",
        "src/data_ingestion",
        "src/data_processing",
        "src/strategies",
        "src/backtesting",
        "src/live_trading",
        "src/utils",
        "tests/unit",
        "tests/integration",
    ]
    
    # Create directories
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    logger.info("Directory structure created successfully.")


def install_dependencies(dev_mode=False):
    """Install required dependencies."""
    logger.info("Installing dependencies...")
    
    # Check if requirements.txt exists
    req_file = Path("requirements.txt")
    if not req_file.exists():
        logger.error("requirements.txt not found. Cannot install dependencies.")
        return False
    
    try:
        # Install dependencies
        cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        if dev_mode:
            # Add development dependencies
            cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-e", "."]
        
        logger.info(f"Running: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        
        logger.info("Dependencies installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing dependencies: {e}")
        return False


def create_credentials_template():
    """Create a template for the credentials.yaml file."""
    logger.info("Creating credentials template...")
    
    credentials_path = Path("config/credentials.yaml")
    
    # Don't overwrite existing credentials
    if credentials_path.exists():
        logger.info("Credentials file already exists. Skipping.")
        return
    
    # Create template
    credentials = {
        "noaa_token": "YOUR_NOAA_TOKEN_HERE",
        "eia_api_key": "YOUR_EIA_API_KEY_HERE",
    }
    
    # Write to file
    with open(credentials_path, 'w') as f:
        yaml.dump(credentials, f, default_flow_style=False)
    
    logger.info(f"Credentials template created at {credentials_path}")
    logger.info("Please update with your actual API keys.")


def create_config_file():
    """Create the main configuration file."""
    logger.info("Creating configuration file...")
    
    config_path = Path("config/config.yaml")
    
    # Don't overwrite existing config
    if config_path.exists():
        logger.info("Configuration file already exists. Skipping.")
        return
    
    # Create config
    config = {
        "data_sources": {
            "eia": {
                "base_url": "https://api.eia.gov/v2/",
                "endpoints": {
                    "prices": "natural-gas/pri/sum/data/",
                    "storage": "natural-gas/stor/wkly/data/"
                },
                "update_frequency": "daily"
            },
            "noaa": {
                "base_url": "https://www.ncdc.noaa.gov/cdo-web/api/v2/",
                "endpoints": {
                    "data": "data",
                    "stations": "stations",
                    "datasets": "datasets"
                },
                "default_station": "GHCND:USW00094728",  # Central Park, NY
                "update_frequency": "daily"
            }
        },
        "backtesting": {
            "default_initial_capital": 1000000,
            "default_transaction_cost": 0.0001,
            "default_slippage": 0.0001
        },
        "trading": {
            "market_hours": {
                "open_time": "09:00",
                "close_time": "17:30",
                "timezone": "America/New_York",
                "trading_days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            },
            "signal_generation": {
                "frequency": "daily",
                "time": "18:00"
            }
        },
        "logging": {
            "level": "INFO",
            "file_path": "logs/trading_system.log",
            "max_size_mb": 10,
            "backup_count": 5
        }
    }
    
    # Write to file
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Configuration file created at {config_path}")


def create_gitignore():
    """Create a .gitignore file."""
    logger.info("Creating .gitignore file...")
    
    gitignore_path = Path(".gitignore")
    
    # Don't overwrite existing .gitignore
    if gitignore_path.exists():
        logger.info(".gitignore file already exists. Skipping.")
        return
    
    # Create .gitignore content
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Jupyter Notebooks
.ipynb_checkpoints

# Project specific
config/credentials.yaml
data/raw/
data/processed/
logs/
models/

# OS specific
.DS_Store
Thumbs.db
"""
    
    # Write to file
    with open(gitignore_path, 'w') as f:
        f.write(gitignore_content)
    
    logger.info(f".gitignore file created at {gitignore_path}")


def create_init_files():
    """Create __init__.py files in all src directories."""
    logger.info("Creating __init__.py files...")
    
    # Find all directories under src
    src_dir = Path("src")
    if not src_dir.exists():
        logger.error("src directory not found. Cannot create __init__.py files.")
        return
    
    # Create __init__.py in each directory
    for dir_path in src_dir.glob("**"):
        if dir_path.is_dir():
            init_file = dir_path / "__init__.py"
            if not init_file.exists():
                with open(init_file, 'w') as f:
                    f.write(f'"""Natural Gas Trading System - {dir_path.name} module."""\n')
                logger.info(f"Created {init_file}")
    
    # Create root __init__.py
    root_init = src_dir / "__init__.py"
    if not root_init.exists():
        with open(root_init, 'w') as f:
            f.write('"""Natural Gas Trading System."""\n')
        logger.info(f"Created {root_init}")


def create_setup_cfg():
    """Create setup.cfg file for the project."""
    logger.info("Creating setup.cfg file...")
    
    setup_cfg_path = Path("setup.cfg")
    
    # Don't overwrite existing setup.cfg
    if setup_cfg_path.exists():
        logger.info("setup.cfg file already exists. Skipping.")
        return
    
    # Create setup.cfg content
    setup_cfg_content = """[metadata]
name = natural_gas_trading
version = 0.1.0
description = Natural Gas Trading System
author = Your Name
author_email = your.email@example.com
license = MIT

[options]
packages = find:
python_requires = >=3.8
install_requires =
    numpy>=1.22.0
    pandas>=1.4.0
    matplotlib>=3.5.0
    requests>=2.27.0
    pyyaml>=6.0.0

[options.packages.find]
exclude =
    tests*
    notebooks*

[flake8]
max-line-length = 100
exclude = .git,__pycache__,build,dist

[tool:pytest]
testpaths = tests
python_files = test_*.py
"""
    
    # Write to file
    with open(setup_cfg_path, 'w') as f:
        f.write(setup_cfg_content)
    
    logger.info(f"setup.cfg file created at {setup_cfg_path}")


def create_readme():
    """Create a README.md file."""
    logger.info("Creating README.md file...")
    
    readme_path = Path("README.md")
    
    # Don't overwrite existing README
    if readme_path.exists():
        logger.info("README.md file already exists. Skipping.")
        return
    
    # Create README content
    readme_content = """# Natural Gas Trading System

A quantitative trading system for natural gas markets, integrating weather data, storage levels, and price analysis.

## Features

- Data ingestion from EIA and NOAA APIs
- Weather-based trading strategies
- Storage-based trading strategies
- Technical analysis
- Backtesting engine
- Live trading signal generation
- Performance visualization
- Scheduled execution

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/natural-gas-trading.git
   cd natural-gas-trading
   ```

2. Run the setup script:
   ```
   python setup.py install
   ```

3. Configure your API credentials:
   - Edit `config/credentials.yaml` with your NOAA and EIA API keys

## Usage

### Running the System

To run the trading system with synthetic data:
```
python run_with_synthetic_data.py
```

To run the trading system with real API data:
```
python example_run.py --data real
```

To check system health:
```
python check_system.py
```

### Dashboard

To launch the dashboard:
```
streamlit run dashboard.py
```

### Scheduling

To schedule the trading system to run daily:
```
python schedule_trading_system.py --frequency daily
```

## Project Structure

- `config/`: Configuration files
- `data/`: Data storage
- `src/`: Source code
  - `data_ingestion/`: API connectors
  - `data_processing/`: Data pipeline
  - `strategies/`: Trading strategies
  - `backtesting/`: Backtesting engine
  - `live_trading/`: Live trading components
  - `utils/`: Utility functions
- `tests/`: Test suite
- `notebooks/`: Jupyter notebooks for analysis

## License

MIT License
"""
    
    # Write to file
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    logger.info(f"README.md file created at {readme_path}")


def main():
    """Main function to run the setup process."""
    parser = argparse.ArgumentParser(description='Setup the Natural Gas Trading System')
    parser.add_argument('--dev', action='store_true', help='Install in development mode')
    parser.add_argument('--skip-deps', action='store_true', help='Skip dependency installation')
    args = parser.parse_args()
    
    logger.info("Starting Natural Gas Trading System setup...")
    
    # Create directory structure
    create_directory_structure()
    
    # Create configuration files
    create_config_file()
    create_credentials_template()
    create_gitignore()
    create_setup_cfg()
    create_readme()
    
    # Create __init__.py files
    create_init_files()
    
    # Install dependencies
    if not args.skip_deps:
        install_dependencies(dev_mode=args.dev)
    
    logger.info("Setup completed successfully!")
    logger.info("Next steps:")
    logger.info("1. Update your API credentials in config/credentials.yaml")
    logger.info("2. Run 'python check_system.py' to verify your setup")
    logger.info("3. Try running 'python example_run.py' to test the system")


if __name__ == "__main__":
    main() 
# Natural Gas Trading System

A robust, quantitative natural gas trading system with data pipelines, strategy modules, backtesting, and live trading capabilities.

## Project Overview

This project implements a sophisticated natural gas trading system that combines fundamental data (EIA storage, production, consumption), weather data (NOAA forecasts), and technical indicators to generate alpha signals for natural gas futures trading.

### Key Features

- **Data Ingestion**: Automated pipelines for EIA, NOAA, and market data
- **Feature Engineering**: Weather-based features, storage analysis, and technical indicators
- **Strategy Development**: Modular alpha, risk, and portfolio construction components
- **Backtesting**: Vector-based backtesting engine with transaction costs and slippage
- **Analytics**: Performance analysis and strategy comparison tools
- **Live Trading**: Signal generation, trade recommendations, and position management

## Getting Started

### Prerequisites

- Python 3.8+
- Git

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/natural-gas-trading.git
   cd natural-gas-trading
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up configuration:
   ```
   cp config/credentials.yaml.example config/credentials.yaml
   # Edit credentials.yaml with your API keys
   ```

## API Setup

The system requires the following API keys to function properly:

1. **NOAA API** - For weather data
   - Register at https://www.ncdc.noaa.gov/cdo-web/token
   - Add your token to `config/credentials.yaml`

2. **EIA API** - For natural gas storage and price data
   - Register at https://www.eia.gov/opendata/register.php
   - Add your key to `config/credentials.yaml`

## Project Structure

```
natural-gas-trading/
├── config/                  # Configuration files
│   ├── config.yaml          # System configuration
│   └── credentials.yaml     # API keys (git-ignored)
├── data/                    # Data storage
│   ├── raw/                 # Raw data from APIs
│   ├── processed/           # Processed datasets
│   └── signals/             # Generated trading signals
├── src/                     # Source code
│   ├── data_ingestion/      # API connectors and data fetchers
│   ├── data_processing/     # Cleaning and feature engineering
│   ├── strategies/          # Trading strategies
│   ├── backtesting/         # Backtesting framework
│   ├── live_trading/        # Live trading components
│   ├── utils/               # Utilities
│   └── visualization/       # Visualizations and dashboards
├── notebooks/               # Jupyter notebooks for analysis
├── tests/                   # Unit and integration tests
├── logs/                    # System logs
└── run_trading_system.py    # Main entry point
```

## Usage

### Running Backtests

To run a backtest of the weather-storage strategy over 2 years:

```bash
python run_trading_system.py --mode backtest --strategy weather --years 2
```

To compare all strategies:

```bash
python run_trading_system.py --mode backtest --strategy ensemble --years 2
```

To run individual backtests for all strategies with plotting:

```bash
python run_trading_system.py --mode backtest --strategy all --years 2 --plot
```

### Running Live Trading

To generate a single trading signal:

```bash
python run_trading_system.py --mode live --once
```

To run the continuous signal generation service (60-minute intervals):

```bash
python run_trading_system.py --mode live --interval 60
```

## System Architecture

### Data Pipeline

1. **Data Ingestion**
   - NOAA weather data fetcher for temperature and precipitation
   - EIA natural gas storage and pricing data
   - Integration with market data sources

2. **Feature Engineering**
   - Weather features: Heating/Cooling Degree Days (HDD/CDD)
   - Storage deviation from seasonal norms
   - Price momentum and mean reversion indicators
   - Cross-asset signals

### Trading Strategies

1. **Weather-Storage Strategy**
   - Combines temperature anomalies with storage deviations
   - Seasonal adjustment based on time of year
   - Momentum overlay for trend confirmation

2. **Momentum Strategy**
   - Captures medium-term price trends
   - Volatility-adjusted position sizing

3. **Mean Reversion Strategy**
   - Identifies overbought/oversold conditions
   - Statistical significance testing

4. **Fundamental Strategy**
   - Storage-based signals
   - Supply/demand imbalance indicators
   - Seasonal components

### Risk Management

1. **Position Sizing**
   - Volatility-based sizing
   - Maximum position limits
   - Kelly criterion implementation

2. **Stop Loss**
   - Trailing stops
   - Volatility-based stops
   - Time-based exits

### Performance Metrics

- **Return Metrics**: Total return, annualized return
- **Risk Metrics**: Sharpe ratio, Sortino ratio, max drawdown
- **Trading Metrics**: Win rate, profit factor, average win/loss

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NOAA for weather data access
- EIA for natural gas market data
- CME Group for futures market information 
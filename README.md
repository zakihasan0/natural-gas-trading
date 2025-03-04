# Natural Gas Trading System

A robust, quantitative natural gas trading system with data pipelines, strategy modules, backtesting, and live trading capabilities.

## Project Overview

This project implements a sophisticated natural gas trading system that combines fundamental data (EIA storage, production, consumption), weather data (NOAA forecasts), and technical indicators to generate alpha signals for natural gas futures trading.

### Key Features

- **Data Ingestion**: Automated pipelines for EIA, NOAA, and market data
- **Strategy Development**: Modular alpha, risk, and portfolio construction components
- **Backtesting**: Vector-based and event-driven backtesting engines
- **Analytics**: Factor research and performance analysis tools
- **Live Trading**: Order execution, risk monitoring, and broker integration

## Getting Started

### Prerequisites

- Python 3.8+
- Git

### Installation

1. Clone the repository:
   ```
   git clone [your-repo-url]
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

## Project Structure

```
natural-gas-trading/
├── .github/workflows/       # CI/CD pipelines
├── config/                  # Configuration files
├── data/                    # Data directories
├── dags/                    # Airflow DAGs
├── src/                     # Source code
│   ├── data_ingestion/      # Data fetchers
│   ├── data_processing/     # Data cleaning and transformations
│   ├── analytics/           # Factor research
│   ├── strategies/          # Trading strategies
│   ├── backtesting/         # Backtesting engines
│   ├── live_trading/        # Live trading execution
│   └── utils/               # Utilities
├── notebooks/               # Jupyter notebooks
└── tests/                   # Test suite
```

## Usage

### Running Data Pipelines

```python
# Example code for running a data pipeline
from src.data_ingestion.eia_fetcher import fetch_eia_data

fetch_eia_data(start_date='2022-01-01', end_date='2022-12-31')
```

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/unit/test_basic.py

# Run tests with coverage report
python -m pytest --cov=src/ tests/
```

### Backtesting a Strategy

```python
# Example code for backtesting
from src.strategies.alpha_model import MomentumModel
from src.backtesting.vectorbt_runner import backtest_strategy

model = MomentumModel(lookback_period=20)
results = backtest_strategy(model, start_date='2022-01-01', end_date='2022-12-31')
results.plot()
```

## Development Status

This project is currently in active development. The following components have been implemented:

- [x] Project structure and environment setup
- [x] Basic configuration files
- [x] EIA data fetcher module
- [x] Logging utilities
- [x] Unit testing framework
- [ ] Weather data fetcher
- [ ] Futures data fetcher
- [ ] Data processing modules
- [ ] Strategy implementation
- [ ] Backtesting engine
- [ ] Live trading integration

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request

## License

[Your License] - See the LICENSE file for details.

## Acknowledgments

- Data sources: EIA, NOAA, CME Group
- Tools: VectorBT, Pandas, Scikit-learn 
"""
Alpha Model - Trading strategies for natural gas.

This module implements various trading strategies:
1. Momentum strategy
2. Mean reversion strategy
3. Fundamental strategy (using storage, weather, and production data)
4. Combination strategy
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import os
import logging
from abc import ABC, abstractmethod
from datetime import datetime

# Import utility logger
from src.utils.logger import get_logger

# Configure logging
logger = get_logger(__name__)


class AlphaModel(ABC):
    """
    Abstract base class for all alpha models.
    """
    
    def __init__(self, name: str):
        """
        Initialize the alpha model.
        
        Args:
            name: Name of the model
        """
        self.name = name
        logger.info(f"Initialized {self.name} alpha model")
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals from the given data.
        
        Args:
            data: DataFrame with market data and features
        
        Returns:
            Series with trading signals (values between -1 and 1)
        """
        pass
    
    def normalize_signals(self, signals: pd.Series) -> pd.Series:
        """
        Normalize signals to be between -1 and 1.
        
        Args:
            signals: Series with raw signals
        
        Returns:
            Series with normalized signals
        """
        # If signals are already between -1 and 1, return as is
        if signals.min() >= -1 and signals.max() <= 1:
            return signals
        
        # Otherwise, normalize using min-max scaling
        min_val = signals.min()
        max_val = signals.max()
        
        # If min equals max, return neutral signals
        if min_val == max_val:
            return pd.Series(0, index=signals.index)
        
        # Scale to [-1, 1]
        normalized = 2 * (signals - min_val) / (max_val - min_val) - 1
        
        return normalized
    
    def combine_signals(self, signals_list: List[pd.Series], weights: Optional[List[float]] = None) -> pd.Series:
        """
        Combine multiple signals into a single signal.
        
        Args:
            signals_list: List of signal Series
            weights: List of weights for each signal. If None, equal weights are used.
        
        Returns:
            Series with combined signals
        """
        if not signals_list:
            logger.warning("No signals to combine")
            return pd.Series()
        
        # Use equal weights if not provided
        if weights is None:
            weights = [1.0 / len(signals_list)] * len(signals_list)
        
        # Check if weights sum to 1
        if abs(sum(weights) - 1.0) > 1e-6:
            logger.warning(f"Weights do not sum to 1 (sum={sum(weights)}), normalizing")
            weights = [w / sum(weights) for w in weights]
        
        # Get common index
        common_index = signals_list[0].index
        for signals in signals_list[1:]:
            common_index = common_index.intersection(signals.index)
        
        # Initialize combined signals
        combined = pd.Series(0.0, index=common_index)
        
        # Add weighted signals
        for signals, weight in zip(signals_list, weights):
            signals_aligned = signals.reindex(common_index)
            combined += signals_aligned * weight
        
        return combined


class MomentumModel(AlphaModel):
    """
    Momentum-based trading strategy.
    
    This strategy generates buy signals when price momentum is positive
    and sell signals when momentum is negative.
    """
    
    def __init__(
        self,
        lookback_period: int = 20,
        volatility_lookback: int = 20,
        price_col: str = 'close',
        normalize_by_volatility: bool = True
    ):
        """
        Initialize the momentum model.
        
        Args:
            lookback_period: Number of periods to use for calculating momentum
            volatility_lookback: Number of periods to use for volatility calculation
            price_col: Column name for price data
            normalize_by_volatility: Whether to normalize signals by volatility
        """
        super().__init__(name=f"Momentum_{lookback_period}")
        self.lookback_period = lookback_period
        self.volatility_lookback = volatility_lookback
        self.price_col = price_col
        self.normalize_by_volatility = normalize_by_volatility
        
        logger.info(f"Momentum model parameters: lookback={lookback_period}, "
                   f"vol_lookback={volatility_lookback}, normalize={normalize_by_volatility}")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate momentum signals from the given data.
        
        Args:
            data: DataFrame with market data including price
        
        Returns:
            Series with momentum signals (-1 to 1)
        """
        if self.price_col not in data.columns:
            logger.error(f"Price column '{self.price_col}' not found in data")
            return pd.Series(index=data.index)
        
        # Calculate momentum
        momentum = data[self.price_col].pct_change(periods=self.lookback_period)
        
        # Optionally normalize by volatility
        if self.normalize_by_volatility:
            # Calculate rolling volatility
            volatility = data[self.price_col].pct_change().rolling(window=self.volatility_lookback).std()
            
            # Avoid division by zero
            volatility = volatility.replace(0, np.nan).fillna(volatility.median())
            
            # Normalize momentum by volatility
            signals = momentum / volatility
        else:
            signals = momentum
        
        # Normalize signals to [-1, 1] range
        signals = self.normalize_signals(signals)
        
        # Fill NaN values at the beginning
        signals = signals.fillna(0)
        
        logger.info(f"Generated momentum signals with mean={signals.mean():.4f}, std={signals.std():.4f}")
        return signals


class MeanReversionModel(AlphaModel):
    """
    Mean reversion trading strategy.
    
    This strategy generates buy signals when price is below its moving average
    and sell signals when price is above its moving average.
    """
    
    def __init__(
        self,
        lookback_period: int = 20,
        z_score_threshold: float = 2.0,
        price_col: str = 'close'
    ):
        """
        Initialize the mean reversion model.
        
        Args:
            lookback_period: Number of periods to use for moving average
            z_score_threshold: Z-score threshold for generating signals
            price_col: Column name for price data
        """
        super().__init__(name=f"MeanReversion_{lookback_period}")
        self.lookback_period = lookback_period
        self.z_score_threshold = z_score_threshold
        self.price_col = price_col
        
        logger.info(f"Mean reversion model parameters: lookback={lookback_period}, "
                   f"z_score_threshold={z_score_threshold}")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate mean reversion signals from the given data.
        
        Args:
            data: DataFrame with market data including price
        
        Returns:
            Series with mean reversion signals (-1 to 1)
        """
        if self.price_col not in data.columns:
            logger.error(f"Price column '{self.price_col}' not found in data")
            return pd.Series(index=data.index)
        
        # Calculate moving average
        ma = data[self.price_col].rolling(window=self.lookback_period).mean()
        
        # Calculate standard deviation
        std = data[self.price_col].rolling(window=self.lookback_period).std()
        
        # Calculate z-score (how many standard deviations from the mean)
        z_score = (data[self.price_col] - ma) / std
        
        # Generate signals based on z-score
        signals = -z_score / self.z_score_threshold
        
        # Clip signals to [-1, 1] range
        signals = signals.clip(-1, 1)
        
        # Fill NaN values at the beginning
        signals = signals.fillna(0)
        
        logger.info(f"Generated mean reversion signals with mean={signals.mean():.4f}, std={signals.std():.4f}")
        return signals


class RSIModel(AlphaModel):
    """
    Relative Strength Index (RSI) trading strategy.
    
    This strategy generates buy signals when RSI is below the oversold level
    and sell signals when RSI is above the overbought level.
    """
    
    def __init__(
        self,
        rsi_period: int = 14,
        oversold_threshold: float = 30,
        overbought_threshold: float = 70,
        rsi_col: str = 'rsi_14',
        price_col: str = 'close'
    ):
        """
        Initialize the RSI model.
        
        Args:
            rsi_period: Number of periods to use for RSI calculation
            oversold_threshold: RSI level below which to generate buy signals
            overbought_threshold: RSI level above which to generate sell signals
            rsi_col: Column name for RSI data (if already calculated)
            price_col: Column name for price data (if RSI needs to be calculated)
        """
        super().__init__(name=f"RSI_{rsi_period}")
        self.rsi_period = rsi_period
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
        self.rsi_col = rsi_col
        self.price_col = price_col
        
        logger.info(f"RSI model parameters: period={rsi_period}, "
                   f"oversold={oversold_threshold}, overbought={overbought_threshold}")
    
    def calculate_rsi(self, price_series: pd.Series) -> pd.Series:
        """
        Calculate the RSI indicator.
        
        Args:
            price_series: Series with price data
        
        Returns:
            Series with RSI values
        """
        # Calculate price changes
        delta = price_series.diff()
        
        # Get positive and negative price changes
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        
        # Calculate relative strength
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate RSI signals from the given data.
        
        Args:
            data: DataFrame with market data including RSI or price
        
        Returns:
            Series with RSI signals (-1 to 1)
        """
        # Use provided RSI column if available, otherwise calculate it
        if self.rsi_col in data.columns:
            rsi = data[self.rsi_col]
        elif self.price_col in data.columns:
            rsi = self.calculate_rsi(data[self.price_col])
        else:
            logger.error(f"Neither RSI column '{self.rsi_col}' nor price column '{self.price_col}' found in data")
            return pd.Series(index=data.index)
        
        # Generate signals based on RSI levels
        signals = pd.Series(0, index=rsi.index)
        
        # Buy signals when RSI is below oversold threshold
        signals[rsi < self.oversold_threshold] = 1
        
        # Sell signals when RSI is above overbought threshold
        signals[rsi > self.overbought_threshold] = -1
        
        # Scale signals between oversold and overbought thresholds
        mask = (rsi >= self.oversold_threshold) & (rsi <= self.overbought_threshold)
        signals[mask] = (50 - rsi[mask]) / (self.overbought_threshold - self.oversold_threshold) * 2
        
        # Fill NaN values at the beginning
        signals = signals.fillna(0)
        
        logger.info(f"Generated RSI signals with mean={signals.mean():.4f}, std={signals.std():.4f}")
        return signals


class FundamentalModel(AlphaModel):
    """
    Fundamental trading strategy based on natural gas storage, weather, and production data.
    
    This strategy generates signals based on deviations from normal levels of storage,
    weather, and production.
    """
    
    def __init__(
        self,
        storage_weight: float = 0.4,
        weather_weight: float = 0.3,
        production_weight: float = 0.3,
        storage_col: str = 'storage_deviation_pct',
        weather_col: str = 'weather_HTDD_anomaly',
        production_col: str = 'production_yoy_change_pct'
    ):
        """
        Initialize the fundamental model.
        
        Args:
            storage_weight: Weight for storage signals
            weather_weight: Weight for weather signals
            production_weight: Weight for production signals
            storage_col: Column name for storage deviation data
            weather_col: Column name for weather anomaly data
            production_col: Column name for production year-over-year change data
        """
        super().__init__(name="Fundamental")
        self.storage_weight = storage_weight
        self.weather_weight = weather_weight
        self.production_weight = production_weight
        
        self.storage_col = storage_col
        self.weather_col = weather_col
        self.production_col = production_col
        
        logger.info(f"Fundamental model weights: storage={storage_weight}, "
                   f"weather={weather_weight}, production={production_weight}")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate fundamental signals from the given data.
        
        Args:
            data: DataFrame with market data including fundamental indicators
        
        Returns:
            Series with fundamental signals (-1 to 1)
        """
        signals = pd.Series(0, index=data.index)
        weights_used = 0
        
        # Generate storage signals
        if self.storage_col in data.columns:
            # Lower than normal storage is bullish (positive signal)
            storage_signals = -self.normalize_signals(data[self.storage_col])
            signals += storage_signals * self.storage_weight
            weights_used += self.storage_weight
        else:
            logger.warning(f"Storage column '{self.storage_col}' not found in data")
        
        # Generate weather signals (heating degree days anomaly)
        if self.weather_col in data.columns:
            # Higher than normal HDDs (colder) is bullish for natural gas
            weather_signals = self.normalize_signals(data[self.weather_col])
            
            # Only apply during heating season
            if 'heating_season' in data.columns:
                weather_signals = weather_signals * data['heating_season']
            
            signals += weather_signals * self.weather_weight
            weights_used += self.weather_weight
        else:
            logger.warning(f"Weather column '{self.weather_col}' not found in data")
        
        # Generate production signals
        if self.production_col in data.columns:
            # Higher than normal production is bearish (negative signal)
            production_signals = -self.normalize_signals(data[self.production_col])
            signals += production_signals * self.production_weight
            weights_used += self.production_weight
        else:
            logger.warning(f"Production column '{self.production_col}' not found in data")
        
        # Renormalize if not all data sources were available
        if weights_used > 0 and weights_used < 1:
            signals = signals / weights_used
        
        # Ensure signals are in [-1, 1] range
        signals = signals.clip(-1, 1)
        
        # Fill NaN values
        signals = signals.fillna(0)
        
        logger.info(f"Generated fundamental signals with mean={signals.mean():.4f}, std={signals.std():.4f}")
        return signals


class SeasonalModel(AlphaModel):
    """
    Seasonal trading strategy based on historical price patterns.
    
    This strategy generates signals based on seasonal patterns in natural gas prices.
    """
    
    def __init__(
        self,
        month_weights: Optional[Dict[int, float]] = None,
    ):
        """
        Initialize the seasonal model.
        
        Args:
            month_weights: Dictionary mapping months (1-12) to weights (-1 to 1)
                          If None, default seasonal pattern is used
        """
        super().__init__(name="Seasonal")
        
        # Default seasonal pattern for natural gas
        # Typically bullish in winter, bearish in spring/summer
        if month_weights is None:
            self.month_weights = {
                1: 0.8,    # January - winter, high demand
                2: 0.6,    # February - winter, high demand
                3: 0.0,    # March - transition month
                4: -0.4,   # April - spring, low demand
                5: -0.6,   # May - spring, low demand, injection season
                6: -0.6,   # June - summer, low heating demand
                7: -0.2,   # July - summer, some cooling demand
                8: 0.0,    # August - summer, some cooling demand
                9: 0.2,    # September - transition month
                10: 0.4,   # October - fall, start of heating season
                11: 0.6,   # November - fall, heating season
                12: 0.8    # December - winter, high demand
            }
        else:
            self.month_weights = month_weights
        
        logger.info(f"Seasonal model initialized with month weights: {self.month_weights}")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate seasonal signals from the given data.
        
        Args:
            data: DataFrame with market data (index should be DatetimeIndex)
        
        Returns:
            Series with seasonal signals (-1 to 1)
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            logger.error("Data index is not DatetimeIndex, converting")
            try:
                index = pd.DatetimeIndex(data.index)
            except:
                logger.error("Could not convert index to DatetimeIndex")
                return pd.Series(0, index=data.index)
        else:
            index = data.index
        
        # Extract month from dates
        months = index.month
        
        # Generate signals based on month weights
        signals = pd.Series([self.month_weights.get(month, 0) for month in months], index=index)
        
        logger.info(f"Generated seasonal signals with mean={signals.mean():.4f}, std={signals.std():.4f}")
        return signals


class CombinationModel(AlphaModel):
    """
    Combination of multiple alpha models.
    
    This strategy combines signals from multiple alpha models with specified weights.
    """
    
    def __init__(
        self,
        models: List[AlphaModel],
        weights: Optional[List[float]] = None,
        name: str = "Combination"
    ):
        """
        Initialize the combination model.
        
        Args:
            models: List of alpha models to combine
            weights: List of weights for each model. If None, equal weights are used.
            name: Name of the model
        """
        super().__init__(name=name)
        self.models = models
        
        # Use equal weights if not provided
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            # Check if weights sum to 1
            if abs(sum(weights) - 1.0) > 1e-6:
                logger.warning(f"Weights do not sum to 1 (sum={sum(weights)}), normalizing")
                self.weights = [w / sum(weights) for w in weights]
            else:
                self.weights = weights
        
        logger.info(f"Combination model initialized with {len(models)} models")
        for i, (model, weight) in enumerate(zip(self.models, self.weights)):
            logger.info(f"  Model {i+1}: {model.name} (weight={weight:.2f})")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate combined signals from the given data.
        
        Args:
            data: DataFrame with market data
        
        Returns:
            Series with combined signals (-1 to 1)
        """
        signals_list = []
        
        # Generate signals from each model
        for model in self.models:
            model_signals = model.generate_signals(data)
            signals_list.append(model_signals)
        
        # Combine signals using weights
        combined_signals = self.combine_signals(signals_list, self.weights)
        
        logger.info(f"Generated combined signals with mean={combined_signals.mean():.4f}, "
                   f"std={combined_signals.std():.4f}")
        return combined_signals


def create_default_models() -> Dict[str, AlphaModel]:
    """
    Create a dictionary of default alpha models.
    
    Returns:
        Dictionary mapping model names to AlphaModel instances
    """
    models = {}
    
    # Create momentum models with different lookback periods
    for lookback in [5, 10, 20, 60]:
        model = MomentumModel(lookback_period=lookback)
        models[model.name] = model
    
    # Create mean reversion models with different lookback periods
    for lookback in [5, 10, 20]:
        model = MeanReversionModel(lookback_period=lookback)
        models[model.name] = model
    
    # Create RSI model
    models["RSI_14"] = RSIModel()
    
    # Create fundamental model
    models["Fundamental"] = FundamentalModel()
    
    # Create seasonal model
    models["Seasonal"] = SeasonalModel()
    
    # Create combination model with all models
    all_models = list(models.values())
    models["Combination"] = CombinationModel(all_models)
    
    # Create combination model with technical indicators only
    tech_models = [m for m in all_models if not isinstance(m, (FundamentalModel, SeasonalModel))]
    models["Technical"] = CombinationModel(tech_models, name="Technical")
    
    # Create combination model with fundamental and seasonal indicators only
    fund_models = [m for m in all_models if isinstance(m, (FundamentalModel, SeasonalModel))]
    models["Fundamental_Seasonal"] = CombinationModel(fund_models, name="Fundamental_Seasonal")
    
    return models


if __name__ == "__main__":
    # Example usage
    try:
        # Load sample data (simulated here)
        index = pd.date_range(start='2020-01-01', end='2023-01-01', freq='D')
        np.random.seed(42)
        
        # Create sample price data with seasonal pattern
        trend = np.linspace(0, 10, len(index))
        seasonal = 2 * np.sin(2 * np.pi * np.arange(len(index)) / 365)
        noise = np.random.normal(0, 1, len(index))
        
        price = 10 + trend + seasonal + noise
        
        # Create sample fundamental data
        storage_deviation = -5 * np.sin(2 * np.pi * np.arange(len(index)) / 365) + np.random.normal(0, 2, len(index))
        weather_anomaly = 3 * np.sin(2 * np.pi * np.arange(len(index)) / 365 + np.pi/2) + np.random.normal(0, 1, len(index))
        production_change = np.random.normal(0, 3, len(index))
        
        # Create DataFrame
        data = pd.DataFrame({
            'close': price,
            'storage_deviation_pct': storage_deviation,
            'weather_HTDD_anomaly': weather_anomaly,
            'production_yoy_change_pct': production_change,
            'heating_season': np.where(np.array([d.month for d in index]).reshape(-1) in [1, 2, 3, 10, 11, 12], 1, 0)
        }, index=index)
        
        # Calculate RSI (just for the example)
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        data['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Create models
        models = create_default_models()
        
        # Generate signals for each model
        signals = {}
        for name, model in models.items():
            signals[name] = model.generate_signals(data)
        
        # Create DataFrame with all signals
        signals_df = pd.DataFrame(signals)
        
        # Print statistics
        print("Signal Statistics:")
        print(signals_df.describe().T[['mean', 'std', 'min', 'max']])
        
        # Save signals to CSV
        output_dir = Path(__file__).parents[2] / 'data' / 'interim'
        os.makedirs(output_dir, exist_ok=True)
        signals_df.to_csv(output_dir / 'example_signals.csv')
        
        logger.info(f"Example signals saved to {output_dir / 'example_signals.csv'}")
        
    except Exception as e:
        logger.error(f"Error in example: {e}")
        raise 
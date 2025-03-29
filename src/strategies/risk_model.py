"""
Risk Model - Position sizing and risk management for natural gas trading.

This module implements various risk management approaches:
1. Fixed position sizing
2. Volatility-based position sizing
3. Stop-loss and take-profit rules
4. Maximum drawdown control
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import os
import logging
from datetime import datetime

# Import utility logger
from src.utils.logger import get_logger

# Configure logging
logger = get_logger(__name__)


class RiskModel:
    """
    Base risk model for managing position sizes and risk constraints.
    """
    
    def __init__(self, max_position_size: float = 1.0, max_leverage: float = 1.0):
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage
        self.logger = get_logger(__name__)

    def _get_price_column(self, data: pd.DataFrame) -> str:
        """Get the appropriate price column name."""
        if 'close' in data.columns:
            return 'close'
        elif 'price' in data.columns:
            return 'price'
        else:
            raise ValueError("No price column ('close' or 'price') found in data")

    def _calculate_returns(self, data: pd.DataFrame) -> pd.Series:
        """Calculate returns from price data."""
        price_col = self._get_price_column(data)
        return data[price_col].pct_change()

    def size_position(self, positions: pd.Series, data: pd.DataFrame) -> pd.Series:
        """Base position sizing method."""
        return positions.clip(-self.max_position_size, self.max_position_size)


class VolatilityRisk(RiskModel):
    """
    Volatility-based risk model that adjusts position sizes based on recent volatility.
    """
    
    def __init__(self, target_volatility: float = 0.15, lookback: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.target_volatility = target_volatility
        self.lookback = lookback

    def size_position(self, positions: pd.Series, data: pd.DataFrame) -> pd.Series:
        """Size positions based on volatility."""
        try:
            # Calculate returns if not in data
            if 'returns' not in data.columns:
                returns = self._calculate_returns(data)
            else:
                returns = data['returns']

            # Calculate rolling volatility
            vol = returns.rolling(window=self.lookback).std() * np.sqrt(252)
            
            # Scale positions by volatility
            vol_adj_positions = positions * (self.target_volatility / vol.fillna(self.target_volatility))
            
            # Apply base class position limits
            sized_positions = super().size_position(vol_adj_positions, data)
            
            self.logger.info(f"Sized volatility-adjusted positions with mean={sized_positions.mean():.4f}, std={sized_positions.std():.4f}")
            return sized_positions
            
        except Exception as e:
            self.logger.error(f"Error in volatility-based position sizing: {str(e)}")
            return positions


class StopLossRisk(RiskModel):
    """
    Risk model that applies stop-loss and take-profit rules to positions.
    """
    
    def __init__(self, stop_loss: float = 0.05, take_profit: float = 0.10, **kwargs):
        super().__init__(**kwargs)
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def size_position(self, positions: pd.Series, data: pd.DataFrame) -> pd.Series:
        """Apply stop loss and take profit to positions."""
        try:
            # Calculate returns if not in data
            if 'returns' not in data.columns:
                returns = self._calculate_returns(data)
            else:
                returns = data['returns']

            # Calculate cumulative returns for each position
            cum_returns = returns.cumsum()
            
            # Apply stop loss and take profit
            stop_loss_mask = cum_returns < -self.stop_loss
            take_profit_mask = cum_returns > self.take_profit
            
            # Zero out positions that hit stops
            positions = positions.copy()
            positions[stop_loss_mask | take_profit_mask] = 0
            
            # Apply base class position limits
            sized_positions = super().size_position(positions, data)
            
            self.logger.info(f"Sized stop-loss adjusted positions with mean={sized_positions.mean():.4f}, std={sized_positions.std():.4f}")
            return sized_positions
            
        except Exception as e:
            self.logger.error(f"Error in stop-loss position sizing: {str(e)}")
            return positions


class DrawdownControlRisk(RiskModel):
    """
    Risk model that dynamically adjusts position sizes based on recent drawdowns.
    """
    
    def __init__(self, max_drawdown: float = 0.15, scaling_factor: float = 2.0, lookback_window: int = 60, **kwargs):
        super().__init__(**kwargs)
        self.max_drawdown = max_drawdown
        self.scaling_factor = scaling_factor
        self.lookback_window = lookback_window

    def size_position(self, positions: pd.Series, data: pd.DataFrame) -> pd.Series:
        """Size positions based on drawdown control."""
        try:
            # Calculate returns if not in data
            if 'returns' not in data.columns:
                returns = self._calculate_returns(data)
            else:
                returns = data['returns']

            # Calculate rolling drawdown
            cum_returns = (1 + returns).cumprod()
            rolling_max = cum_returns.rolling(window=self.lookback_window, min_periods=1).max()
            drawdown = (cum_returns - rolling_max) / rolling_max
            
            # Scale positions based on drawdown
            scale = 1 - (abs(drawdown) / self.max_drawdown) * self.scaling_factor
            scale = scale.clip(0, 1)
            
            # Apply scaling to positions
            scaled_positions = positions * scale
            
            # Apply base class position limits
            sized_positions = super().size_position(scaled_positions, data)
            
            self.logger.info(f"Sized drawdown-controlled positions with mean={sized_positions.mean():.4f}, std={sized_positions.std():.4f}")
            return sized_positions
            
        except Exception as e:
            self.logger.error(f"Error in drawdown control position sizing: {str(e)}")
            return positions


class CompositeRisk(RiskModel):
    """
    Composite risk model that combines multiple risk models.
    """
    
    def __init__(self, risk_models: List[RiskModel], **kwargs):
        super().__init__(**kwargs)
        self.risk_models = risk_models
        self.name = "CompositeRisk_" + "+".join([model.__class__.__name__ for model in risk_models])
        self.logger.info(f"Initialized {self.name} risk model with max_position_size={self.max_position_size}, max_leverage={self.max_leverage}")
        self.logger.info(f"Composite risk model initialized with {len(risk_models)} models: {[model.__class__.__name__ for model in risk_models]}")

    def size_position(self, positions: pd.Series, data: pd.DataFrame) -> pd.Series:
        """Apply all risk models in sequence."""
        try:
            # Calculate returns if not in data
            if 'returns' not in data.columns:
                data = data.copy()
                data['returns'] = self._calculate_returns(data)

            # Apply each risk model in sequence
            for model in self.risk_models:
                positions = model.size_position(positions, data)
            
            # Apply final base class position limits
            return super().size_position(positions, data)
            
        except Exception as e:
            self.logger.error(f"Error in composite position sizing: {str(e)}")
            return positions


def create_default_risk_model(config: Optional[Dict] = None) -> RiskModel:
    """
    Create a default risk model with parameters from config.
    
    Args:
        config: Configuration dictionary with risk parameters
    
    Returns:
        RiskModel instance
    """
    # Default parameters
    max_position_size = 1.0
    max_leverage = 1.0
    target_volatility = 0.15
    volatility_lookback = 20
    stop_loss_pct = 0.05
    max_drawdown = 0.15
    
    # Override with config if provided
    if config and 'risk' in config:
        risk_config = config['risk']
        max_position_size = risk_config.get('max_position_size_pct', max_position_size)
        max_leverage = risk_config.get('max_leverage', max_leverage)
        target_volatility = risk_config.get('target_volatility', target_volatility)
        volatility_lookback = risk_config.get('volatility_lookback', volatility_lookback)
        stop_loss_pct = risk_config.get('stop_loss_pct', stop_loss_pct)
        max_drawdown = risk_config.get('max_drawdown_pct', max_drawdown)
    
    # Create individual risk models
    volatility_risk = VolatilityRisk(
        target_volatility=target_volatility,
        lookback=volatility_lookback,
        max_position_size=max_position_size,
        max_leverage=max_leverage
    )
    
    stop_loss_risk = StopLossRisk(
        stop_loss=stop_loss_pct,
        take_profit=stop_loss_pct*2,  # Typically 2:1 reward:risk ratio
        max_position_size=max_position_size,
        max_leverage=max_leverage
    )
    
    drawdown_risk = DrawdownControlRisk(
        max_drawdown=max_drawdown,
        scaling_factor=2.0,
        lookback_window=60,
        max_position_size=max_position_size,
        max_leverage=max_leverage
    )
    
    # Create composite risk model
    composite_risk = CompositeRisk(
        risk_models=[volatility_risk, stop_loss_risk, drawdown_risk],
        max_position_size=max_position_size,
        max_leverage=max_leverage
    )
    
    return composite_risk


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
        
        # Create sample alpha signals
        alpha_signals = pd.Series(np.sin(2 * np.pi * np.arange(len(index)) / 180), index=index)
        
        # Create DataFrame
        data = pd.DataFrame({
            'close': price
        }, index=index)
        
        # Calculate returns
        returns = data['close'].pct_change()
        
        # Create risk models
        basic_risk = RiskModel(max_position_size=1.0)
        vol_risk = VolatilityRisk(target_volatility=0.15, lookback=20)
        stop_loss_risk = StopLossRisk(stop_loss=0.05, take_profit=0.10)
        drawdown_risk = DrawdownControlRisk(max_drawdown=0.15)
        
        # Size positions with each risk model
        basic_positions = basic_risk.size_position(alpha_signals, data)
        vol_positions = vol_risk.size_position(alpha_signals, data)
        stop_loss_positions = stop_loss_risk.size_position(alpha_signals, data)
        drawdown_positions = drawdown_risk.size_position(alpha_signals, returns)
        
        # Create composite risk model
        composite_risk = CompositeRisk(
            risk_models=[vol_risk, stop_loss_risk, drawdown_risk],
            max_position_size=1.0
        )
        composite_positions = composite_risk.size_position(alpha_signals, data)
        
        # Create DataFrame with all positions
        positions_df = pd.DataFrame({
            'alpha_signals': alpha_signals,
            'basic': basic_positions,
            'volatility': vol_positions,
            'stop_loss': stop_loss_positions,
            'drawdown': drawdown_positions,
            'composite': composite_positions
        })
        
        # Print statistics
        print("Position Statistics:")
        print(positions_df.describe().T[['mean', 'std', 'min', 'max']])
        
        # Save positions to CSV
        output_dir = Path(__file__).parents[2] / 'data' / 'interim'
        os.makedirs(output_dir, exist_ok=True)
        positions_df.to_csv(output_dir / 'example_positions.csv')
        
        logger.info(f"Example positions saved to {output_dir / 'example_positions.csv'}")
        
    except Exception as e:
        logger.error(f"Error in example: {e}")
        raise 
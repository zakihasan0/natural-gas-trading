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
    
    def __init__(
        self,
        max_position_size: float = 1.0,
        max_leverage: float = 1.0,
        name: str = "DefaultRisk"
    ):
        """
        Initialize the risk model.
        
        Args:
            max_position_size: Maximum allowed position size (0.0 to 1.0)
            max_leverage: Maximum allowed leverage
            name: Name of the risk model
        """
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage
        self.name = name
        
        logger.info(f"Initialized {self.name} risk model with max_position_size={max_position_size}, "
                   f"max_leverage={max_leverage}")
    
    def size_position(self, alpha_signals: pd.Series) -> pd.Series:
        """
        Calculate position sizes from alpha signals, applying risk constraints.
        
        Args:
            alpha_signals: Series with alpha signals (-1 to 1)
        
        Returns:
            Series with position sizes scaled by risk constraints
        """
        # Simply scale alpha signals by max position size
        positions = alpha_signals * self.max_position_size
        
        logger.info(f"Sized positions with mean={positions.mean():.4f}, std={positions.std():.4f}")
        return positions


class VolatilityBasedRisk(RiskModel):
    """
    Volatility-based risk model that adjusts position sizes based on recent volatility.
    """
    
    def __init__(
        self,
        target_volatility: float = 0.15,
        volatility_lookback: int = 20,
        max_position_size: float = 1.0,
        max_leverage: float = 1.0,
        price_col: str = 'close',
        vol_floor: float = 0.05
    ):
        """
        Initialize the volatility-based risk model.
        
        Args:
            target_volatility: Target annualized portfolio volatility
            volatility_lookback: Number of periods to use for volatility calculation
            max_position_size: Maximum allowed position size (0.0 to 1.0)
            max_leverage: Maximum allowed leverage
            price_col: Column name for price data
            vol_floor: Minimum volatility to use (to avoid division by zero)
        """
        super().__init__(max_position_size, max_leverage, name=f"VolatilityRisk_{volatility_lookback}")
        self.target_volatility = target_volatility
        self.volatility_lookback = volatility_lookback
        self.price_col = price_col
        self.vol_floor = vol_floor
        
        logger.info(f"Volatility risk model parameters: target_vol={target_volatility}, "
                   f"lookback={volatility_lookback}")
    
    def calculate_volatility(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate historical volatility from price data.
        
        Args:
            data: DataFrame with market data including price
        
        Returns:
            Series with rolling volatility estimates (annualized)
        """
        if self.price_col not in data.columns:
            logger.error(f"Price column '{self.price_col}' not found in data")
            return pd.Series(self.vol_floor, index=data.index)
        
        # Calculate daily returns
        returns = data[self.price_col].pct_change()
        
        # Calculate rolling volatility (standard deviation of returns)
        volatility = returns.rolling(window=self.volatility_lookback).std()
        
        # Annualize volatility (assuming 252 trading days per year)
        annualized_volatility = volatility * np.sqrt(252)
        
        # Apply a floor to avoid division by zero
        annualized_volatility = annualized_volatility.clip(lower=self.vol_floor)
        
        return annualized_volatility
    
    def size_position(self, alpha_signals: pd.Series, data: pd.DataFrame) -> pd.Series:
        """
        Calculate position sizes from alpha signals, scaling by volatility.
        
        Args:
            alpha_signals: Series with alpha signals (-1 to 1)
            data: DataFrame with market data for volatility calculation
        
        Returns:
            Series with position sizes scaled by volatility
        """
        # Calculate volatility
        volatility = self.calculate_volatility(data)
        
        # Scale positions inversely with volatility
        # Higher volatility -> smaller positions
        vol_scalar = self.target_volatility / volatility
        
        # Apply max leverage constraint
        vol_scalar = vol_scalar.clip(upper=self.max_leverage)
        
        # Scale alpha signals by volatility scalar and max position size
        positions = alpha_signals * vol_scalar * self.max_position_size
        
        # Ensure positions are within allowed limits
        positions = positions.clip(lower=-self.max_position_size, upper=self.max_position_size)
        
        logger.info(f"Sized volatility-adjusted positions with mean={positions.mean():.4f}, "
                   f"std={positions.std():.4f}")
        return positions


class StopLossRisk(RiskModel):
    """
    Risk model that applies stop-loss and take-profit rules to positions.
    """
    
    def __init__(
        self,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.10,
        max_position_size: float = 1.0,
        max_leverage: float = 1.0,
        price_col: str = 'close'
    ):
        """
        Initialize the stop-loss risk model.
        
        Args:
            stop_loss_pct: Stop loss percentage (e.g., 0.05 = 5% loss)
            take_profit_pct: Take profit percentage (e.g., 0.10 = 10% gain)
            max_position_size: Maximum allowed position size (0.0 to 1.0)
            max_leverage: Maximum allowed leverage
            price_col: Column name for price data
        """
        super().__init__(max_position_size, max_leverage, name="StopLossRisk")
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.price_col = price_col
        
        # State tracking
        self.entry_prices = {}  # Map from date to entry price
        self.active_positions = {}  # Map from date to position size
        
        logger.info(f"Stop loss risk model parameters: stop_loss={stop_loss_pct}, "
                   f"take_profit={take_profit_pct}")
    
    def apply_stop_loss(
        self,
        positions: pd.Series,
        data: pd.DataFrame
    ) -> pd.Series:
        """
        Apply stop-loss and take-profit rules to positions.
        
        Args:
            positions: Series with position sizes
            data: DataFrame with market data including price
        
        Returns:
            Series with adjusted position sizes after applying stop-loss rules
        """
        if self.price_col not in data.columns:
            logger.error(f"Price column '{self.price_col}' not found in data")
            return positions
        
        # Get price data
        prices = data[self.price_col]
        
        # Create copy of positions to modify
        adjusted_positions = positions.copy()
        
        # Apply stop-loss and take-profit rules iteratively
        for i, date in enumerate(adjusted_positions.index):
            if i == 0:
                # First day, just record the position and entry price
                if adjusted_positions.iloc[0] != 0:
                    self.entry_prices[date] = prices.iloc[0]
                    self.active_positions[date] = adjusted_positions.iloc[0]
                continue
            
            current_price = prices.iloc[i]
            current_position = adjusted_positions.iloc[i]
            
            # Check if position changed (new trade)
            if i > 0 and current_position != adjusted_positions.iloc[i-1]:
                # Record new entry price
                if current_position != 0:
                    self.entry_prices[date] = current_price
                    self.active_positions[date] = current_position
            
            # Get the most recent entry price and active position
            if self.active_positions:
                entry_date = max(d for d in self.active_positions.keys() if d <= date)
                active_position = self.active_positions[entry_date]
                entry_price = self.entry_prices[entry_date]
                
                # Calculate price change since entry
                price_change_pct = (current_price - entry_price) / entry_price
                
                # Apply stop-loss for long positions
                if active_position > 0 and price_change_pct < -self.stop_loss_pct:
                    logger.info(f"Stop loss triggered on {date}: price_change={price_change_pct:.4f}")
                    adjusted_positions.loc[date] = 0
                    self.active_positions[date] = 0
                
                # Apply stop-loss for short positions
                elif active_position < 0 and price_change_pct > self.stop_loss_pct:
                    logger.info(f"Stop loss triggered on {date}: price_change={price_change_pct:.4f}")
                    adjusted_positions.loc[date] = 0
                    self.active_positions[date] = 0
                
                # Apply take-profit for long positions
                elif active_position > 0 and price_change_pct > self.take_profit_pct:
                    logger.info(f"Take profit triggered on {date}: price_change={price_change_pct:.4f}")
                    adjusted_positions.loc[date] = 0
                    self.active_positions[date] = 0
                
                # Apply take-profit for short positions
                elif active_position < 0 and price_change_pct < -self.take_profit_pct:
                    logger.info(f"Take profit triggered on {date}: price_change={price_change_pct:.4f}")
                    adjusted_positions.loc[date] = 0
                    self.active_positions[date] = 0
        
        return adjusted_positions
    
    def size_position(self, alpha_signals: pd.Series, data: pd.DataFrame) -> pd.Series:
        """
        Calculate position sizes from alpha signals, applying stop-loss rules.
        
        Args:
            alpha_signals: Series with alpha signals (-1 to 1)
            data: DataFrame with market data for stop-loss calculation
        
        Returns:
            Series with position sizes after applying stop-loss rules
        """
        # First, apply basic position sizing
        positions = super().size_position(alpha_signals)
        
        # Then apply stop-loss rules
        adjusted_positions = self.apply_stop_loss(positions, data)
        
        logger.info(f"Sized stop-loss adjusted positions with mean={adjusted_positions.mean():.4f}, "
                   f"std={adjusted_positions.std():.4f}")
        return adjusted_positions


class DrawdownControlRisk(RiskModel):
    """
    Risk model that dynamically adjusts position sizes based on recent drawdowns.
    """
    
    def __init__(
        self,
        max_drawdown: float = 0.15,
        scaling_factor: float = 2.0,
        lookback_window: int = 60,
        max_position_size: float = 1.0,
        max_leverage: float = 1.0
    ):
        """
        Initialize the drawdown control risk model.
        
        Args:
            max_drawdown: Maximum allowed drawdown (e.g., 0.15 = 15% drawdown)
            scaling_factor: How quickly to reduce positions as drawdown approaches max
            lookback_window: Number of periods to use for drawdown calculation
            max_position_size: Maximum allowed position size (0.0 to 1.0)
            max_leverage: Maximum allowed leverage
        """
        super().__init__(max_position_size, max_leverage, name="DrawdownControlRisk")
        self.max_drawdown = max_drawdown
        self.scaling_factor = scaling_factor
        self.lookback_window = lookback_window
        
        logger.info(f"Drawdown control risk model parameters: max_drawdown={max_drawdown}, "
                   f"scaling_factor={scaling_factor}, lookback_window={lookback_window}")
    
    def calculate_drawdown(self, returns: pd.Series) -> pd.Series:
        """
        Calculate running drawdown from returns series.
        
        Args:
            returns: Series with daily returns
        
        Returns:
            Series with drawdowns
        """
        # Calculate cumulative returns
        cum_returns = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cum_returns.expanding().max()
        
        # Calculate drawdown
        drawdown = (cum_returns / running_max) - 1
        
        return drawdown
    
    def calculate_position_scalar(self, drawdown: pd.Series) -> pd.Series:
        """
        Calculate position scaling factor based on drawdown.
        
        Args:
            drawdown: Series with drawdowns
        
        Returns:
            Series with position scaling factors
        """
        # Calculate how close we are to max drawdown (0 = no drawdown, 1 = at max drawdown)
        # Note: drawdown is negative, so we use abs
        drawdown_ratio = abs(drawdown) / self.max_drawdown
        
        # Calculate scaling factor (exponential decay as we approach max drawdown)
        # When drawdown_ratio = 0, scalar = 1
        # When drawdown_ratio = 1, scalar = exp(-scaling_factor)
        scalar = np.exp(-self.scaling_factor * drawdown_ratio)
        
        return scalar
    
    def size_position(self, alpha_signals: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Calculate position sizes from alpha signals, scaling by drawdown.
        
        Args:
            alpha_signals: Series with alpha signals (-1 to 1)
            returns: Series with daily returns for drawdown calculation
        
        Returns:
            Series with position sizes scaled by drawdown
        """
        # Check if indexes match
        if not alpha_signals.index.equals(returns.index):
            logger.warning("Alpha signals and returns indexes don't match, reindexing returns")
            returns = returns.reindex(alpha_signals.index)
        
        # Calculate drawdown
        drawdown = self.calculate_drawdown(returns)
        
        # Calculate position scaling factor
        position_scalar = self.calculate_position_scalar(drawdown)
        
        # Scale alpha signals by position scalar and max position size
        positions = alpha_signals * position_scalar * self.max_position_size
        
        logger.info(f"Sized drawdown-adjusted positions with mean={positions.mean():.4f}, "
                   f"std={positions.std():.4f}")
        return positions


class CompositeRiskModel(RiskModel):
    """
    Composite risk model that combines multiple risk models.
    """
    
    def __init__(
        self,
        risk_models: List[RiskModel],
        max_position_size: float = 1.0,
        max_leverage: float = 1.0
    ):
        """
        Initialize the composite risk model.
        
        Args:
            risk_models: List of risk models to combine
            max_position_size: Maximum allowed position size (0.0 to 1.0)
            max_leverage: Maximum allowed leverage
        """
        model_names = [model.name for model in risk_models]
        super().__init__(max_position_size, max_leverage, name=f"CompositeRisk_{'+'.join(model_names)}")
        
        self.risk_models = risk_models
        
        logger.info(f"Composite risk model initialized with {len(risk_models)} models: {model_names}")
    
    def size_position(self, alpha_signals: pd.Series, data: Optional[pd.DataFrame] = None,
                     returns: Optional[pd.Series] = None) -> pd.Series:
        """
        Calculate position sizes from alpha signals, applying all risk models.
        
        Args:
            alpha_signals: Series with alpha signals (-1 to 1)
            data: DataFrame with market data (for volatility and stop-loss calculation)
            returns: Series with daily returns (for drawdown calculation)
        
        Returns:
            Series with position sizes after applying all risk models
        """
        # Start with full positions based on alpha signals
        positions = alpha_signals.copy()
        
        # Apply each risk model in sequence
        for model in self.risk_models:
            if isinstance(model, VolatilityBasedRisk) and data is not None:
                positions = model.size_position(positions, data)
            elif isinstance(model, StopLossRisk) and data is not None:
                positions = model.size_position(positions, data)
            elif isinstance(model, DrawdownControlRisk) and returns is not None:
                positions = model.size_position(positions, returns)
            else:
                positions = model.size_position(positions)
        
        # Ensure final positions are within allowed limits
        positions = positions.clip(lower=-self.max_position_size, upper=self.max_position_size)
        
        logger.info(f"Sized composite risk-adjusted positions with mean={positions.mean():.4f}, "
                   f"std={positions.std():.4f}")
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
    volatility_risk = VolatilityBasedRisk(
        target_volatility=target_volatility,
        volatility_lookback=volatility_lookback,
        max_position_size=max_position_size,
        max_leverage=max_leverage
    )
    
    stop_loss_risk = StopLossRisk(
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=stop_loss_pct*2,  # Typically 2:1 reward:risk ratio
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
    composite_risk = CompositeRiskModel(
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
        vol_risk = VolatilityBasedRisk(target_volatility=0.15, volatility_lookback=20)
        stop_loss_risk = StopLossRisk(stop_loss_pct=0.05, take_profit_pct=0.10)
        drawdown_risk = DrawdownControlRisk(max_drawdown=0.15)
        
        # Size positions with each risk model
        basic_positions = basic_risk.size_position(alpha_signals)
        vol_positions = vol_risk.size_position(alpha_signals, data)
        stop_loss_positions = stop_loss_risk.size_position(alpha_signals, data)
        drawdown_positions = drawdown_risk.size_position(alpha_signals, returns)
        
        # Create composite risk model
        composite_risk = CompositeRiskModel(
            risk_models=[vol_risk, stop_loss_risk, drawdown_risk],
            max_position_size=1.0
        )
        composite_positions = composite_risk.size_position(alpha_signals, data, returns)
        
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
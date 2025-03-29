"""
Risk model for position sizing and risk management.
"""

import numpy as np
import pandas as pd
from typing import Optional
from src.utils.logger import get_logger

class RiskModel:
    def __init__(
        self,
        max_position_size: float = 0.25,  # Balanced position size
        stop_loss_pct: float = 0.03,  # More room for price movement
        max_drawdown_pct: float = 0.12,  # Balanced drawdown limit
        volatility_lookback: int = 20,
        volatility_target: float = 0.12  # Balanced volatility target
    ):
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.volatility_lookback = volatility_lookback
        self.volatility_target = volatility_target
        self.logger = get_logger(__name__)
        
        self.current_drawdown = 0
        self.peak_equity = None
        self.last_price = None
        self.entry_price = None
        self.current_position = 0
        
    def size_position(self, signal: float, data: pd.DataFrame) -> float:
        """
        Size the position based on risk parameters and market conditions.
        
        Args:
            signal: Alpha signal (-1 to 1)
            data: Market data DataFrame
            
        Returns:
            Position size (-1 to 1)
        """
        try:
            # Get current price
            if 'price' in data.columns:
                price = data['price'].iloc[-1]
            elif 'close' in data.columns:
                price = data['close'].iloc[-1]
            else:
                self.logger.warning("No price column found in data")
                return 0.0
            
            # Initialize peak equity
            if self.peak_equity is None:
                self.peak_equity = price
            
            # Update peak equity and drawdown
            if price > self.peak_equity:
                self.peak_equity = price
            self.current_drawdown = (self.peak_equity - price) / self.peak_equity
            
            # Calculate volatility scaling
            returns = data['price'].pct_change() if 'price' in data.columns else data['close'].pct_change()
            volatility = returns.rolling(window=self.volatility_lookback).std() * np.sqrt(252)
            vol_scale = min(1.5, self.volatility_target / volatility.iloc[-1]) if not volatility.empty else 1.0
            
            # Base position size with signal strength scaling
            signal_strength = abs(signal)
            position_size = signal * self.max_position_size * vol_scale * signal_strength
            
            # Apply drawdown control with smoother reduction
            if self.current_drawdown > self.max_drawdown_pct:
                drawdown_scale = 1.0 - 0.5 * (self.current_drawdown - self.max_drawdown_pct) / self.max_drawdown_pct
                position_size *= max(0.2, drawdown_scale)  # Keep at least 20% of position
            
            # Apply stop loss if we have an existing position
            if self.current_position != 0 and self.entry_price is not None:
                price_change = (price - self.entry_price) / self.entry_price
                if (self.current_position > 0 and price_change < -self.stop_loss_pct) or \
                   (self.current_position < 0 and price_change > self.stop_loss_pct):
                    position_size = 0  # Exit position
            
            # Update state
            if position_size != 0 and self.current_position == 0:
                self.entry_price = price
            elif position_size == 0:
                self.entry_price = None
            
            self.last_price = price
            self.current_position = position_size
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error in size_position: {str(e)}")
            return 0.0

def create_default_risk_model() -> RiskModel:
    """
    Create a default risk model instance.
    
    Returns:
        RiskModel instance with default parameters
    """
    return RiskModel(
        max_position_size=0.25,
        stop_loss_pct=0.03,
        max_drawdown_pct=0.12,
        volatility_lookback=20,
        volatility_target=0.12
    ) 
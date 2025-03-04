"""
Portfolio Construction - Combines alpha signals and risk constraints to generate final positions.

This module implements the portfolio construction layer for natural gas trading:
1. Takes raw alpha signals from strategy models
2. Applies risk constraints from risk models
3. Generates target positions for execution
4. Handles portfolio rebalancing and position sizing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from pathlib import Path
import os
import logging
from datetime import datetime, timedelta

# Import utility logger
from src.utils.logger import get_logger

# Import from other modules
from src.strategies.alpha_model import AlphaModel
from src.strategies.risk_model import RiskModel

# Configure logging
logger = get_logger(__name__)


class PortfolioConstructor:
    """
    Base portfolio constructor class.
    """
    
    def __init__(
        self,
        alpha_model: Optional[AlphaModel] = None,
        risk_model: Optional[RiskModel] = None,
        name: str = "DefaultPortfolioConstructor"
    ):
        """
        Initialize portfolio constructor.
        
        Args:
            alpha_model: Alpha model to generate trading signals
            risk_model: Risk model to apply constraints
            name: Name of the portfolio constructor
        """
        self.alpha_model = alpha_model
        self.risk_model = risk_model
        self.name = name
        
        logger.info(f"Initialized portfolio constructor: {self.name}")
    
    def generate_positions(
        self,
        data: pd.DataFrame,
        current_positions: Optional[Dict[str, float]] = None
    ) -> pd.Series:
        """
        Generate target positions for a single asset.
        
        Args:
            data: Market data with features for alpha generation
            current_positions: Current positions in the portfolio (for rebalancing)
        
        Returns:
            Series with target position sizes
        """
        # Generate alpha signals
        if self.alpha_model is None:
            raise ValueError("Alpha model must be set to generate positions")
        
        alpha_signals = self.alpha_model.generate_signals(data)
        
        # Apply risk constraints
        if self.risk_model is None:
            logger.warning("No risk model set, using raw alpha signals")
            return alpha_signals
        
        # Check if we have price returns for drawdown control
        returns = None
        if "returns" in data.columns:
            returns = data["returns"]
        
        # Apply risk model
        positions = self.risk_model.size_position(alpha_signals, data, returns)
        
        logger.info(f"Generated positions with mean={positions.mean():.4f}, "
                   f"std={positions.std():.4f}, min={positions.min():.4f}, max={positions.max():.4f}")
        
        return positions
    
    def set_alpha_model(self, alpha_model: AlphaModel):
        """
        Set the alpha model.
        
        Args:
            alpha_model: Alpha model to use
        """
        self.alpha_model = alpha_model
        logger.info(f"Set alpha model: {alpha_model.name}")
    
    def set_risk_model(self, risk_model: RiskModel):
        """
        Set the risk model.
        
        Args:
            risk_model: Risk model to use
        """
        self.risk_model = risk_model
        logger.info(f"Set risk model: {risk_model.name}")


class MultiAssetPortfolioConstructor(PortfolioConstructor):
    """
    Portfolio constructor for multiple assets.
    """
    
    def __init__(
        self,
        alpha_models: Optional[Dict[str, AlphaModel]] = None,
        risk_models: Optional[Dict[str, RiskModel]] = None,
        portfolio_risk_model: Optional[RiskModel] = None,
        name: str = "MultiAssetPortfolio"
    ):
        """
        Initialize multi-asset portfolio constructor.
        
        Args:
            alpha_models: Dictionary mapping symbols to alpha models
            risk_models: Dictionary mapping symbols to risk models
            portfolio_risk_model: Risk model for portfolio-level constraints
            name: Name of the portfolio constructor
        """
        super().__init__(name=name)
        self.alpha_models = alpha_models or {}
        self.risk_models = risk_models or {}
        self.portfolio_risk_model = portfolio_risk_model
        
        logger.info(f"Initialized multi-asset portfolio constructor with {len(self.alpha_models)} "
                   f"alpha models and {len(self.risk_models)} risk models")
    
    def add_asset(
        self,
        symbol: str,
        alpha_model: AlphaModel,
        risk_model: Optional[RiskModel] = None
    ):
        """
        Add an asset to the portfolio.
        
        Args:
            symbol: Asset symbol
            alpha_model: Alpha model for the asset
            risk_model: Risk model for the asset (optional)
        """
        self.alpha_models[symbol] = alpha_model
        if risk_model:
            self.risk_models[symbol] = risk_model
            
        logger.info(f"Added asset {symbol} with alpha model {alpha_model.name} and "
                  f"risk model {risk_model.name if risk_model else 'None'}")
    
    def remove_asset(self, symbol: str):
        """
        Remove an asset from the portfolio.
        
        Args:
            symbol: Asset symbol to remove
        """
        if symbol in self.alpha_models:
            del self.alpha_models[symbol]
        
        if symbol in self.risk_models:
            del self.risk_models[symbol]
            
        logger.info(f"Removed asset {symbol} from portfolio")
    
    def generate_positions_for_asset(
        self,
        symbol: str,
        data: pd.DataFrame,
        current_positions: Optional[Dict[str, float]] = None
    ) -> pd.Series:
        """
        Generate target positions for a single asset.
        
        Args:
            symbol: Asset symbol
            data: Market data with features for alpha generation
            current_positions: Current positions in the portfolio (for rebalancing)
        
        Returns:
            Series with target position sizes
        """
        if symbol not in self.alpha_models:
            raise ValueError(f"No alpha model found for symbol {symbol}")
        
        alpha_model = self.alpha_models[symbol]
        alpha_signals = alpha_model.generate_signals(data)
        
        # Apply asset-specific risk constraints if available
        if symbol in self.risk_models:
            risk_model = self.risk_models[symbol]
            
            # Check if we have price returns for drawdown control
            returns = None
            if "returns" in data.columns:
                returns = data["returns"]
            
            # Apply risk model
            positions = risk_model.size_position(alpha_signals, data, returns)
        else:
            positions = alpha_signals
        
        logger.info(f"Generated positions for {symbol} with mean={positions.mean():.4f}, "
                   f"std={positions.std():.4f}, min={positions.min():.4f}, max={positions.max():.4f}")
        
        return positions
    
    def generate_positions(
        self,
        data_dict: Dict[str, pd.DataFrame],
        current_positions: Optional[Dict[str, float]] = None
    ) -> Dict[str, pd.Series]:
        """
        Generate target positions for all assets in the portfolio.
        
        Args:
            data_dict: Dictionary mapping symbols to market data
            current_positions: Current positions in the portfolio (for rebalancing)
        
        Returns:
            Dictionary mapping symbols to target position sizes
        """
        positions_dict = {}
        raw_positions = {}
        
        # Generate positions for each asset
        for symbol, asset_data in data_dict.items():
            if symbol in self.alpha_models:
                try:
                    positions = self.generate_positions_for_asset(symbol, asset_data, current_positions)
                    positions_dict[symbol] = positions
                    raw_positions[symbol] = positions
                except Exception as e:
                    logger.error(f"Error generating positions for {symbol}: {e}")
        
        # Apply portfolio-level risk constraints if available
        if self.portfolio_risk_model and positions_dict:
            # Combine all positions into a single DataFrame
            position_df = pd.DataFrame(positions_dict)
            
            # Apply portfolio risk model
            # This is a simplified approach - a real implementation would
            # consider correlations between assets
            for symbol in position_df.columns:
                # Scale positions to fit within portfolio constraints
                # This assumes the portfolio risk model exposes a method
                # to scale positions for the entire portfolio
                if symbol in data_dict:
                    try:
                        asset_data = data_dict[symbol]
                        # Check if we have price returns for drawdown control
                        returns = None
                        if "returns" in asset_data.columns:
                            returns = asset_data["returns"]
                        
                        # Apply risk model at the asset level
                        positions_dict[symbol] = self.portfolio_risk_model.size_position(
                            position_df[symbol], asset_data, returns
                        )
                    except Exception as e:
                        logger.error(f"Error applying portfolio risk model to {symbol}: {e}")
        
        logger.info(f"Generated positions for {len(positions_dict)} assets")
        return positions_dict
    
    def calculate_portfolio_weight_changes(
        self,
        target_positions: Dict[str, pd.Series],
        current_positions: Dict[str, float],
        prices: Dict[str, float]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calculate portfolio weight changes from current to target positions.
        
        Args:
            target_positions: Dictionary of target position Series for each symbol
            current_positions: Dictionary of current positions for each symbol
            prices: Dictionary of current prices for each symbol
        
        Returns:
            Tuple of (target_weights, weight_changes) dictionaries
        """
        # Calculate portfolio value
        portfolio_value = sum(current_positions.get(symbol, 0) * prices.get(symbol, 0) 
                             for symbol in set(current_positions) | set(target_positions))
        
        if portfolio_value <= 0:
            logger.warning("Portfolio value is zero or negative, using equal weights")
            portfolio_value = 1.0
        
        # Calculate current weights
        current_weights = {
            symbol: current_positions.get(symbol, 0) * prices.get(symbol, 0) / portfolio_value
            for symbol in set(current_positions) | set(target_positions)
        }
        
        # Get the most recent target position for each symbol
        latest_targets = {}
        for symbol, pos_series in target_positions.items():
            if not pos_series.empty:
                latest_targets[symbol] = pos_series.iloc[-1]
        
        # Calculate target weights (assuming target positions are in units, not weights)
        target_weights = {
            symbol: latest_targets.get(symbol, 0) * prices.get(symbol, 0) / portfolio_value
            for symbol in set(latest_targets) | set(current_positions)
        }
        
        # Calculate weight changes
        weight_changes = {
            symbol: target_weights.get(symbol, 0) - current_weights.get(symbol, 0)
            for symbol in set(target_weights) | set(current_weights)
        }
        
        logger.info(f"Calculated weight changes for {len(weight_changes)} assets")
        return target_weights, weight_changes
    
    def generate_orders(
        self,
        target_positions: Dict[str, pd.Series],
        current_positions: Dict[str, float],
        prices: Dict[str, float],
        portfolio_value: float,
        min_trade_size: float = 0.01,
        max_trade_size: float = 1.0
    ) -> Dict[str, Dict[str, float]]:
        """
        Generate orders to transition from current to target positions.
        
        Args:
            target_positions: Dictionary of target position Series for each symbol
            current_positions: Dictionary of current positions for each symbol
            prices: Dictionary of current prices for each symbol
            portfolio_value: Total portfolio value
            min_trade_size: Minimum trade size as fraction of portfolio value
            max_trade_size: Maximum trade size as fraction of portfolio value
        
        Returns:
            Dictionary of orders with symbol, quantity, and side
        """
        orders = {}
        
        # Get the most recent target position for each symbol
        latest_targets = {}
        for symbol, pos_series in target_positions.items():
            if not pos_series.empty:
                latest_targets[symbol] = pos_series.iloc[-1]
        
        # Calculate quantity changes
        for symbol in set(latest_targets) | set(current_positions):
            current_position = current_positions.get(symbol, 0)
            target_position = latest_targets.get(symbol, 0)
            
            # Calculate quantity difference
            quantity_diff = target_position - current_position
            
            # Skip small trades
            if abs(quantity_diff) * prices.get(symbol, 0) < portfolio_value * min_trade_size:
                continue
            
            # Limit large trades
            if abs(quantity_diff) * prices.get(symbol, 0) > portfolio_value * max_trade_size:
                direction = 1 if quantity_diff > 0 else -1
                quantity_diff = direction * (portfolio_value * max_trade_size / prices.get(symbol, 1.0))
            
            # Create order
            if quantity_diff != 0:
                orders[symbol] = {
                    "symbol": symbol,
                    "quantity": abs(quantity_diff),
                    "side": "BUY" if quantity_diff > 0 else "SELL"
                }
        
        logger.info(f"Generated {len(orders)} orders")
        return orders


class RebalancingStrategy:
    """
    Base class for rebalancing strategies.
    """
    
    def __init__(self, name: str = "BaseRebalancer"):
        """
        Initialize rebalancing strategy.
        
        Args:
            name: Name of the rebalancing strategy
        """
        self.name = name
        logger.info(f"Initialized rebalancing strategy: {self.name}")
    
    def should_rebalance(
        self,
        current_time: datetime,
        last_rebalance_time: Optional[datetime],
        current_positions: Dict[str, float],
        target_positions: Dict[str, pd.Series],
        prices: Dict[str, float]
    ) -> bool:
        """
        Determine if the portfolio should be rebalanced.
        
        Args:
            current_time: Current time
            last_rebalance_time: Time of last rebalance (None if never rebalanced)
            current_positions: Current positions
            target_positions: Target positions
            prices: Current prices
        
        Returns:
            True if should rebalance, False otherwise
        """
        # Base strategy always rebalances
        return True


class TimeBasedRebalancer(RebalancingStrategy):
    """
    Time-based rebalancing strategy.
    """
    
    def __init__(
        self,
        frequency: str = "D",
        specific_times: Optional[List[str]] = None,
        name: str = "TimeBasedRebalancer"
    ):
        """
        Initialize time-based rebalancing strategy.
        
        Args:
            frequency: Rebalancing frequency ('D' for daily, 'W' for weekly, etc.)
            specific_times: List of specific times of day to rebalance (e.g. ["09:30", "16:00"])
            name: Name of the rebalancing strategy
        """
        super().__init__(name=name)
        self.frequency = frequency
        self.specific_times = specific_times
        
        logger.info(f"Initialized time-based rebalancer with frequency {frequency}, "
                   f"specific times: {specific_times}")
    
    def should_rebalance(
        self,
        current_time: datetime,
        last_rebalance_time: Optional[datetime],
        current_positions: Dict[str, float],
        target_positions: Dict[str, pd.Series],
        prices: Dict[str, float]
    ) -> bool:
        """
        Determine if the portfolio should be rebalanced based on time.
        
        Args:
            current_time: Current time
            last_rebalance_time: Time of last rebalance (None if never rebalanced)
            current_positions: Current positions
            target_positions: Target positions
            prices: Current prices
        
        Returns:
            True if should rebalance, False otherwise
        """
        # If never rebalanced, do it now
        if last_rebalance_time is None:
            return True
        
        # Check if specific times are set
        if self.specific_times:
            current_time_str = current_time.strftime("%H:%M")
            if current_time_str in self.specific_times:
                return True
        
        # Check based on frequency
        if self.frequency == "D":
            if current_time.date() > last_rebalance_time.date():
                return True
        elif self.frequency == "W":
            delta = current_time - last_rebalance_time
            if delta.days >= 7:
                return True
        elif self.frequency == "M":
            if (current_time.year > last_rebalance_time.year or 
                (current_time.year == last_rebalance_time.year and 
                 current_time.month > last_rebalance_time.month)):
                return True
        elif self.frequency == "H":
            if current_time.hour != last_rebalance_time.hour or current_time.date() != last_rebalance_time.date():
                return True
        
        return False


class ThresholdRebalancer(RebalancingStrategy):
    """
    Threshold-based rebalancing strategy.
    """
    
    def __init__(
        self,
        threshold: float = 0.05,
        name: str = "ThresholdRebalancer"
    ):
        """
        Initialize threshold-based rebalancing strategy.
        
        Args:
            threshold: Minimum position deviation to trigger rebalance (as fraction)
            name: Name of the rebalancing strategy
        """
        super().__init__(name=name)
        self.threshold = threshold
        
        logger.info(f"Initialized threshold rebalancer with threshold {threshold}")
    
    def should_rebalance(
        self,
        current_time: datetime,
        last_rebalance_time: Optional[datetime],
        current_positions: Dict[str, float],
        target_positions: Dict[str, pd.Series],
        prices: Dict[str, float]
    ) -> bool:
        """
        Determine if the portfolio should be rebalanced based on position deviation.
        
        Args:
            current_time: Current time
            last_rebalance_time: Time of last rebalance (None if never rebalanced)
            current_positions: Current positions
            target_positions: Target positions
            prices: Current prices
        
        Returns:
            True if should rebalance, False otherwise
        """
        # If never rebalanced, do it now
        if last_rebalance_time is None:
            return True
        
        # Calculate portfolio value
        portfolio_value = sum(current_positions.get(symbol, 0) * prices.get(symbol, 0) 
                             for symbol in current_positions)
        
        if portfolio_value <= 0:
            return True
        
        # Get latest target positions
        latest_targets = {}
        for symbol, pos_series in target_positions.items():
            if not pos_series.empty:
                latest_targets[symbol] = pos_series.iloc[-1]
        
        # Check for significant deviations
        for symbol in set(latest_targets) | set(current_positions):
            current_position = current_positions.get(symbol, 0)
            target_position = latest_targets.get(symbol, 0)
            
            current_value = current_position * prices.get(symbol, 0)
            target_value = target_position * prices.get(symbol, 0)
            
            # Calculate weight deviation
            current_weight = current_value / portfolio_value if portfolio_value > 0 else 0
            target_weight = target_value / portfolio_value if portfolio_value > 0 else 0
            
            if abs(current_weight - target_weight) > self.threshold:
                return True
        
        return False


class HybridRebalancer(RebalancingStrategy):
    """
    Hybrid rebalancing strategy combining time and threshold.
    """
    
    def __init__(
        self,
        time_rebalancer: TimeBasedRebalancer,
        threshold_rebalancer: ThresholdRebalancer,
        name: str = "HybridRebalancer"
    ):
        """
        Initialize hybrid rebalancing strategy.
        
        Args:
            time_rebalancer: Time-based rebalancer
            threshold_rebalancer: Threshold-based rebalancer
            name: Name of the rebalancing strategy
        """
        super().__init__(name=name)
        self.time_rebalancer = time_rebalancer
        self.threshold_rebalancer = threshold_rebalancer
        
        logger.info(f"Initialized hybrid rebalancer with {time_rebalancer.name} and {threshold_rebalancer.name}")
    
    def should_rebalance(
        self,
        current_time: datetime,
        last_rebalance_time: Optional[datetime],
        current_positions: Dict[str, float],
        target_positions: Dict[str, pd.Series],
        prices: Dict[str, float]
    ) -> bool:
        """
        Determine if the portfolio should be rebalanced based on time or threshold.
        
        Args:
            current_time: Current time
            last_rebalance_time: Time of last rebalance (None if never rebalanced)
            current_positions: Current positions
            target_positions: Target positions
            prices: Current prices
        
        Returns:
            True if should rebalance, False otherwise
        """
        # Check time conditions
        time_rebalance = self.time_rebalancer.should_rebalance(
            current_time, last_rebalance_time, current_positions, target_positions, prices
        )
        
        # Check threshold conditions
        threshold_rebalance = self.threshold_rebalancer.should_rebalance(
            current_time, last_rebalance_time, current_positions, target_positions, prices
        )
        
        # Rebalance if either condition is met
        return time_rebalance or threshold_rebalance


class PortfolioManager:
    """
    Portfolio manager that combines portfolio construction and rebalancing.
    """
    
    def __init__(
        self,
        portfolio_constructor: PortfolioConstructor,
        rebalancer: RebalancingStrategy,
        order_generator: Optional[Callable] = None,
        name: str = "PortfolioManager"
    ):
        """
        Initialize portfolio manager.
        
        Args:
            portfolio_constructor: Portfolio constructor to generate positions
            rebalancer: Rebalancing strategy
            order_generator: Function to generate orders from position changes
            name: Name of the portfolio manager
        """
        self.portfolio_constructor = portfolio_constructor
        self.rebalancer = rebalancer
        self.order_generator = order_generator
        self.name = name
        
        # State tracking
        self.current_positions = {}
        self.last_rebalance_time = None
        self.target_positions = {}
        
        logger.info(f"Initialized portfolio manager: {self.name}")
    
    def update_state(
        self,
        current_time: datetime,
        data_dict: Dict[str, pd.DataFrame],
        prices: Dict[str, float]
    ):
        """
        Update portfolio state with new data.
        
        Args:
            current_time: Current time
            data_dict: Dictionary of market data by symbol
            prices: Dictionary of current prices by symbol
        """
        # Generate new target positions
        self.target_positions = self.portfolio_constructor.generate_positions(
            data_dict, self.current_positions
        )
        
        # Check if we should rebalance
        if self.rebalancer.should_rebalance(
            current_time, self.last_rebalance_time, self.current_positions,
            self.target_positions, prices
        ):
            logger.info(f"Rebalancing portfolio at {current_time}")
            
            # Calculate portfolio value
            portfolio_value = sum(self.current_positions.get(symbol, 0) * prices.get(symbol, 0) 
                                 for symbol in set(self.current_positions) | set(self.target_positions))
            
            # Generate orders
            if self.order_generator:
                orders = self.order_generator(
                    self.target_positions, self.current_positions, prices, portfolio_value
                )
            else:
                # Use default order generator
                orders = self._default_order_generator(
                    self.target_positions, self.current_positions, prices, portfolio_value
                )
            
            # Update rebalance time
            self.last_rebalance_time = current_time
            
            return orders
        
        return {}
    
    def _default_order_generator(
        self,
        target_positions: Dict[str, pd.Series],
        current_positions: Dict[str, float],
        prices: Dict[str, float],
        portfolio_value: float
    ) -> Dict[str, Dict[str, float]]:
        """
        Default order generator.
        
        Args:
            target_positions: Dictionary of target position Series for each symbol
            current_positions: Dictionary of current positions for each symbol
            prices: Dictionary of current prices for each symbol
            portfolio_value: Total portfolio value
        
        Returns:
            Dictionary of orders with symbol, quantity, and side
        """
        orders = {}
        
        # Get the most recent target position for each symbol
        latest_targets = {}
        for symbol, pos_series in target_positions.items():
            if not pos_series.empty:
                latest_targets[symbol] = pos_series.iloc[-1]
        
        # Calculate quantity changes
        for symbol in set(latest_targets) | set(current_positions):
            current_position = current_positions.get(symbol, 0)
            target_position = latest_targets.get(symbol, 0)
            
            # Calculate quantity difference
            quantity_diff = target_position - current_position
            
            # Skip small trades
            min_trade_size = 0.01  # 1% of portfolio
            if abs(quantity_diff) * prices.get(symbol, 0) < portfolio_value * min_trade_size:
                continue
            
            # Create order
            if quantity_diff != 0:
                orders[symbol] = {
                    "symbol": symbol,
                    "quantity": abs(quantity_diff),
                    "side": "BUY" if quantity_diff > 0 else "SELL"
                }
        
        logger.info(f"Generated {len(orders)} orders")
        return orders
    
    def update_positions(self, symbol: str, quantity: float):
        """
        Update current positions after a trade.
        
        Args:
            symbol: Symbol that was traded
            quantity: Signed quantity that was traded (positive for buys, negative for sells)
        """
        current_position = self.current_positions.get(symbol, 0)
        new_position = current_position + quantity
        
        # Update position
        self.current_positions[symbol] = new_position
        
        logger.info(f"Updated position for {symbol}: {current_position} -> {new_position} ({quantity})")


def create_portfolio_manager(
    config: Dict[str, Any],
    alpha_models: Dict[str, AlphaModel],
    risk_models: Dict[str, RiskModel]
) -> PortfolioManager:
    """
    Factory function to create a portfolio manager.
    
    Args:
        config: Configuration dictionary
        alpha_models: Dictionary of alpha models by symbol
        risk_models: Dictionary of risk models by symbol
    
    Returns:
        PortfolioManager instance
    """
    # Create portfolio constructor
    constructor = MultiAssetPortfolioConstructor(
        alpha_models=alpha_models,
        risk_models=risk_models,
        name=config.get("portfolio_constructor_name", "DefaultPortfolioConstructor")
    )
    
    # Create rebalancer
    rebalance_config = config.get("rebalance", {})
    rebalance_type = rebalance_config.get("type", "time").lower()
    
    if rebalance_type == "time":
        rebalancer = TimeBasedRebalancer(
            frequency=rebalance_config.get("frequency", "D"),
            specific_times=rebalance_config.get("specific_times", None)
        )
    elif rebalance_type == "threshold":
        rebalancer = ThresholdRebalancer(
            threshold=rebalance_config.get("threshold", 0.05)
        )
    elif rebalance_type == "hybrid":
        time_rebalancer = TimeBasedRebalancer(
            frequency=rebalance_config.get("frequency", "D"),
            specific_times=rebalance_config.get("specific_times", None)
        )
        threshold_rebalancer = ThresholdRebalancer(
            threshold=rebalance_config.get("threshold", 0.05)
        )
        rebalancer = HybridRebalancer(time_rebalancer, threshold_rebalancer)
    else:
        rebalancer = RebalancingStrategy()
    
    # Create portfolio manager
    portfolio_manager = PortfolioManager(
        portfolio_constructor=constructor,
        rebalancer=rebalancer,
        name=config.get("portfolio_manager_name", "DefaultPortfolioManager")
    )
    
    return portfolio_manager


if __name__ == "__main__":
    # Example usage
    try:
        # Import needed modules for the example
        from src.strategies.alpha_model import MomentumModel, RSIModel, CombinationModel
        from src.strategies.risk_model import StopLossRisk, VolatilityBasedRisk, CompositeRiskModel
        
        # Create sample data
        index = pd.date_range(start='2020-01-01', end='2020-01-31', freq='D')
        
        # Sample data for natural gas
        ng_data = pd.DataFrame({
            'open': np.linspace(2.0, 2.5, len(index)) + np.random.normal(0, 0.1, len(index)),
            'high': np.linspace(2.1, 2.6, len(index)) + np.random.normal(0, 0.1, len(index)),
            'low': np.linspace(1.9, 2.4, len(index)) + np.random.normal(0, 0.1, len(index)),
            'close': np.linspace(2.0, 2.5, len(index)) + np.random.normal(0, 0.1, len(index)),
            'volume': np.random.randint(100000, 500000, len(index)),
            'rsi_14': np.random.uniform(30, 70, len(index)),
            'returns': np.random.normal(0, 0.02, len(index))
        }, index=index)
        
        # Sample data for crude oil
        cl_data = pd.DataFrame({
            'open': np.linspace(50, 55, len(index)) + np.random.normal(0, 1.0, len(index)),
            'high': np.linspace(51, 56, len(index)) + np.random.normal(0, 1.0, len(index)),
            'low': np.linspace(49, 54, len(index)) + np.random.normal(0, 1.0, len(index)),
            'close': np.linspace(50, 55, len(index)) + np.random.normal(0, 1.0, len(index)),
            'volume': np.random.randint(100000, 500000, len(index)),
            'rsi_14': np.random.uniform(30, 70, len(index)),
            'returns': np.random.normal(0, 0.02, len(index))
        }, index=index)
        
        # Create alpha and risk models
        momentum_model = MomentumModel(lookback_period=10)
        rsi_model = RSIModel()
        
        # Combination model for natural gas
        ng_alpha = CombinationModel(
            models=[momentum_model, rsi_model],
            weights=[0.6, 0.4],
            name="NG_Combo"
        )
        
        # Momentum model for crude oil
        cl_alpha = MomentumModel(lookback_period=5, name="CL_Momentum")
        
        # Risk models
        vol_risk = VolatilityBasedRisk(target_volatility=0.15, volatility_lookback=20)
        stop_loss_risk = StopLossRisk(stop_loss_pct=0.05, take_profit_pct=0.10)
        
        # Composite risk model
        composite_risk = CompositeRiskModel(
            risk_models=[vol_risk, stop_loss_risk],
            max_position_size=1.0,
            max_leverage=1.0
        )
        
        # Create portfolio constructor
        constructor = MultiAssetPortfolioConstructor(
            alpha_models={"NG": ng_alpha, "CL": cl_alpha},
            risk_models={"NG": composite_risk, "CL": vol_risk}
        )
        
        # Create rebalancer
        rebalancer = HybridRebalancer(
            time_rebalancer=TimeBasedRebalancer(frequency="D"),
            threshold_rebalancer=ThresholdRebalancer(threshold=0.05)
        )
        
        # Create portfolio manager
        manager = PortfolioManager(
            portfolio_constructor=constructor,
            rebalancer=rebalancer
        )
        
        # Run simulation
        current_prices = {"NG": 2.2, "CL": 52.0}
        data_dict = {"NG": ng_data, "CL": cl_data}
        
        # Initial update
        orders = manager.update_state(
            current_time=index[0],
            data_dict=data_dict,
            prices=current_prices
        )
        
        print(f"Initial orders: {orders}")
        
        # Simulate some trades
        for symbol, order in orders.items():
            quantity = order["quantity"] if order["side"] == "BUY" else -order["quantity"]
            manager.update_positions(symbol, quantity)
        
        # Update a few days later with new prices
        current_prices = {"NG": 2.3, "CL": 51.5}
        
        orders = manager.update_state(
            current_time=index[5],
            data_dict=data_dict,
            prices=current_prices
        )
        
        print(f"Day 5 orders: {orders}")
        print(f"Current positions: {manager.current_positions}")
        
        # Save output to CSV
        output_dir = Path(__file__).parents[2] / 'data' / 'interim'
        os.makedirs(output_dir, exist_ok=True)
        
        # Get target positions for all symbols
        all_targets = {}
        for symbol, pos_series in manager.target_positions.items():
            all_targets[f"{symbol}_target"] = pos_series
        
        # Create DataFrame with all target positions
        targets_df = pd.DataFrame(all_targets)
        targets_df.to_csv(output_dir / 'target_positions.csv')
        
        logger.info(f"Example completed successfully, targets saved to {output_dir / 'target_positions.csv'}")
        
    except Exception as e:
        logger.error(f"Error in example: {e}")
        raise 
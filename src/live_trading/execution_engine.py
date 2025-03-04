"""
Execution Engine - Handles order management and trade execution.

This module implements the execution layer for natural gas trading:
1. Converts target positions to orders
2. Manages order lifecycle (creation, submission, cancellation)
3. Tracks execution and fills
4. Provides simulation capabilities for backtesting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from enum import Enum
from datetime import datetime, timedelta
import time
import uuid
import logging
from pathlib import Path
import os
import json

# Import utility logger
from src.utils.logger import get_logger

# Configure logging
logger = get_logger(__name__)


class OrderType(Enum):
    """Enum for different order types."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(Enum):
    """Enum for order side (buy/sell)."""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Enum for order status."""
    CREATED = "CREATED"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class Order:
    """
    Class representing a trading order.
    """
    
    def __init__(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "DAY",
        order_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ):
        """
        Initialize an order.
        
        Args:
            symbol: Trading symbol (e.g., "NG=F" for natural gas futures)
            side: Buy or sell
            quantity: Order quantity (positive value)
            order_type: Type of order (MARKET, LIMIT, etc.)
            limit_price: Price for limit orders
            stop_price: Price for stop orders
            time_in_force: How long the order is valid (DAY, GTC, etc.)
            order_id: Unique order identifier (auto-generated if None)
            timestamp: Time of order creation (auto-set to now if None)
        """
        self.symbol = symbol
        self.side = side
        self.quantity = abs(quantity)  # Always store as positive, use side for direction
        self.order_type = order_type
        self.limit_price = limit_price
        self.stop_price = stop_price
        self.time_in_force = time_in_force
        self.order_id = order_id if order_id else str(uuid.uuid4())
        self.timestamp = timestamp if timestamp else datetime.now()
        self.status = OrderStatus.CREATED
        
        # Tracking fills
        self.filled_quantity = 0.0
        self.average_fill_price = 0.0
        self.fills = []  # List of individual fills
        
        # Validation
        self._validate()
        
        logger.info(f"Created order {self.order_id}: {self.side.value} {self.quantity} {self.symbol} "
                   f"at {self.timestamp}")
    
    def _validate(self):
        """Validate the order parameters."""
        if self.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and self.limit_price is None:
            raise ValueError(f"Limit price must be specified for {self.order_type.value} orders")
        
        if self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and self.stop_price is None:
            raise ValueError(f"Stop price must be specified for {self.order_type.value} orders")
    
    def update_status(self, new_status: OrderStatus):
        """
        Update the order status.
        
        Args:
            new_status: New status to set
        """
        old_status = self.status
        self.status = new_status
        logger.info(f"Order {self.order_id} status updated: {old_status.value} -> {new_status.value}")
    
    def add_fill(self, quantity: float, price: float, timestamp: Optional[datetime] = None):
        """
        Add a fill to the order.
        
        Args:
            quantity: Quantity filled
            price: Fill price
            timestamp: Time of fill (defaults to now)
        """
        fill_time = timestamp if timestamp else datetime.now()
        
        # Validate quantity
        if quantity <= 0:
            raise ValueError("Fill quantity must be positive")
        
        if self.filled_quantity + quantity > self.quantity:
            raise ValueError(f"Fill quantity {quantity} would exceed remaining order quantity")
        
        # Record the fill
        fill_data = {
            "quantity": quantity,
            "price": price,
            "timestamp": fill_time
        }
        self.fills.append(fill_data)
        
        # Update filled quantity and average price
        total_value_before = self.filled_quantity * self.average_fill_price
        total_value_new_fill = quantity * price
        
        self.filled_quantity += quantity
        if self.filled_quantity > 0:
            self.average_fill_price = (total_value_before + total_value_new_fill) / self.filled_quantity
        
        # Update status
        if abs(self.filled_quantity - self.quantity) < 1e-6:  # Account for floating point imprecision
            self.update_status(OrderStatus.FILLED)
        elif self.filled_quantity > 0:
            self.update_status(OrderStatus.PARTIALLY_FILLED)
        
        logger.info(f"Fill added to order {self.order_id}: {quantity} @ {price}, "
                   f"total filled: {self.filled_quantity}/{self.quantity}")
    
    def cancel(self):
        """Cancel the order if it's not already filled."""
        if self.status not in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
            self.update_status(OrderStatus.CANCELLED)
        else:
            logger.warning(f"Cannot cancel order {self.order_id} with status {self.status.value}")
    
    def is_active(self) -> bool:
        """
        Check if the order is still active.
        
        Returns:
            True if order is active, False otherwise
        """
        return self.status in [OrderStatus.CREATED, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert order to dictionary for serialization.
        
        Returns:
            Dictionary representation of the order
        """
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "order_type": self.order_type.value,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "time_in_force": self.time_in_force,
            "status": self.status.value,
            "timestamp": str(self.timestamp),
            "filled_quantity": self.filled_quantity,
            "average_fill_price": self.average_fill_price,
            "fills": self.fills
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Order':
        """
        Create order from dictionary.
        
        Args:
            data: Dictionary with order data
        
        Returns:
            Order object
        """
        # Copy the dict to avoid modifying the original
        order_data = data.copy()
        
        # Convert string enums back to Enum objects
        order_data['side'] = OrderSide(order_data['side'])
        order_data['order_type'] = OrderType(order_data['order_type'])
        
        # Handle timestamp
        if isinstance(order_data['timestamp'], str):
            order_data['timestamp'] = datetime.fromisoformat(order_data['timestamp'])
        
        # Create the order
        order = cls(
            symbol=order_data['symbol'],
            side=order_data['side'],
            quantity=order_data['quantity'],
            order_type=order_data['order_type'],
            limit_price=order_data['limit_price'],
            stop_price=order_data['stop_price'],
            time_in_force=order_data['time_in_force'],
            order_id=order_data['order_id'],
            timestamp=order_data['timestamp']
        )
        
        # Restore status and fills
        order.status = OrderStatus(data['status'])
        order.filled_quantity = data['filled_quantity']
        order.average_fill_price = data['average_fill_price']
        order.fills = data['fills']
        
        return order


class ExecutionEngine:
    """
    Base execution engine class for order management.
    """
    
    def __init__(self, name: str = "BaseExecutionEngine"):
        """
        Initialize the execution engine.
        
        Args:
            name: Name of the execution engine
        """
        self.name = name
        self.orders = {}  # Dictionary mapping order_id to Order objects
        self.positions = {}  # Dictionary mapping symbol to current position
        self.trades = []  # List of completed trades
        self.order_callbacks = []  # List of callbacks for order updates
        
        logger.info(f"Initialized execution engine: {self.name}")
    
    def register_callback(self, callback):
        """
        Register a callback for order updates.
        
        Args:
            callback: Function to call when order status changes
        """
        self.order_callbacks.append(callback)
        logger.info(f"Registered callback for order updates: {callback.__name__}")
    
    def _notify_callbacks(self, order: Order):
        """
        Notify all registered callbacks about an order update.
        
        Args:
            order: Order that was updated
        """
        for callback in self.order_callbacks:
            try:
                callback(order)
            except Exception as e:
                logger.error(f"Error in order callback {callback.__name__}: {e}")
    
    def submit_order(self, order: Order) -> str:
        """
        Submit an order for execution.
        
        Args:
            order: Order to submit
        
        Returns:
            Order ID
        """
        # Store the order
        self.orders[order.order_id] = order
        
        # Update status
        order.update_status(OrderStatus.SUBMITTED)
        
        # Notify callbacks
        self._notify_callbacks(order)
        
        logger.info(f"Submitted order {order.order_id}")
        return order.order_id
    
    def create_and_submit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> str:
        """
        Create and submit an order in one step.
        
        Args:
            symbol: Trading symbol
            side: Buy or sell
            quantity: Order quantity
            order_type: Type of order
            limit_price: Price for limit orders
            stop_price: Price for stop orders
        
        Returns:
            Order ID
        """
        order = Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price
        )
        
        return self.submit_order(order)
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: ID of order to cancel
        
        Returns:
            True if successfully cancelled, False otherwise
        """
        if order_id not in self.orders:
            logger.warning(f"Cannot cancel unknown order {order_id}")
            return False
        
        order = self.orders[order_id]
        if not order.is_active():
            logger.warning(f"Cannot cancel non-active order {order_id} with status {order.status.value}")
            return False
        
        order.cancel()
        self._notify_callbacks(order)
        
        logger.info(f"Cancelled order {order_id}")
        return True
    
    def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancel all active orders, optionally filtered by symbol.
        
        Args:
            symbol: If provided, only cancel orders for this symbol
        
        Returns:
            Number of orders cancelled
        """
        cancelled_count = 0
        
        for order_id, order in list(self.orders.items()):
            if order.is_active() and (symbol is None or order.symbol == symbol):
                if self.cancel_order(order_id):
                    cancelled_count += 1
        
        logger.info(f"Cancelled {cancelled_count} orders")
        return cancelled_count
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get an order by ID.
        
        Args:
            order_id: Order ID to look up
        
        Returns:
            Order object or None if not found
        """
        return self.orders.get(order_id)
    
    def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Get all active orders, optionally filtered by symbol.
        
        Args:
            symbol: If provided, only return orders for this symbol
        
        Returns:
            List of active orders
        """
        return [
            order for order in self.orders.values()
            if order.is_active() and (symbol is None or order.symbol == symbol)
        ]
    
    def update_position(self, symbol: str, quantity: float, price: float, timestamp: Optional[datetime] = None):
        """
        Update the position for a symbol based on a fill.
        
        Args:
            symbol: Trading symbol
            quantity: Signed quantity (positive for buys, negative for sells)
            price: Fill price
            timestamp: Time of the fill
        """
        current_position = self.positions.get(symbol, 0.0)
        new_position = current_position + quantity
        
        # Record the trade
        trade_time = timestamp if timestamp else datetime.now()
        trade = {
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "timestamp": trade_time,
            "previous_position": current_position,
            "new_position": new_position
        }
        self.trades.append(trade)
        
        # Update position
        self.positions[symbol] = new_position
        
        logger.info(f"Updated position for {symbol}: {current_position} -> {new_position} "
                   f"({quantity} @ {price})")
    
    def get_position(self, symbol: str) -> float:
        """
        Get current position for a symbol.
        
        Args:
            symbol: Trading symbol
        
        Returns:
            Current position quantity (positive for long, negative for short, 0 for flat)
        """
        return self.positions.get(symbol, 0.0)
    
    def get_all_positions(self) -> Dict[str, float]:
        """
        Get all current positions.
        
        Returns:
            Dictionary mapping symbols to position quantities
        """
        return self.positions.copy()
    
    def save_state(self, file_path: str):
        """
        Save the current state to a file.
        
        Args:
            file_path: Path to save state to
        """
        state = {
            "orders": {order_id: order.to_dict() for order_id, order in self.orders.items()},
            "positions": self.positions,
            "trades": self.trades
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Saved execution state to {file_path}")
    
    def load_state(self, file_path: str):
        """
        Load state from a file.
        
        Args:
            file_path: Path to load state from
        """
        with open(file_path, 'r') as f:
            state = json.load(f)
        
        self.orders = {order_id: Order.from_dict(order_data) for order_id, order_data in state["orders"].items()}
        self.positions = state["positions"]
        self.trades = state["trades"]
        
        logger.info(f"Loaded execution state from {file_path} with {len(self.orders)} orders and "
                   f"{len(self.positions)} positions")


class SimulatedExecutionEngine(ExecutionEngine):
    """
    Simulated execution engine for backtesting.
    """
    
    def __init__(
        self,
        price_data: pd.DataFrame,
        delay_seconds: float = 0.0,
        fill_probability: float = 1.0,
        slippage_stdev: float = 0.0,
        price_col: str = 'close',
        name: str = "SimulatedExecution"
    ):
        """
        Initialize the simulated execution engine.
        
        Args:
            price_data: DataFrame with price data for simulation
            delay_seconds: Simulated delay before orders are filled
            fill_probability: Probability that a market order gets filled
            slippage_stdev: Standard deviation for random slippage (as fraction of price)
            price_col: Column to use for price data
            name: Name of the execution engine
        """
        super().__init__(name=name)
        self.price_data = price_data
        self.delay_seconds = delay_seconds
        self.fill_probability = fill_probability
        self.slippage_stdev = slippage_stdev
        self.price_col = price_col
        
        # Ensure price data has a DatetimeIndex
        if not isinstance(self.price_data.index, pd.DatetimeIndex):
            raise ValueError("Price data must have a DatetimeIndex")
        
        # Keep track of current simulation time
        self.simulation_time = None
        
        logger.info(f"Initialized simulated execution engine with {len(price_data)} price points, "
                   f"delay={delay_seconds}s, fill_prob={fill_probability}, "
                   f"slippage_stdev={slippage_stdev}")
    
    def set_simulation_time(self, timestamp: datetime):
        """
        Set the current simulation time.
        
        Args:
            timestamp: Current simulation time
        """
        self.simulation_time = timestamp
        
        # Process any pending orders
        self._process_pending_orders()
    
    def _get_current_price(self, symbol: str) -> float:
        """
        Get the current price for a symbol.
        
        Args:
            symbol: Trading symbol
        
        Returns:
            Current price
        """
        if self.simulation_time is None:
            raise ValueError("Simulation time must be set before getting prices")
        
        # Find the closest price data point on or before the current time
        idx = self.price_data.index[self.price_data.index <= self.simulation_time]
        if len(idx) == 0:
            raise ValueError(f"No price data available before {self.simulation_time}")
        
        # Get the most recent price
        latest_idx = idx[-1]
        return self.price_data.loc[latest_idx, self.price_col]
    
    def submit_order(self, order: Order) -> str:
        """
        Submit an order in the simulation.
        
        Args:
            order: Order to submit
        
        Returns:
            Order ID
        """
        # Call parent method to register the order
        order_id = super().submit_order(order)
        
        # If simulation time is set, process pending orders immediately
        if self.simulation_time is not None:
            self._process_pending_orders()
        
        return order_id
    
    def _process_pending_orders(self):
        """Process any pending orders based on current price data."""
        for order_id, order in list(self.orders.items()):
            if order.status == OrderStatus.SUBMITTED:
                # Apply simulated delay
                if self.delay_seconds > 0:
                    order_age = (self.simulation_time - order.timestamp).total_seconds()
                    if order_age < self.delay_seconds:
                        continue  # Not enough time has passed
                
                # Process the order
                self._process_order(order)
    
    def _process_order(self, order: Order):
        """
        Process a single order.
        
        Args:
            order: Order to process
        """
        # Check if order should be filled based on fill probability
        if np.random.random() > self.fill_probability:
            logger.info(f"Order {order.order_id} randomly rejected based on fill_probability")
            order.update_status(OrderStatus.REJECTED)
            self._notify_callbacks(order)
            return
        
        try:
            # Get current price
            current_price = self._get_current_price(order.symbol)
            
            # Apply slippage
            if self.slippage_stdev > 0:
                # Apply random slippage in the adverse direction based on order side
                slippage_factor = np.random.normal(0, self.slippage_stdev)
                if order.side == OrderSide.BUY:
                    # For buys, positive slippage means paying more
                    slippage_factor = abs(slippage_factor)
                else:
                    # For sells, negative slippage means receiving less
                    slippage_factor = -abs(slippage_factor)
                
                fill_price = current_price * (1 + slippage_factor)
            else:
                fill_price = current_price
            
            # For limit orders, check if price is favorable
            if order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and fill_price > order.limit_price:
                    # Price too high for buy limit order
                    return
                elif order.side == OrderSide.SELL and fill_price < order.limit_price:
                    # Price too low for sell limit order
                    return
            
            # For stop orders, check if stop price is triggered
            if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
                if order.side == OrderSide.BUY and current_price < order.stop_price:
                    # Price too low for buy stop order
                    return
                elif order.side == OrderSide.SELL and current_price > order.stop_price:
                    # Price too high for sell stop order
                    return
                
                # If it's a stop-limit order, we also need to check the limit price
                if order.order_type == OrderType.STOP_LIMIT:
                    if order.side == OrderSide.BUY and fill_price > order.limit_price:
                        # Price too high for buy stop-limit order
                        return
                    elif order.side == OrderSide.SELL and fill_price < order.limit_price:
                        # Price too low for sell stop-limit order
                        return
            
            # Fill the order
            order.add_fill(order.quantity, fill_price, self.simulation_time)
            
            # Update position
            position_change = order.quantity if order.side == OrderSide.BUY else -order.quantity
            self.update_position(order.symbol, position_change, fill_price, self.simulation_time)
            
            # Notify callbacks
            self._notify_callbacks(order)
            
        except Exception as e:
            logger.error(f"Error processing order {order.order_id}: {e}")
            order.update_status(OrderStatus.REJECTED)
            self._notify_callbacks(order)


class LiveExecutionEngine(ExecutionEngine):
    """
    Live execution engine that connects to a broker.
    This is a placeholder class - in a real implementation,
    this would connect to a specific broker API.
    """
    
    def __init__(self, broker_config: Dict[str, Any], name: str = "LiveExecution"):
        """
        Initialize the live execution engine.
        
        Args:
            broker_config: Configuration for broker connection
            name: Name of the execution engine
        """
        super().__init__(name=name)
        self.broker_config = broker_config
        self.broker_connection = None
        
        logger.info(f"Initialized live execution engine with broker config: {broker_config}")
    
    def connect(self):
        """Establish connection to the broker."""
        # In a real implementation, this would connect to the broker API
        logger.info("Connecting to broker...")
        self.broker_connection = "MOCK_CONNECTION"
        logger.info("Connected to broker")
    
    def disconnect(self):
        """Disconnect from the broker."""
        # In a real implementation, this would disconnect from the broker API
        logger.info("Disconnecting from broker...")
        self.broker_connection = None
        logger.info("Disconnected from broker")
    
    def check_connection(self) -> bool:
        """
        Check if broker connection is active.
        
        Returns:
            True if connected, False otherwise
        """
        return self.broker_connection is not None
    
    def submit_order(self, order: Order) -> str:
        """
        Submit an order to the broker.
        
        Args:
            order: Order to submit
        
        Returns:
            Order ID
        """
        if not self.check_connection():
            raise ConnectionError("Not connected to broker")
        
        # In a real implementation, this would submit the order to the broker API
        logger.info(f"Submitting order to broker: {order.to_dict()}")
        
        # For demonstration, just simulate successful submission
        return super().submit_order(order)
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order with the broker.
        
        Args:
            order_id: ID of order to cancel
        
        Returns:
            True if successfully cancelled, False otherwise
        """
        if not self.check_connection():
            raise ConnectionError("Not connected to broker")
        
        # In a real implementation, this would cancel the order with the broker API
        logger.info(f"Cancelling order with broker: {order_id}")
        
        # For demonstration, just simulate successful cancellation
        return super().cancel_order(order_id)


def create_execution_engine(config: Dict[str, Any], price_data: Optional[pd.DataFrame] = None) -> ExecutionEngine:
    """
    Factory function to create an execution engine based on configuration.
    
    Args:
        config: Configuration dictionary
        price_data: Price data for simulation (required for simulated execution)
    
    Returns:
        ExecutionEngine instance
    """
    execution_type = config.get("execution_type", "simulated").lower()
    
    if execution_type == "simulated":
        if price_data is None:
            raise ValueError("Price data must be provided for simulated execution")
        
        return SimulatedExecutionEngine(
            price_data=price_data,
            delay_seconds=config.get("delay_seconds", 0.0),
            fill_probability=config.get("fill_probability", 1.0),
            slippage_stdev=config.get("slippage_stdev", 0.0),
            price_col=config.get("price_col", "close"),
            name=config.get("name", "SimulatedExecution")
        )
    
    elif execution_type == "live":
        broker_config = config.get("broker_config", {})
        return LiveExecutionEngine(
            broker_config=broker_config,
            name=config.get("name", "LiveExecution")
        )
    
    else:
        raise ValueError(f"Unknown execution type: {execution_type}")


if __name__ == "__main__":
    # Example usage
    try:
        # Create sample price data for simulation
        index = pd.date_range(start='2020-01-01', end='2020-01-10', freq='D')
        price_data = pd.DataFrame({
            'open': np.linspace(2.0, 2.5, len(index)) + np.random.normal(0, 0.1, len(index)),
            'high': np.linspace(2.1, 2.6, len(index)) + np.random.normal(0, 0.1, len(index)),
            'low': np.linspace(1.9, 2.4, len(index)) + np.random.normal(0, 0.1, len(index)),
            'close': np.linspace(2.0, 2.5, len(index)) + np.random.normal(0, 0.1, len(index)),
            'volume': np.random.randint(100000, 500000, len(index))
        }, index=index)
        
        # Create simulated execution engine
        engine = SimulatedExecutionEngine(
            price_data=price_data,
            delay_seconds=0.5,
            slippage_stdev=0.001,
            fill_probability=0.95
        )
        
        # Define a callback for order updates
        def order_update_callback(order):
            print(f"Order update: {order.order_id} - {order.status.value}")
        
        # Register the callback
        engine.register_callback(order_update_callback)
        
        # Set simulation time to start of the data
        engine.set_simulation_time(index[0])
        
        # Submit market order
        order_id = engine.create_and_submit_order(
            symbol="NG=F",
            side=OrderSide.BUY,
            quantity=1.0,
            order_type=OrderType.MARKET
        )
        
        # Move simulation time forward
        for time_idx in index[1:5]:
            engine.set_simulation_time(time_idx)
            
            # Check position after each time step
            position = engine.get_position("NG=F")
            print(f"Position at {time_idx}: {position}")
        
        # Submit limit order
        limit_order_id = engine.create_and_submit_order(
            symbol="NG=F",
            side=OrderSide.SELL,
            quantity=0.5,
            order_type=OrderType.LIMIT,
            limit_price=2.3
        )
        
        # Move simulation time forward
        for time_idx in index[5:]:
            engine.set_simulation_time(time_idx)
            
            # Check position after each time step
            position = engine.get_position("NG=F")
            print(f"Position at {time_idx}: {position}")
        
        # Save and load state
        output_dir = Path(__file__).parents[2] / 'data' / 'interim'
        os.makedirs(output_dir, exist_ok=True)
        state_file = output_dir / 'execution_state.json'
        
        engine.save_state(str(state_file))
        
        # Create a new engine and load state
        new_engine = SimulatedExecutionEngine(price_data=price_data)
        new_engine.load_state(str(state_file))
        
        # Check that state was loaded correctly
        print(f"Loaded engine has {len(new_engine.orders)} orders and {len(new_engine.positions)} positions")
        print(f"Positions: {new_engine.get_all_positions()}")
        
        logger.info("Example execution completed successfully")
        
    except Exception as e:
        logger.error(f"Error in example: {e}")
        raise 
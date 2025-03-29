"""
Backtesting Engine - Simulate trading strategies on historical data.

This module provides a comprehensive backtesting framework to evaluate
the performance of trading strategies for natural gas.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Import utility logger
from src.utils.logger import get_logger
from src.strategies.alpha_model import AlphaModel
from src.risk.risk_model import RiskModel, create_default_risk_model

# Configure logging
logger = get_logger(__name__)


class Backtester:
    """
    Backtesting engine for evaluating trading strategies.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        alpha_model: AlphaModel,
        risk_model: Optional[RiskModel] = None,
        initial_capital: float = 1_000_000.0,
        transaction_cost: float = 0.0001,  # 1 basis point per trade
        slippage: float = 0.0001,  # 1 basis point per trade
        price_col: str = 'close',
        commission_fixed: float = 0.0,  # Fixed commission per trade
        leverage_limit: float = 1.0,
        use_fractional_sizes: bool = True
    ):
        """
        Initialize backtester with market data and models.
        
        Args:
            data: DataFrame with market data (prices, features)
            alpha_model: Alpha model to generate trading signals
            risk_model: Risk model for position sizing (optional, will use default if None)
            initial_capital: Starting capital for the backtest
            transaction_cost: Cost per trade as a fraction of trade value
            slippage: Trading slippage as a fraction of trade value
            price_col: Column name for price data
            commission_fixed: Fixed commission cost per trade
            leverage_limit: Maximum allowed leverage
            use_fractional_sizes: Whether to allow fractional position sizes
        """
        self.data = data.copy()
        self.alpha_model = alpha_model
        self.risk_model = risk_model if risk_model is not None else create_default_risk_model()
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.price_col = price_col
        self.commission_fixed = commission_fixed
        self.leverage_limit = leverage_limit
        self.use_fractional_sizes = use_fractional_sizes
        
        # Validate data
        required_cols = [price_col]
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Initialize results storage
        self.positions = pd.Series(0, index=self.data.index)
        self.capital_history = pd.Series(initial_capital, index=self.data.index)
        self.trade_history = []
        
        logger.info(f"Initialized backtester with {len(self.data)} bars of data")
    
    def run(self) -> Dict[str, Any]:
        """
        Run the backtest.
        
        Returns:
            Dict containing backtest results and performance metrics
        """
        logger.info(f"Starting backtest from {self.data.index[0]} to {self.data.index[-1]}")
        
        # Generate alpha signals
        alpha_signals = self.alpha_model.generate_signals(self.data)
        logger.info(f"Generated alpha signals: {len(alpha_signals[alpha_signals != 0])} active signals")
        
        # Apply risk model for position sizing
        raw_positions = pd.Series(0.0, index=self.data.index)
        for i in range(len(self.data)):
            today = self.data.index[i]
            signal = alpha_signals.at[today]
            position = self.risk_model.size_position(signal, self.data.iloc[:i+1])
            raw_positions.at[today] = position
            
        logger.info(f"Sized positions based on risk model")
        
        # Track metrics through the simulation
        current_position = 0
        entry_price = 0
        equity = self.initial_capital
        
        for i in range(1, len(self.data)):
            yesterday = self.data.index[i-1]
            today = self.data.index[i]
            
            # Get yesterday's closing price and today's opening price
            yesterday_price = self.data.at[yesterday, self.price_col]
            today_price = self.data.at[today, self.price_col]
            
            # Get the target position
            target_position = raw_positions.at[today]
            
            # Calculate position change and associated costs
            position_change = target_position - current_position
            
            if position_change != 0:
                # Calculate transaction costs
                execution_price = today_price * (1 + self.slippage * np.sign(position_change))
                value_traded = abs(position_change) * execution_price
                cost = value_traded * self.transaction_cost + self.commission_fixed
                
                # Update equity
                equity -= cost
                
                # Record the trade
                self.trade_history.append({
                    'date': today,
                    'price': execution_price,
                    'size': position_change,
                    'cost': cost,
                    'type': 'buy' if position_change > 0 else 'sell'
                })
                
                # Update position
                current_position = target_position
                entry_price = execution_price if current_position != 0 else 0
            
            # Calculate P&L for the day
            if current_position != 0:
                equity_change = current_position * (today_price - yesterday_price)
                equity += equity_change
            
            # Store position and equity
            self.positions.at[today] = current_position
            self.capital_history.at[today] = equity
        
        # Calculate performance metrics
        results = self._calculate_performance_metrics()
        
        logger.info(f"Completed backtest with final equity: ${results['final_equity']:.2f}")
        return results
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate performance metrics from the backtest results.
        
        Returns:
            Dict of performance metrics
        """
        equity_curve = self.capital_history
        
        # Calculate returns
        returns = equity_curve.pct_change().fillna(0)
        
        # Basic metrics
        total_return = equity_curve.iloc[-1] / self.initial_capital - 1
        annual_return = ((1 + total_return) ** (252 / len(returns)) - 1)
        daily_std = returns.std()
        sharpe = np.sqrt(252) * returns.mean() / daily_std if daily_std > 0 else 0
        
        # Drawdown analysis
        drawdown = 1 - equity_curve / equity_curve.cummax()
        max_drawdown = drawdown.max()
        
        # Trade analysis
        if self.trade_history:
            trade_df = pd.DataFrame(self.trade_history)
            num_trades = len(trade_df)
            profit_trades = sum(1 for t in self.trade_history if t['type'] == 'sell' and t['size'] < 0)
            loss_trades = num_trades - profit_trades
            win_rate = profit_trades / num_trades if num_trades > 0 else 0
        else:
            num_trades = 0
            win_rate = 0
        
        return {
            'initial_equity': self.initial_capital,
            'final_equity': equity_curve.iloc[-1],
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'equity_curve': equity_curve,
            'returns': returns,
            'drawdown': drawdown,
            'positions': self.positions
        }
    
    def plot_results(self, figsize: Tuple[int, int] = (15, 15)) -> None:
        """
        Plot backtest results.
        
        Args:
            figsize: Size of the figure (width, height)
        """
        results = self._calculate_performance_metrics()
        
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        
        # Plot equity curve
        axes[0].plot(results['equity_curve'])
        axes[0].set_title('Equity Curve')
        axes[0].set_ylabel('Equity ($)')
        axes[0].grid(True)
        
        # Plot returns
        axes[1].plot(results['returns'].cumsum())
        axes[1].set_title('Cumulative Returns')
        axes[1].set_ylabel('Cumulative Return')
        axes[1].grid(True)
        
        # Plot drawdown
        axes[2].fill_between(results['drawdown'].index, 0, results['drawdown'], color='red', alpha=0.3)
        axes[2].set_title('Drawdown')
        axes[2].set_ylabel('Drawdown (%)')
        axes[2].grid(True)
        
        # Plot positions
        axes[3].plot(self.positions)
        axes[3].set_title('Position Size')
        axes[3].set_ylabel('Position')
        axes[3].set_xlabel('Date')
        axes[3].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Print performance summary
        print(f"Performance Summary:")
        print(f"Initial Equity: ${results['initial_equity']:.2f}")
        print(f"Final Equity: ${results['final_equity']:.2f}")
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Annual Return: {results['annual_return']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"Number of Trades: {results['num_trades']}")
        print(f"Win Rate: {results['win_rate']:.2%}")


class MultiStrategyBacktester:
    """
    Run and compare multiple strategies in a single backtest.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        strategies: Dict[str, Tuple[AlphaModel, RiskModel]],
        initial_capital: float = 1_000_000.0,
        transaction_cost: float = 0.0001
    ):
        """
        Initialize with multiple strategies.
        
        Args:
            data: DataFrame with market data
            strategies: Dict of strategy_name -> (alpha_model, risk_model) pairs
            initial_capital: Starting capital for each strategy
            transaction_cost: Cost per trade as a fraction of trade value
        """
        self.data = data.copy()
        self.strategies = strategies
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.results = {}
        
        logger.info(f"Initialized multi-strategy backtester with {len(strategies)} strategies")
    
    def run(self) -> Dict[str, Dict[str, Any]]:
        """
        Run backtest for all strategies.
        
        Returns:
            Dict of strategy_name -> backtest results
        """
        for name, (alpha, risk) in self.strategies.items():
            logger.info(f"Running backtest for strategy: {name}")
            
            backtester = Backtester(
                data=self.data,
                alpha_model=alpha,
                risk_model=risk,
                initial_capital=self.initial_capital,
                transaction_cost=self.transaction_cost
            )
            
            self.results[name] = backtester.run()
        
        return self.results
    
    def plot_equity_curves(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot equity curves for all strategies.
        
        Args:
            figsize: Size of the figure (width, height)
        """
        plt.figure(figsize=figsize)
        
        for name, result in self.results.items():
            plt.plot(result['equity_curve'], label=name)
        
        plt.title('Strategy Comparison: Equity Curves')
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def compare_metrics(self) -> pd.DataFrame:
        """
        Compare performance metrics across strategies.
        
        Returns:
            DataFrame with performance metrics for each strategy
        """
        metrics = ['total_return', 'annual_return', 'sharpe_ratio', 
                  'max_drawdown', 'num_trades', 'win_rate']
        
        comparison = {}
        
        for name, result in self.results.items():
            comparison[name] = {metric: result[metric] for metric in metrics}
        
        return pd.DataFrame(comparison).T 
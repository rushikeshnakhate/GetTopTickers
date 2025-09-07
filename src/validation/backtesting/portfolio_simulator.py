"""
Portfolio Simulator for simulating actual trading performance.

This module simulates the actual trading of selected tickers to provide
realistic performance estimates including transaction costs and market impact.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime, timedelta
import yfinance as yf

logger = logging.getLogger(__name__)


class PortfolioSimulator:
    """Simulate actual trading performance with realistic constraints."""
    
    def __init__(self, initial_capital: float = 1000000, 
                 transaction_cost: float = 0.001,
                 market_impact: float = 0.0005):
        """
        Initialize the PortfolioSimulator.
        
        Args:
            initial_capital: Initial portfolio capital
            transaction_cost: Transaction cost as percentage of trade value
            market_impact: Market impact cost as percentage of trade value
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.market_impact = market_impact
        self.simulation_results = []
        
    def simulate_portfolio_trading(self, selected_tickers: List[str],
                                 start_date: str, end_date: str,
                                 rebalance_frequency: str = 'monthly',
                                 rebalance_threshold: float = 0.05) -> Dict:
        """
        Simulate portfolio trading with realistic constraints.
        
        Args:
            selected_tickers: List of selected tickers
            start_date: Start date for simulation
            end_date: End date for simulation
            rebalance_frequency: Rebalancing frequency
            rebalance_threshold: Threshold for triggering rebalancing
            
        Returns:
            Dictionary with simulation results
        """
        logger.info(f"Simulating portfolio trading from {start_date} to {end_date}")
        
        try:
            # Get price data for all tickers
            price_data = self._get_price_data(selected_tickers, start_date, end_date)
            
            if price_data.empty:
                return {'error': 'Unable to fetch price data for simulation'}
            
            # Initialize portfolio
            portfolio = self._initialize_portfolio(selected_tickers)
            
            # Run simulation
            simulation_log = []
            current_date = pd.to_datetime(start_date)
            end_date_dt = pd.to_datetime(end_date)
            
            while current_date <= end_date_dt:
                # Check if rebalancing is needed
                if self._should_rebalance(current_date, rebalance_frequency):
                    # Get current prices
                    current_prices = self._get_current_prices(price_data, current_date)
                    
                    if not current_prices.empty:
                        # Rebalance portfolio
                        rebalance_result = self._rebalance_portfolio(
                            portfolio, current_prices, rebalance_threshold
                        )
                        
                        # Update portfolio
                        portfolio = rebalance_result['new_portfolio']
                        
                        # Log rebalancing
                        simulation_log.append({
                            'date': current_date,
                            'action': 'rebalance',
                            'portfolio_value': rebalance_result['portfolio_value'],
                            'transaction_cost': rebalance_result['transaction_cost'],
                            'market_impact': rebalance_result['market_impact'],
                            'holdings': portfolio.copy()
                        })
                
                # Calculate daily portfolio value
                current_prices = self._get_current_prices(price_data, current_date)
                if not current_prices.empty:
                    portfolio_value = self._calculate_portfolio_value(portfolio, current_prices)
                    
                    simulation_log.append({
                        'date': current_date,
                        'action': 'valuation',
                        'portfolio_value': portfolio_value,
                        'holdings': portfolio.copy()
                    })
                
                # Move to next day
                current_date += timedelta(days=1)
            
            # Calculate final performance metrics
            performance_metrics = self._calculate_simulation_performance(simulation_log)
            
            results = {
                'simulation_parameters': {
                    'initial_capital': self.initial_capital,
                    'transaction_cost': self.transaction_cost,
                    'market_impact': self.market_impact,
                    'rebalance_frequency': rebalance_frequency,
                    'rebalance_threshold': rebalance_threshold
                },
                'simulation_log': simulation_log,
                'performance_metrics': performance_metrics,
                'final_portfolio': portfolio
            }
            
            # Store results
            self.simulation_results.append({
                'timestamp': datetime.now(),
                'selected_tickers': selected_tickers,
                'start_date': start_date,
                'end_date': end_date,
                'results': results
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in portfolio simulation: {str(e)}")
            return {'error': str(e)}
    
    def simulate_with_transaction_costs(self, selected_tickers: List[str],
                                      start_date: str, end_date: str,
                                      cost_scenarios: List[Dict]) -> Dict:
        """
        Simulate portfolio with different transaction cost scenarios.
        
        Args:
            selected_tickers: List of selected tickers
            start_date: Start date for simulation
            end_date: End date for simulation
            cost_scenarios: List of cost scenario dictionaries
            
        Returns:
            Dictionary with cost scenario results
        """
        logger.info(f"Simulating with {len(cost_scenarios)} cost scenarios")
        
        scenario_results = {}
        
        for i, scenario in enumerate(cost_scenarios):
            scenario_name = scenario.get('name', f'scenario_{i+1}')
            
            # Update simulator parameters
            original_transaction_cost = self.transaction_cost
            original_market_impact = self.market_impact
            
            self.transaction_cost = scenario.get('transaction_cost', self.transaction_cost)
            self.market_impact = scenario.get('market_impact', self.market_impact)
            
            # Run simulation
            simulation_result = self.simulate_portfolio_trading(
                selected_tickers, start_date, end_date,
                scenario.get('rebalance_frequency', 'monthly')
            )
            
            scenario_results[scenario_name] = {
                'scenario_parameters': scenario,
                'simulation_result': simulation_result
            }
            
            # Restore original parameters
            self.transaction_cost = original_transaction_cost
            self.market_impact = original_market_impact
        
        return {
            'cost_scenarios': scenario_results,
            'scenario_comparison': self._compare_cost_scenarios(scenario_results)
        }
    
    def simulate_liquidity_constraints(self, selected_tickers: List[str],
                                     start_date: str, end_date: str,
                                     liquidity_constraints: Dict) -> Dict:
        """
        Simulate portfolio with liquidity constraints.
        
        Args:
            selected_tickers: List of selected tickers
            start_date: Start date for simulation
            end_date: End date for simulation
            liquidity_constraints: Dictionary with liquidity constraints
            
        Returns:
            Dictionary with liquidity-constrained simulation results
        """
        logger.info("Simulating with liquidity constraints")
        
        # This would implement liquidity constraints
        # For now, return a placeholder
        return {
            'liquidity_constraints': liquidity_constraints,
            'simulation_result': 'Liquidity constraints simulation not yet implemented'
        }
    
    def _get_price_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Get price data for all tickers."""
        try:
            data = yf.download(tickers, start=start_date, end=end_date, 
                             progress=False, group_by='ticker')
            
            if data.empty:
                return pd.DataFrame()
            
            # Handle different data structures
            if isinstance(data.columns, pd.MultiIndex):
                close_prices = data.xs('Adj Close', level=1, axis=1)
            else:
                close_prices = data['Adj Close'] if 'Adj Close' in data.columns else data
            
            return close_prices
            
        except Exception as e:
            logger.error(f"Error fetching price data: {str(e)}")
            return pd.DataFrame()
    
    def _initialize_portfolio(self, tickers: List[str]) -> Dict:
        """Initialize portfolio with equal weights."""
        portfolio = {}
        weight_per_ticker = 1.0 / len(tickers)
        
        for ticker in tickers:
            portfolio[ticker] = {
                'weight': weight_per_ticker,
                'shares': 0,
                'value': 0
            }
        
        return portfolio
    
    def _should_rebalance(self, current_date: pd.Timestamp, frequency: str) -> bool:
        """Check if rebalancing should occur on current date."""
        if frequency == 'daily':
            return True
        elif frequency == 'weekly':
            return current_date.weekday() == 0  # Monday
        elif frequency == 'monthly':
            return current_date.day == 1
        elif frequency == 'quarterly':
            return current_date.month in [1, 4, 7, 10] and current_date.day == 1
        else:
            return False
    
    def _get_current_prices(self, price_data: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
        """Get prices for a specific date."""
        try:
            # Find the closest available date
            available_dates = price_data.index[price_data.index <= date]
            if len(available_dates) == 0:
                return pd.Series(dtype=float)
            
            closest_date = available_dates[-1]
            return price_data.loc[closest_date]
            
        except Exception as e:
            logger.warning(f"Error getting prices for {date}: {str(e)}")
            return pd.Series(dtype=float)
    
    def _rebalance_portfolio(self, current_portfolio: Dict, current_prices: pd.Series,
                           rebalance_threshold: float) -> Dict:
        """Rebalance portfolio to target weights."""
        # Calculate current portfolio value
        current_value = self._calculate_portfolio_value(current_portfolio, current_prices)
        
        # Calculate target values (equal weight)
        num_tickers = len(current_portfolio)
        target_weight = 1.0 / num_tickers
        target_value_per_ticker = current_value * target_weight
        
        # Calculate new shares and transaction costs
        total_transaction_cost = 0
        total_market_impact = 0
        new_portfolio = {}
        
        for ticker in current_portfolio.keys():
            if ticker in current_prices.index:
                current_price = current_prices[ticker]
                target_shares = target_value_per_ticker / current_price
                current_shares = current_portfolio[ticker]['shares']
                
                # Calculate trade size
                shares_to_trade = target_shares - current_shares
                trade_value = abs(shares_to_trade) * current_price
                
                # Calculate costs
                transaction_cost = trade_value * self.transaction_cost
                market_impact = trade_value * self.market_impact
                
                total_transaction_cost += transaction_cost
                total_market_impact += market_impact
                
                # Update portfolio
                new_portfolio[ticker] = {
                    'weight': target_weight,
                    'shares': target_shares,
                    'value': target_value_per_ticker
                }
        
        # Adjust for transaction costs
        net_value = current_value - total_transaction_cost - total_market_impact
        
        return {
            'new_portfolio': new_portfolio,
            'portfolio_value': net_value,
            'transaction_cost': total_transaction_cost,
            'market_impact': total_market_impact
        }
    
    def _calculate_portfolio_value(self, portfolio: Dict, current_prices: pd.Series) -> float:
        """Calculate current portfolio value."""
        total_value = 0
        
        for ticker, holding in portfolio.items():
            if ticker in current_prices.index:
                current_price = current_prices[ticker]
                shares = holding['shares']
                value = shares * current_price
                total_value += value
                
                # Update holding value
                holding['value'] = value
        
        return total_value
    
    def _calculate_simulation_performance(self, simulation_log: List[Dict]) -> Dict:
        """Calculate performance metrics from simulation log."""
        if not simulation_log:
            return {'error': 'No simulation data available'}
        
        # Extract portfolio values
        portfolio_values = []
        transaction_costs = []
        market_impacts = []
        
        for log_entry in simulation_log:
            if log_entry['action'] == 'valuation':
                portfolio_values.append(log_entry['portfolio_value'])
            elif log_entry['action'] == 'rebalance':
                transaction_costs.append(log_entry['transaction_cost'])
                market_impacts.append(log_entry['market_impact'])
        
        if not portfolio_values:
            return {'error': 'No portfolio values in simulation log'}
        
        # Calculate returns
        portfolio_values = pd.Series(portfolio_values)
        returns = portfolio_values.pct_change().dropna()
        
        # Calculate performance metrics
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        annual_return = (1 + returns.mean()) ** 252 - 1
        volatility = returns.std() * (252 ** 0.5)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Calculate drawdown
        max_drawdown = self._calculate_max_drawdown(returns)
        
        # Calculate costs
        total_transaction_cost = sum(transaction_costs)
        total_market_impact = sum(market_impacts)
        total_costs = total_transaction_cost + total_market_impact
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_transaction_cost': total_transaction_cost,
            'total_market_impact': total_market_impact,
            'total_costs': total_costs,
            'cost_ratio': total_costs / self.initial_capital,
            'net_return': total_return - (total_costs / self.initial_capital),
            'num_rebalances': len(transaction_costs)
        }
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if returns.empty:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _compare_cost_scenarios(self, scenario_results: Dict) -> Dict:
        """Compare results across different cost scenarios."""
        comparison = {}
        
        for scenario_name, result in scenario_results.items():
            if 'simulation_result' in result and 'performance_metrics' in result['simulation_result']:
                metrics = result['simulation_result']['performance_metrics']
                comparison[scenario_name] = {
                    'total_return': metrics.get('total_return', 0),
                    'annual_return': metrics.get('annual_return', 0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'total_costs': metrics.get('total_costs', 0),
                    'net_return': metrics.get('net_return', 0)
                }
        
        return comparison
    
    def get_simulation_history(self) -> pd.DataFrame:
        """Get history of portfolio simulations."""
        if not self.simulation_results:
            return pd.DataFrame()
        
        history_data = []
        for result in self.simulation_results:
            if 'performance_metrics' in result['results']:
                metrics = result['results']['performance_metrics']
                history_data.append({
                    'timestamp': result['timestamp'],
                    'start_date': result['start_date'],
                    'end_date': result['end_date'],
                    'num_tickers': len(result['selected_tickers']),
                    'total_return': metrics.get('total_return', 0),
                    'annual_return': metrics.get('annual_return', 0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'total_costs': metrics.get('total_costs', 0)
                })
        
        return pd.DataFrame(history_data)
    
    def clear_history(self):
        """Clear simulation history."""
        self.simulation_results.clear()
        logger.info("Portfolio simulation history cleared")

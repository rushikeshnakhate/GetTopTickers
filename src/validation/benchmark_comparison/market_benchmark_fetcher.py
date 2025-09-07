"""
Market Benchmark Fetcher for NSE benchmark data.

This module fetches and calculates performance metrics for various NSE benchmarks
including NSE 50, NSE 100, and NSE 500 for comparison with PyPortTickerSelector results.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class MarketBenchmarkFetcher:
    """Fetch market benchmark data for comparison with PyPortTickerSelector results."""
    
    # NSE 50 (Nifty 50) constituents - top 50 companies by market cap
    NSE_50_TICKERS = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'BHARTIARTL.NS', 'INFY.NS',
        'ICICIBANK.NS', 'SBIN.NS', 'LICI.NS', 'ITC.NS', 'HINDUNILVR.NS',
        'LT.NS', 'KOTAKBANK.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS',
        'SUNPHARMA.NS', 'NESTLEIND.NS', 'ULTRACEMCO.NS', 'TITAN.NS', 'POWERGRID.NS',
        'NTPC.NS', 'ONGC.NS', 'COALINDIA.NS', 'TECHM.NS', 'WIPRO.NS',
        'HCLTECH.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'DRREDDY.NS', 'CIPLA.NS',
        'APOLLOHOSP.NS', 'DIVISLAB.NS', 'EICHERMOT.NS', 'GRASIM.NS', 'HDFCLIFE.NS',
        'HEROMOTOCO.NS', 'HINDALCO.NS', 'JSWSTEEL.NS', 'M&M.NS', 'SBILIFE.NS',
        'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'UPL.NS', 'BRITANNIA.NS',
        'ADANIPORTS.NS', 'ADANIENT.NS', 'ADANIGREEN.NS', 'ADANITRANS.NS', 'ADANIPOWER.NS'
    ]
    
    # NSE 100 constituents (subset for demonstration - in practice, you'd have all 100)
    NSE_100_TICKERS = NSE_50_TICKERS + [
        'BAJAJHLDNG.NS', 'BERGEPAINT.NS', 'BIOCON.NS', 'BOSCHLTD.NS', 'CADILAHC.NS',
        'CHOLAFIN.NS', 'COLPAL.NS', 'CONCOR.NS', 'DABUR.NS', 'GAIL.NS',
        'GODREJCP.NS', 'GODREJPROP.NS', 'HAVELLS.NS', 'HDFCAMC.NS', 'IDBI.NS',
        'IDFCFIRSTB.NS', 'INDUSINDBK.NS', 'INFIBEAM.NS', 'IRCTC.NS', 'JINDALSTEL.NS',
        'LALPATHLAB.NS', 'LUPIN.NS', 'MCDOWELL-N.NS', 'MINDTREE.NS', 'MOTHERSON.NS',
        'MPHASIS.NS', 'MRF.NS', 'MUTHOOTFIN.NS', 'NAUKRI.NS', 'PAGEIND.NS',
        'PETRONET.NS', 'PIDILITIND.NS', 'PNB.NS', 'RBLBANK.NS', 'SAIL.NS',
        'SHREECEM.NS', 'SIEMENS.NS', 'TATACOMM.NS', 'TORNTPHARM.NS', 'VEDL.NS',
        'VOLTAS.NS', 'ZEEL.NS', 'ZOMATO.NS', 'PAYTM.NS', 'NYKAA.NS',
        'POLICYBZR.NS', 'DELTACORP.NS', 'BANDHANBNK.NS', 'FEDERALBNK.NS', 'IDEA.NS'
    ]
    
    # NSE 500 constituents (subset for demonstration)
    NSE_500_TICKERS = NSE_100_TICKERS + [
        'ABBOTINDIA.NS', 'ACC.NS', 'AUBANK.NS', 'ASTRAL.NS', 'BALRAMCHIN.NS',
        'BANDHANBNK.NS', 'BATAINDIA.NS', 'BHARATFORG.NS', 'BHEL.NS', 'BPCL.NS',
        'CANFINHOME.NS', 'CENTRALBK.NS', 'CROMPTON.NS', 'CUMMINSIND.NS', 'DALBHARAT.NS',
        'DEEPAKNTR.NS', 'DIXON.NS', 'DMART.NS', 'ESCORTS.NS', 'EXIDEIND.NS',
        'FEDERALBNK.NS', 'GLENMARK.NS', 'GODREJIND.NS', 'GODREJAGRO.NS', 'GSPL.NS',
        'HINDPETRO.NS', 'HINDZINC.NS', 'IBULHSGFIN.NS', 'ICICIGI.NS', 'ICICIPRULI.NS',
        'IDEA.NS', 'IGL.NS', 'INDIANB.NS', 'INDIGO.NS', 'IOC.NS',
        'IPCALAB.NS', 'JUBLFOOD.NS', 'JUSTDIAL.NS', 'KANSBANK.NS', 'LALPATHLAB.NS',
        'LICHSGFIN.NS', 'LTI.NS', 'LTTS.NS', 'MANAPPURAM.NS', 'MARICO.NS',
        'MCDOWELL-N.NS', 'MINDTREE.NS', 'MOTHERSON.NS', 'MPHASIS.NS', 'MRF.NS',
        'MUTHOOTFIN.NS', 'NAUKRI.NS', 'NMDC.NS', 'OBEROI.NS', 'OFSS.NS',
        'PAGEIND.NS', 'PEL.NS', 'PETRONET.NS', 'PIDILITIND.NS', 'PNB.NS',
        'POLYCAB.NS', 'PVR.NS', 'RBLBANK.NS', 'RECLTD.NS', 'RELAXO.NS',
        'SAIL.NS', 'SHREECEM.NS', 'SIEMENS.NS', 'SRF.NS', 'STAR.NS',
        'SUNTV.NS', 'TATACOMM.NS', 'TATACONSUM.NS', 'TATAPOWER.NS', 'TECHM.NS',
        'TORNTPHARM.NS', 'TRENT.NS', 'TVSMOTOR.NS', 'UBL.NS', 'VEDL.NS',
        'VOLTAS.NS', 'WHIRLPOOL.NS', 'YESBANK.NS', 'ZEEL.NS', 'ZOMATO.NS'
    ]
    
    BENCHMARKS = {
        'nse_50': NSE_50_TICKERS,
        'nse_100': NSE_100_TICKERS,
        'nse_500': NSE_500_TICKERS
    }
    
    def __init__(self):
        """Initialize the MarketBenchmarkFetcher."""
        self.cache = {}
        
    def get_benchmark_tickers(self, benchmark: str) -> List[str]:
        """
        Get benchmark ticker list.
        
        Args:
            benchmark: Benchmark name ('nse_50', 'nse_100', 'nse_500')
            
        Returns:
            List of ticker symbols
        """
        if benchmark not in self.BENCHMARKS:
            raise ValueError(f"Unknown benchmark: {benchmark}. Available: {list(self.BENCHMARKS.keys())}")
        
        return self.BENCHMARKS[benchmark].copy()
    
    def get_benchmark_performance(self, benchmark: str, start_date: str, 
                                end_date: str, rebalance_frequency: str = 'monthly') -> Dict:
        """
        Calculate benchmark portfolio performance.
        
        Args:
            benchmark: Benchmark name ('nse_50', 'nse_100', 'nse_500')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            rebalance_frequency: Rebalancing frequency ('daily', 'monthly', 'quarterly')
            
        Returns:
            Dictionary with performance metrics
        """
        cache_key = f"{benchmark}_{start_date}_{end_date}_{rebalance_frequency}"
        
        if cache_key in self.cache:
            logger.info(f"Returning cached benchmark performance for {benchmark}")
            return self.cache[cache_key]
        
        try:
            tickers = self.get_benchmark_tickers(benchmark)
            logger.info(f"Fetching data for {len(tickers)} tickers in {benchmark}")
            
            # Fetch data with error handling
            data = yf.download(tickers, start=start_date, end=end_date, 
                             progress=False, group_by='ticker')
            
            if data.empty:
                raise ValueError(f"No data found for benchmark {benchmark}")
            
            # Handle different data structures from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                # Multi-level columns
                close_prices = data.xs('Adj Close', level=1, axis=1)
            else:
                # Single level columns
                close_prices = data['Adj Close'] if 'Adj Close' in data.columns else data
            
            # Clean data
            close_prices = close_prices.dropna(how='all')
            
            if close_prices.empty:
                raise ValueError(f"No valid price data for benchmark {benchmark}")
            
            # Calculate returns
            returns = close_prices.pct_change().dropna()
            
            # Apply rebalancing frequency
            if rebalance_frequency == 'monthly':
                returns = returns.resample('M').last()
            elif rebalance_frequency == 'quarterly':
                returns = returns.resample('Q').last()
            # For daily, keep as is
            
            # Equal weight portfolio
            portfolio_returns = returns.mean(axis=1)
            
            # Calculate metrics
            metrics = self._calculate_metrics(portfolio_returns)
            metrics['benchmark'] = benchmark
            metrics['period'] = f"{start_date} to {end_date}"
            metrics['rebalance_frequency'] = rebalance_frequency
            metrics['num_tickers'] = len(tickers)
            
            # Cache results
            self.cache[cache_key] = metrics
            
            logger.info(f"Successfully calculated performance for {benchmark}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating benchmark performance for {benchmark}: {str(e)}")
            raise
    
    def _calculate_metrics(self, returns: pd.Series) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            returns: Series of portfolio returns
            
        Returns:
            Dictionary with performance metrics
        """
        if returns.empty:
            return self._get_empty_metrics()
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + returns.mean()) ** 252 - 1
        volatility = returns.std() * (252 ** 0.5)
        
        # Risk-adjusted metrics
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Downside metrics
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * (252 ** 0.5) if len(downside_returns) > 0 else 0
        sortino_ratio = annual_return / downside_volatility if downside_volatility > 0 else 0
        
        # Drawdown metrics
        max_drawdown = self._calculate_max_drawdown(returns)
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate and other metrics
        win_rate = (returns > 0).mean()
        avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
        avg_loss = returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0
        profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Skewness and Kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_loss_ratio': profit_loss_ratio,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'num_observations': len(returns)
        }
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            returns: Series of returns
            
        Returns:
            Maximum drawdown as a negative percentage
        """
        if returns.empty:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _get_empty_metrics(self) -> Dict:
        """Return empty metrics dictionary."""
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_loss_ratio': 0.0,
            'skewness': 0.0,
            'kurtosis': 0.0,
            'num_observations': 0
        }
    
    def get_benchmark_returns_series(self, benchmark: str, start_date: str, 
                                   end_date: str) -> pd.Series:
        """
        Get benchmark returns series for statistical testing.
        
        Args:
            benchmark: Benchmark name
            start_date: Start date
            end_date: End date
            
        Returns:
            Series of benchmark returns
        """
        try:
            tickers = self.get_benchmark_tickers(benchmark)
            data = yf.download(tickers, start=start_date, end=end_date, 
                             progress=False, group_by='ticker')
            
            if isinstance(data.columns, pd.MultiIndex):
                close_prices = data.xs('Adj Close', level=1, axis=1)
            else:
                close_prices = data['Adj Close'] if 'Adj Close' in data.columns else data
            
            close_prices = close_prices.dropna(how='all')
            returns = close_prices.pct_change().dropna()
            portfolio_returns = returns.mean(axis=1)
            
            return portfolio_returns
            
        except Exception as e:
            logger.error(f"Error fetching benchmark returns for {benchmark}: {str(e)}")
            return pd.Series(dtype=float)
    
    def clear_cache(self):
        """Clear the performance cache."""
        self.cache.clear()
        logger.info("Benchmark performance cache cleared")

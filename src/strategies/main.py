import logging
import pandas as pd
from typing import Dict, List

from pandas import DataFrame
from tabulate import tabulate

from src.strategies.bollinger_bands_breakout_strategy import BollingerBandsBreakoutStrategy
from src.strategies.breakout import BreakoutStrategy
from src.strategies.keltner_channel_breakout_strategy import KeltnerChannelBreakoutStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.top_gainers.top_gainers import TopGainersStrategy
from src.strategies.top_gainers.top_gainers_RSI_Strategy import TopGainersRSIStrategy
from src.strategies.top_gainers.top_gainers_combined_strategy import TopGainersCombinedStrategy
from src.strategies.top_loosers.top_losers import TopLosersStrategy


class StrategyRunner:
    """Class to execute multiple trading strategies and aggregate results."""

    def __init__(self, df: pd.DataFrame, top_n: int = 5):
        self.df = df
        self.top_n = top_n

    def get_stocks_for_all_strategies(self) -> Dict[str, List[dict]]:
        """Runs multiple strategies and returns their results."""
        strategies = {
            "KeltnerChannelBreakoutStrategy": KeltnerChannelBreakoutStrategy(self.df, self.top_n),
            "BollingerBandsBreakoutStrategy": BollingerBandsBreakoutStrategy(self.df, self.top_n),
            # "TopGainersCombinedStrategy": TopGainersCombinedStrategy(self.df, self.top_n),
            # "TopGainersRSIStrategy": TopGainersRSIStrategy(self.df, self.top_n),
            # "TopGainersStrategy": TopGainersStrategy(self.df, self.top_n),
            # "TopLosersStrategy": TopLosersStrategy(self.df, self.top_n),
            # "BreakoutStrategy": BreakoutStrategy(self.df, self.top_n),
            # "MeanReversionStrategy": MeanReversionStrategy(self.df, self.top_n),
            # "MomentumStrategy": MomentumStrategy(self.df, self.top_n),\
        }

        results = {}
        for name, strategy in strategies.items():
            result = strategy.run()
            results[name] = result
            logging.info(f"{name} Strategy Results:")
            logging.info(tabulate(result, headers='keys', tablefmt='pretty'))
        return results


def run_all_strategy(df: DataFrame, top_n: int = 5):
    strategy_runner = StrategyRunner(df, top_n)
    strategy_results = strategy_runner.get_stocks_for_all_strategies()
    return strategy_results

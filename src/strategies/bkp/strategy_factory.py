import logging
from pathlib import Path

from src.strategies.bkp.mean_reversion import MeanReversionStrategy
from src.strategies.bkp.momentum import MomentumStrategy
from src.utils.logging_config import setup_logging

# Setup logging
logger = logging.getLogger(__name__)
project_directory = Path(__file__).resolve().parent.parent
setup_logging(project_directory)


class StrategyFactory:
    @staticmethod
    def get_strategy(strategy_name, stock_data):
        strategies = {
            "momentum": MomentumStrategy,
            "mean_reversion": MeanReversionStrategy
        }
        return strategies.get(strategy_name)(stock_data)

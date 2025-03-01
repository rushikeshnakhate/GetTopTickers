import pandas as pd
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """
    Base class for all strategies. All strategies should inherit from this class and implement the `run` method.
    """

    def __init__(self, df: pd.DataFrame, top_n: int = 5):
        self.df = df  # DataFrame containing stock data and indicators
        self.top_n = top_n

    @abstractmethod
    def run(self):
        """
        Each subclass will implement the `run` method for their specific strategy.
        """
        pass

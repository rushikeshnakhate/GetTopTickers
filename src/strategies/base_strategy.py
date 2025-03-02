import pandas as pd
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    def __init__(self, df: pd.DataFrame, top_n: int = 5):
        self.df = df
        self.top_n = top_n

    @abstractmethod
    def run(self):
        pass

    def get_tickers(self, sorted_df: pd.DataFrame) -> list:
        """
        Returns a list of tickers in the proper order.
        """
        return sorted_df['Ticker'].tolist()[:self.top_n]

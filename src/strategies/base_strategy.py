import logging
from typing import Union

import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseStrategy:
    def __init__(self, df: pd.DataFrame, top_n: int = 5):
        self.df = df
        self.top_n = top_n

    def run(self) -> Union[pd.DataFrame, list]:
        raise NotImplementedError("Subclasses must implement the 'run' method.")

    def get_tickers(self, sorted_df: Union[pd.DataFrame, list]) -> list:
        """
        Returns a list of tickers in the proper order.
        Handles both DataFrame and list inputs.
        """
        if isinstance(sorted_df, pd.DataFrame):
            if 'Ticker' not in sorted_df.columns:
                raise KeyError("The 'Ticker' column is missing in the DataFrame")
            return sorted_df['Ticker'].tolist()[:self.top_n]
        elif isinstance(sorted_df, list):
            return sorted_df[:self.top_n]
        else:
            raise TypeError(f"Expected a pandas DataFrame or list, but got {type(sorted_df)}")

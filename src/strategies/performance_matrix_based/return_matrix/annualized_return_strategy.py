# Return-Based Strategies
from src.strategies.base_strategy import BaseStrategy


class AnnualizedReturnStrategy(BaseStrategy):
    def run(self):
        sorted_df = self.df.nlargest(self.top_n, 'AnnualizedReturn')
        return self.get_tickers(sorted_df)

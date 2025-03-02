# Win/Loss Ratio Strategies
from src.strategies.base_strategy import BaseStrategy


class WinRateStrategy(BaseStrategy):
    def run(self):
        sorted_df = self.df.nlargest(self.top_n, 'WinRate')
        return self.get_tickers(sorted_df)

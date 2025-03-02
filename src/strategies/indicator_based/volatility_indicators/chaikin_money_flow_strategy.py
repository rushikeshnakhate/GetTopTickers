# Volume-Based Strategies
from src.strategies.base_strategy import BaseStrategy


class ChaikinMoneyFlowStrategy(BaseStrategy):
    def run(self):
        self.df = self.df[self.df['ChaikinMoneyFlow'] > 0]
        sorted_df = self.df.nlargest(self.top_n, 'ChaikinMoneyFlow')
        return self.get_tickers(sorted_df)

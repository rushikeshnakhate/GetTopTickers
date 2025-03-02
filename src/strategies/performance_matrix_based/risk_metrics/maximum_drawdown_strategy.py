# Drawdown and Risk Strategies
from src.strategies.base_strategy import BaseStrategy


class MaximumDrawdownStrategy(BaseStrategy):
    def run(self):
        sorted_df = self.df.nsmallest(self.top_n, 'MaximumDrawdown')
        return self.get_tickers(sorted_df)

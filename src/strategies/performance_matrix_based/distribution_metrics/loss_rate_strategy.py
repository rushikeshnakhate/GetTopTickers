from src.strategies.base_strategy import BaseStrategy


class LossRateStrategy(BaseStrategy):
    def run(self):
        sorted_df = self.df.nsmallest(self.top_n, 'LossRate')
        return self.get_tickers(sorted_df)

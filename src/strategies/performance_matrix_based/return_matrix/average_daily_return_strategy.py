from src.strategies.base_strategy import BaseStrategy


class AverageDailyReturnStrategy(BaseStrategy):
    def run(self):
        sorted_df = self.df.nlargest(self.top_n, 'AverageDailyReturn')
        return self.get_tickers(sorted_df)

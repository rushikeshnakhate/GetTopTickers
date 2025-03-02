from src.strategies.base_strategy import BaseStrategy


class SharpeRatioStrategy(BaseStrategy):
    def run(self):
        sorted_df = self.df.nlargest(self.top_n, 'SharpeRatio')
        return self.get_tickers(sorted_df)

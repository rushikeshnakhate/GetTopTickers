from src.strategies.base_strategy import BaseStrategy


class SortinoRatioStrategy(BaseStrategy):
    def run(self):
        sorted_df = self.df.nlargest(self.top_n, 'SortinoRatio')
        return self.get_tickers(sorted_df)

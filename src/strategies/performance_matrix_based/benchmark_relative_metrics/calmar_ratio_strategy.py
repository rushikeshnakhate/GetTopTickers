from src.strategies.base_strategy import BaseStrategy


class CalmarRatioStrategy(BaseStrategy):
    def run(self):
        sorted_df = self.df.nlargest(self.top_n, 'CalmarRatio')
        return self.get_tickers(sorted_df)

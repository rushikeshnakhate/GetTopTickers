from src.strategies.base_strategy import BaseStrategy


class GainToPainRatioStrategy(BaseStrategy):
    def run(self):
        sorted_df = self.df.nlargest(self.top_n, 'GainToPainRatio')
        return self.get_tickers(sorted_df)

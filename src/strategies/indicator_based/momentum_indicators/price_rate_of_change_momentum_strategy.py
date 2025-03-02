from src.strategies.base_strategy import BaseStrategy


class PriceRateOfChangeMomentumStrategy(BaseStrategy):
    def run(self):
        sorted_df = self.df.nlargest(self.top_n, 'PriceRateOfChange')
        return self.get_tickers(sorted_df)

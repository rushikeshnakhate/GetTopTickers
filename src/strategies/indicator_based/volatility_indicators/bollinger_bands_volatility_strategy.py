# Volatility Strategies
from src.strategies.base_strategy import BaseStrategy


class BollingerBandsVolatilityStrategy(BaseStrategy):
    def run(self):
        self.df['BollingerWidth'] = self.df['BollingerBands'].apply(lambda x: x['Upper Band'] - x['Lower Band'])
        sorted_df = self.df.nsmallest(self.top_n, 'BollingerWidth')
        return self.get_tickers(sorted_df)

# Mean Reversion Strategies
from src.strategies.base_strategy import BaseStrategy


class BollingerBandsMeanReversionStrategy(BaseStrategy):
    def run(self):
        self.df['BollingerLower'] = self.df['BollingerBands'].apply(lambda x: x['Lower Band'])
        self.df = self.df[self.df['Close'] <= self.df['BollingerLower']]
        sorted_df = self.df.nsmallest(self.top_n, 'BollingerLower')
        return self.get_tickers(sorted_df)

from src.strategies.base_strategy import BaseStrategy


class KeltnerChannelsVolatilityStrategy(BaseStrategy):
    def run(self):
        self.df['KeltnerWidth'] = self.df['KeltnerChannel'].apply(lambda x: x[0] - x[1])
        sorted_df = self.df.nsmallest(self.top_n, 'KeltnerWidth')
        return self.get_tickers(sorted_df)

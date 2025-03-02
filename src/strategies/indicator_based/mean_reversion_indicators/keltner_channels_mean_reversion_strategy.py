from src.strategies.base_strategy import BaseStrategy


class KeltnerChannelsMeanReversionStrategy(BaseStrategy):
    def run(self):
        self.df['KeltnerLower'] = self.df['KeltnerChannel'].apply(lambda x: x[1])
        self.df = self.df[self.df['Close'] <= self.df['KeltnerLower']]
        sorted_df = self.df.nsmallest(self.top_n, 'KeltnerLower')
        return self.get_tickers(sorted_df)

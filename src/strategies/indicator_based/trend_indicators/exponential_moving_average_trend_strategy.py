from src.strategies.base_strategy import BaseStrategy


class ExponentialMovingAverageTrendStrategy(BaseStrategy):
    def run(self):
        self.df = self.df[self.df['Close'] > self.df['ExponentialMovingAverage']]
        sorted_df = self.df.nlargest(self.top_n, 'ExponentialMovingAverage')
        return self.get_tickers(sorted_df)

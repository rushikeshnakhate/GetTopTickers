# Trend-Following Strategies
from src.strategies.base_strategy import BaseStrategy


class MovingAverageTrendStrategy(BaseStrategy):
    def run(self):
        self.df = self.df[self.df['Close'] > self.df['MovingAverage']]
        sorted_df = self.df.nlargest(self.top_n, 'MovingAverage')
        return self.get_tickers(sorted_df)

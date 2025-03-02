from src.strategies.base_strategy import BaseStrategy


class OnBalanceVolumeStrategy(BaseStrategy):
    def run(self):
        self.df = self.df[self.df['OnBalanceVolume'] > self.df['OnBalanceVolume'].shift(1)]
        sorted_df = self.df.nlargest(self.top_n, 'OnBalanceVolume')
        return self.get_tickers(sorted_df)

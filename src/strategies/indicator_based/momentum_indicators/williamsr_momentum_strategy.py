from src.strategies.base_strategy import BaseStrategy


class WilliamsRMomentumStrategy(BaseStrategy):
    def run(self):
        self.df = self.df[self.df['WilliamsR'] < -80]  # Oversold
        sorted_df = self.df.nsmallest(self.top_n, 'WilliamsR')
        return self.get_tickers(sorted_df)

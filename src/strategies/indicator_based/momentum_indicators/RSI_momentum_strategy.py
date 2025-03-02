from src.strategies.base_strategy import BaseStrategy


class RSIMomentumStrategy(BaseStrategy):
    def run(self):
        self.df = self.df[self.df['RelativeStrengthIndex'] < 30]  # Oversold
        sorted_df = self.df.nsmallest(self.top_n, 'RelativeStrengthIndex')
        return self.get_tickers(sorted_df)

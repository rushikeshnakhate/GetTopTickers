import pandas as pd

from src.strategies.base_strategy import BaseStrategy


class TopGainersRSIStrategy(BaseStrategy):
    def __init__(self, df: pd.DataFrame, top_n: int = 5):
        super().__init__(df, top_n)

    def run(self):
        # Example: Selecting stocks with the highest RSI scores
        self.df['Gain'] = self.df['RelativeStrengthIndex']

        # Sort by RSI and select the top N
        sorted_gainers = self.df.nlargest(self.top_n, 'Gain')
        return sorted_gainers[['Ticker', 'Gain']]

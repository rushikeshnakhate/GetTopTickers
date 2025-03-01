import pandas as pd

from src.strategies.base_strategy import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    def __init__(self, df: pd.DataFrame, top_n: int = 5):
        super().__init__(df)
        self.top_n = top_n

    def run(self):
        # Example: Selecting stocks that are far from their mean (moving average)
        mean_reversion_stocks = []
        for idx, row in self.df.iterrows():
            moving_avg = row['MovingAverage']
            current_price = row['ExponentialMovingAverage']
            deviation = abs(current_price - moving_avg)
            mean_reversion_stocks.append(
                {'ticker': row['Ticker'], 'deviation': deviation, 'current_price': current_price})

        # Sorting and selecting top N mean reversion stocks
        mean_reversion_stocks = sorted(mean_reversion_stocks, key=lambda x: x['deviation'], reverse=True)[:self.top_n]
        return mean_reversion_stocks

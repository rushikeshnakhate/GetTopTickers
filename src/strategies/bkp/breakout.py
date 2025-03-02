import pandas as pd

from src.strategies.base_strategy import BaseStrategy


class BreakoutStrategy(BaseStrategy):
    def __init__(self, df: pd.DataFrame, top_n: int = 5):
        super().__init__(df)
        self.top_n = top_n

    def run(self):
        # Example: Stocks that break out above the upper Bollinger Band
        breakout_stocks = []
        for idx, row in self.df.iterrows():
            upper_band = row['BollingerBands']['Upper Band']
            current_price = row['ExponentialMovingAverage']
            if current_price > upper_band:
                breakout_stocks.append(
                    {'ticker': row['Ticker'], 'current_price': current_price, 'upper_band': upper_band})

        # Sorting and selecting top N breakout stocks
        breakout_stocks = sorted(breakout_stocks, key=lambda x: x['current_price'], reverse=True)[:self.top_n]
        return breakout_stocks

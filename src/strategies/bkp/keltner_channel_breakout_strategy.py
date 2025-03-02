import ast

import pandas as pd

from src.strategies.base_strategy import BaseStrategy


class KeltnerChannelBreakoutStrategy(BaseStrategy):
    def __init__(self, df: pd.DataFrame, top_n):
        super().__init__(df, top_n)

    def run(self):
        keltner_breakout_tickers = []

        for idx, row in self.df.iterrows():
            keltner_channel = row['KeltnerChannel']
            if isinstance(keltner_channel, str):
                keltner_channel = keltner_channel.replace('np.float64(', '').replace(')', '')
                try:
                    keltner_channel = ast.literal_eval(keltner_channel)
                except (ValueError, SyntaxError) as e:
                    print(f"Error evaluating Keltner Channel for ticker {row['Ticker']}: {e}")
                    keltner_channel = {}

            if isinstance(keltner_channel, tuple) and len(keltner_channel) == 2:
                upper_band = keltner_channel[0]
                current_price = row['ExponentialMovingAverage']

                if current_price > upper_band:
                    keltner_breakout_tickers.append({'ticker': row['Ticker'], 'price': current_price})

        sorted_breakouts = sorted(keltner_breakout_tickers, key=lambda x: x['price'], reverse=True)[:self.top_n]
        return sorted_breakouts

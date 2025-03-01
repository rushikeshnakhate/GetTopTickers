import pandas as pd
import ast
import numpy as np

from src.strategies.base_strategy import BaseStrategy


class TopGainersStrategy(BaseStrategy):
    def __init__(self, df: pd.DataFrame, top_n):
        super().__init__(df, top_n)

    def run(self):
        # Example: Selecting top gainers based on Bollinger Bands' Upper Band
        price_changes = []

        for idx, row in self.df.iterrows():
            # Handle the case where BollingerBands could be a string or dictionary
            bollinger_bands = row['BollingerBands']
            if isinstance(bollinger_bands, str):
                # Replace np.float64(<value>) with just <value>
                bollinger_bands = bollinger_bands.replace('np.float64(', '').replace(')', '')

                try:
                    bollinger_bands = ast.literal_eval(bollinger_bands)
                except (ValueError, SyntaxError) as e:
                    print(f"Error evaluating BollingerBands for ticker {row['Ticker']}: {e}")
                    bollinger_bands = {}

            if isinstance(bollinger_bands, dict) and 'Upper Band' in bollinger_bands:
                upper_band = bollinger_bands['Upper Band']
            else:
                upper_band = None  # Default if no valid BollingerBands data is available

            # Example: Use Exponential Moving Average (EMA) as the current price
            current_price = row['ExponentialMovingAverage']

            if upper_band is not None and upper_band > 0:
                gain_percentage = ((current_price - upper_band) / upper_band) * 100
            else:
                gain_percentage = 0  # No gain if the upper band is invalid or non-positive

            price_changes.append({'ticker': row['Ticker'], 'gain': gain_percentage})

        # Sorting and selecting top N gainers based on gain percentage
        sorted_gainers = sorted(price_changes, key=lambda x: x['gain'], reverse=True)[:self.top_n]
        return sorted_gainers

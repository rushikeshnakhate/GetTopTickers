import pandas as pd
import ast
import numpy as np

from src.strategies.base_strategy import BaseStrategy


class TopLosersStrategy(BaseStrategy):
    def __init__(self, df: pd.DataFrame, top_n=5):
        super().__init__(df, top_n)

    def run(self):
        # Identifying top losers based on Bollinger Bands' Lower Band
        price_changes = []

        for idx, row in self.df.iterrows():
            # Handle cases where BollingerBands is a string or dictionary
            bollinger_bands = row['BollingerBands']
            if isinstance(bollinger_bands, str):
                # Replace np.float64(<value>) with just <value>
                bollinger_bands = bollinger_bands.replace('np.float64(', '').replace(')', '')

                try:
                    bollinger_bands = ast.literal_eval(bollinger_bands)
                except (ValueError, SyntaxError) as e:
                    print(f"Error evaluating BollingerBands for ticker {row['Ticker']}: {e}")
                    bollinger_bands = {}

            if isinstance(bollinger_bands, dict) and 'Lower Band' in bollinger_bands:
                lower_band = bollinger_bands['Lower Band']
            else:
                lower_band = None  # Default if no valid BollingerBands data is available

            # Example: Use Exponential Moving Average (EMA) as the current price
            current_price = row['ExponentialMovingAverage']

            if lower_band is not None and lower_band > 0:
                loss_percentage = ((current_price - lower_band) / lower_band) * 100
            else:
                loss_percentage = 0  # No loss if the lower band is invalid or non-positive

            price_changes.append({'ticker': row['Ticker'], 'loss': loss_percentage})

        # Sorting and selecting top N losers based on loss percentage
        sorted_losers = sorted(price_changes, key=lambda x: x['loss'])[:self.top_n]
        return sorted_losers

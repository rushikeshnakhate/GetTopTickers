import pandas as pd
import ast
import numpy as np

from src.strategies.base_strategy import BaseStrategy


class BollingerBandsBreakoutStrategy(BaseStrategy):
    def __init__(self, df: pd.DataFrame, top_n=5):
        super().__init__(df, top_n)

    def run(self):
        breakout_tickers = []

        for idx, row in self.df.iterrows():
            # Extract Bollinger Bands data
            bollinger_bands = row['BollingerBands']
            if isinstance(bollinger_bands, str):
                # Clean the string representation of the dictionary
                bollinger_bands = bollinger_bands.replace('np.float64(', '').replace(')', '')
                try:
                    bollinger_bands = ast.literal_eval(bollinger_bands)  # Convert string to dictionary
                except (ValueError, SyntaxError) as e:
                    print(f"Error evaluating Bollinger Bands for ticker {row['Ticker']}: {e}")
                    bollinger_bands = {}  # Default to empty dictionary if parsing fails

            # Extract upper band value
            if isinstance(bollinger_bands, dict) and 'Upper Band' in bollinger_bands:
                upper_band = bollinger_bands['Upper Band']
            else:
                upper_band = None  # Skip if no valid upper band is found

            # Use Exponential Moving Average (EMA) as the current price
            current_price = row['ExponentialMovingAverage']

            # Check if the current price is above the upper band
            if upper_band is not None and current_price > upper_band:
                # Calculate the percentage gain above the upper band
                gain_percentage = ((current_price - upper_band) / upper_band) * 100
                breakout_tickers.append({'ticker': row['Ticker'], 'gain_percentage': gain_percentage})

        # Sort the tickers by gain percentage in descending order and select the top `n`
        sorted_breakouts = sorted(breakout_tickers, key=lambda x: x['gain_percentage'], reverse=True)[:self.top_n]
        return sorted_breakouts

import pandas as pd
import ast

from src.strategies.base_strategy import BaseStrategy


class TopGainersCombinedStrategy(BaseStrategy):
    def __init__(self, df: pd.DataFrame, top_n):
        super().__init__(df, top_n)

    def run(self):
        price_changes = []

        for idx, row in self.df.iterrows():
            # Bollinger Bands-based gain calculation
            bollinger_bands = row['BollingerBands']
            if isinstance(bollinger_bands, str):
                bollinger_bands = bollinger_bands.replace('np.float64(', '').replace(')', '')
                try:
                    bollinger_bands = ast.literal_eval(bollinger_bands)
                except (ValueError, SyntaxError):
                    bollinger_bands = {}

            if isinstance(bollinger_bands, dict) and 'Upper Band' in bollinger_bands:
                upper_band = bollinger_bands['Upper Band']
            else:
                upper_band = None  # No valid Bollinger Bands data

            current_price = row['ExponentialMovingAverage']
            bollinger_band_gain = ((current_price - upper_band) / upper_band * 100) if upper_band else 0

            # RSI-based gain calculation (directly using RSI as gain)
            rsi_gain = row['RelativeStrengthIndex']

            # Combining both RSI and Bollinger Band-based gains
            combined_gain = bollinger_band_gain + rsi_gain  # You could weight them as needed

            price_changes.append({
                'ticker': row['Ticker'],
                'bollinger_gain': bollinger_band_gain,
                'rsi_gain': rsi_gain,
                'combined_gain': combined_gain
            })

        # Sorting and selecting top N based on the combined gain
        sorted_gainers = sorted(price_changes, key=lambda x: x['combined_gain'], reverse=True)[:self.top_n]
        return sorted_gainers

import logging

import numpy as np
import pandas as pd
from src.indicators.base_indicator import BaseIndicator


class DonchianChannel(BaseIndicator):
    """Calculates the Donchian Channel."""

    def __init__(self, period=14):
        super().__init__(period)

    def calculate(self, data: pd.DataFrame):
        if data is None or data.empty:
            logging.warn("DonchianChannel no data for stock.")
            return None

        # Calculate upper and lower bands of the Donchian Channel
        upper_band = data['High'].rolling(window=self.period).max()
        lower_band = data['Low'].rolling(window=self.period).min()

        # Return the values as a dictionary
        return {
            'Upper Band': upper_band.iloc[-1],  # Get the last value in the rolling window
            'Lower Band': lower_band.iloc[-1]  # Get the last value in the rolling window
        }

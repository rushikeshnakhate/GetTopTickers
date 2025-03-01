import logging

import numpy as np
import pandas as pd
from src.indicators.base_indicator import BaseIndicator


class Aroon(BaseIndicator):
    """Calculates the Aroon Indicator."""

    def __init__(self, period=14):
        super().__init__(period)

    def calculate(self, data: pd.DataFrame):
        if data is None or data.empty:
            logging.warn("Aroon no data for stock.")
            return None
            # raise ValueError("Data cannot be empty.")

        # Calculate Aroon Up and Aroon Down
        aroon_up = ((data['Close'].rolling(window=self.period).apply(lambda x: (x.argmax())) + 1) / self.period) * 100
        aroon_down = ((data['Close'].rolling(window=self.period).apply(lambda x: (x.argmin())) + 1) / self.period) * 100

        # Return as a dictionary
        return {
            'Aroon Up': aroon_up.iloc[-1],  # Get the last value in the rolling window
            'Aroon Down': aroon_down.iloc[-1]  # Get the last value in the rolling window
        }

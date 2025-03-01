import logging

import numpy as np
import pandas as pd
from src.indicators.base_indicator import BaseIndicator


class PriceRateOfChange(BaseIndicator):
    """Calculates the Rate of Change (ROC)."""

    def __init__(self, period=14):
        super().__init__(period)

    def calculate(self, data: pd.Series):
        if data is None or data.empty:
            logging.warn("PriceRateOfChange no data for stock.")
            # raise ValueError("Data cannot be empty.")
            return None
        roc = ((data - data.shift(self.period)) / data.shift(self.period)) * 100

        return roc.iloc[-1]

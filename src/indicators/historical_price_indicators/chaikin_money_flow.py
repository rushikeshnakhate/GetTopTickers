import logging

import pandas as pd

from src.indicators.base_indicator import BaseIndicator


class ChaikinMoneyFlow(BaseIndicator):
    """Calculates the Chaikin Money Flow (CMF)."""

    def __init__(self, period=20):
        super().__init__(period)

    def calculate(self, data: pd.DataFrame):
        if data is None or data.empty:
            # raise ValueError("Data cannot be empty.")
            logging.warn("ChaikinMoneyFlow no data for stock.")
            return None

        money_flow = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
        cmf = (money_flow * data['Volume']).rolling(window=self.period).sum() / data['Volume'].rolling(
            window=self.period).sum()

        return cmf.iloc[-1]

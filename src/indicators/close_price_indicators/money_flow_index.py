import logging

import numpy as np
import pandas as pd
from src.indicators.base_indicator import BaseIndicator


class MoneyFlowIndex(BaseIndicator):
    """Calculates the Money Flow Index (MFI)."""

    def __init__(self, period=14):
        super().__init__(period)

    def calculate(self, data: pd.DataFrame):
        if data is None or data.empty:
            logging.warn("ForceIndex no data for stock.")
            return None

        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        money_flow = typical_price * data['Volume']
        money_flow_pos = money_flow.where(data['Close'] > data['Close'].shift(), 0)
        money_flow_neg = money_flow.where(data['Close'] < data['Close'].shift(), 0)

        mf_ratio = money_flow_pos.rolling(window=self.period).sum() / money_flow_neg.rolling(window=self.period).sum()

        mfi = 100 - (100 / (1 + mf_ratio))

        return mfi.iloc[-1]

# import numpy as np
# import pandas as pd
# from src.indicators.base_indicator import BaseIndicator
#
#
# class StochasticOscillator(BaseIndicator):
#     """Calculates the Stochastic Oscillator."""
#
#     def __init__(self, period=14, k_period=3, d_period=3):
#         super().__init__(period)
#         self.k_period = k_period
#         self.d_period = d_period
#
#     def calculate(self, data: pd.DataFrame):
#         if data is None or data.empty:
#             raise ValueError("Data cannot be empty.")
#
#         lowest_low = data['Low'].rolling(window=self.period).min()
#         highest_high = data['High'].rolling(window=self.period).max()
#
#         % K = 100 * ((data['Close'] - lowest_low) / (highest_high - lowest_low))
#         % D = %K.rolling(window=self.d_period).mean()
#
#         return %K.iloc[-1], %D.iloc[-1]

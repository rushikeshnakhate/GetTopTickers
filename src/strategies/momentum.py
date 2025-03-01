# src/strategies/momentum.py
from .base_strategy import BaseStrategy
from src.indicators.close_price_indicators.moving_average import MovingAverageIndicator


class MomentumStrategy(BaseStrategy):
    def generate_signals(self):
        ma = MovingAverageIndicator(self.stock_data, window=50).calculate()
        return self.stock_data > ma

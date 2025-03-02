from typing import Dict, Type

import pandas as pd

from src.performance_matrix.base_metrics_group import BaseMetricsGroup
from src.performance_matrix.trade_metrics.gain_to_pain_ratio import GainToPainRatio
from src.performance_matrix.trade_metrics.loss_rate import LossRate
from src.performance_matrix.trade_metrics.profit_factor import ProfitFactor
from src.performance_matrix.trade_metrics.ulcer_index import UlcerIndex
from src.performance_matrix.trade_metrics.win_rate import WinRate


class TradeMetricsGroup(BaseMetricsGroup):
    """Handles trade-related performance metrics."""

    def __init__(self, stock_data: pd.DataFrame, market_data: pd.DataFrame = None, risk_free_rate: float = 0.0):
        metrics = {
            'WinRate': WinRate,
            'LossRate': LossRate,
            'ProfitFactor': ProfitFactor,
            'GainToPainRatio': GainToPainRatio,
            'UlcerIndex': UlcerIndex,
        }
        super().__init__(stock_data=stock_data, market_data=market_data, risk_free_rate=risk_free_rate, metrics=metrics)

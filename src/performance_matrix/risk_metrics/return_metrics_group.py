import pandas as pd

from src.performance_matrix.base_metrics_group import BaseMetricsGroup
from src.performance_matrix.risk_metrics.conditional_value_at_risk import ConditionalValueAtRisk
from src.performance_matrix.risk_metrics.maximum_drawdown import MaximumDrawdown
from src.performance_matrix.risk_metrics.risk_of_ruin import RiskOfRuin
from src.performance_matrix.risk_metrics.value_at_risk import ValueAtRisk
from src.performance_matrix.risk_metrics.volatility import Volatility


class RiskMetricsGroup(BaseMetricsGroup):
    """Handles return-related performance metrics."""

    def __init__(self, stock_data: pd.DataFrame, market_data: pd.DataFrame = None, risk_free_rate: float = 0.0):
        metrics = {
            'Volatility': Volatility,
            'MaximumDrawdown': MaximumDrawdown,
            'ValueAtRisk': ValueAtRisk,
            'ConditionalValueAtRisk': ConditionalValueAtRisk,
            'RiskOfRuin': RiskOfRuin,
        }
        super().__init__(stock_data=stock_data, market_data=market_data, risk_free_rate=risk_free_rate, metrics=metrics)

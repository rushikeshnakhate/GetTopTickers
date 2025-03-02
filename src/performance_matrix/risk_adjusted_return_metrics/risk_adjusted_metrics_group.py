import pandas as pd
from src.performance_matrix.risk_adjusted_return_metrics.sharpe_ratio import SharpeRatio
from src.performance_matrix.risk_adjusted_return_metrics.sortino_ratio import SortinoRatio
from src.performance_matrix.risk_adjusted_return_metrics.calmar_ratio import CalmarRatio
from src.performance_matrix.risk_adjusted_return_metrics.omega_ratio import OmegaRatio
from src.performance_matrix.risk_adjusted_return_metrics.sterling_ratio import SterlingRatio
from src.performance_matrix.risk_adjusted_return_metrics.treynor_ratio import TreynorRatio
from src.performance_matrix.base_metrics_group import BaseMetricsGroup


class RiskAdjustedMetricsGroup(BaseMetricsGroup):
    """Handles risk-adjusted return performance metrics."""

    def __init__(self, stock_data: pd.DataFrame, market_data: pd.DataFrame = None, risk_free_rate: float = 0.0):
        metrics = {
            'SharpeRatio': SharpeRatio,
            'SortinoRatio': SortinoRatio,
            'CalmarRatio': CalmarRatio,
            'OmegaRatio': OmegaRatio,
            'SterlingRatio': SterlingRatio,
            'TreynorRatio': TreynorRatio,
        }
        super().__init__(stock_data=stock_data, market_data=market_data, risk_free_rate=risk_free_rate, metrics=metrics)

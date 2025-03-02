import pandas as pd
from src.performance_matrix.benchmark_relative_metrics.beta import Beta
from src.performance_matrix.benchmark_relative_metrics.alpha import Alpha
from src.performance_matrix.benchmark_relative_metrics.rsquared import RSquared
from src.performance_matrix.benchmark_relative_metrics.information_ratio import InformationRatio
from src.performance_matrix.benchmark_relative_metrics.active_return import ActiveReturn
from src.performance_matrix.benchmark_relative_metrics.tracking_rrror import TrackingError
from src.performance_matrix.benchmark_relative_metrics.upside_capture_ratio import UpsideCaptureRatio
from src.performance_matrix.benchmark_relative_metrics.downside_capture_ratio import DownsideCaptureRatio
from src.performance_matrix.base_metrics_group import BaseMetricsGroup


class BenchmarkRelativeMetricsGroup(BaseMetricsGroup):
    """Handles benchmark-relative performance metrics."""

    def __init__(self, stock_data: pd.DataFrame, market_data: pd.DataFrame = None, risk_free_rate: float = 0.0):
        metrics = {
            'Beta': Beta,
            'Alpha': Alpha,
            'RSquared': RSquared,
            'InformationRatio': InformationRatio,
            'TrackingError': TrackingError,
            'ActiveReturn': ActiveReturn,
            'UpsideCaptureRatio': UpsideCaptureRatio,
            'DownsideCaptureRatio': DownsideCaptureRatio,
        }
        super().__init__(stock_data=stock_data, market_data=market_data, risk_free_rate=risk_free_rate, metrics=metrics)

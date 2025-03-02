import logging
from typing import List, Dict

import pandas as pd

from src.strategies.indicator_based.mean_reversion_indicators.bollinger_bands_mean_reversion_strategy import \
    BollingerBandsMeanReversionStrategy
from src.strategies.indicator_based.mean_reversion_indicators.keltner_channels_mean_reversion_strategy import \
    KeltnerChannelsMeanReversionStrategy
from src.strategies.indicator_based.momentum_indicators.CCI_momentum_strategy import CCIMomentumStrategy
from src.strategies.indicator_based.momentum_indicators.RSI_momentum_strategy import RSIMomentumStrategy
from src.strategies.indicator_based.momentum_indicators.price_rate_of_change_momentum_strategy import \
    PriceRateOfChangeMomentumStrategy
from src.strategies.indicator_based.momentum_indicators.williamsr_momentum_strategy import WilliamsRMomentumStrategy
from src.strategies.indicator_based.trend_indicators.MACD_crossover_strategy import MACDCrossoverStrategy
from src.strategies.indicator_based.trend_indicators.aroon_trend_strategy import AroonTrendStrategy
from src.strategies.indicator_based.trend_indicators.exponential_moving_average_trend_strategy import \
    ExponentialMovingAverageTrendStrategy
from src.strategies.indicator_based.trend_indicators.moving_average_trend_strategy import MovingAverageTrendStrategy
from src.strategies.indicator_based.volatility_indicators.bollinger_bands_volatility_strategy import \
    BollingerBandsVolatilityStrategy
from src.strategies.indicator_based.volatility_indicators.chaikin_money_flow_strategy import ChaikinMoneyFlowStrategy
from src.strategies.indicator_based.volatility_indicators.keltner_channels_volatility_strategy import \
    KeltnerChannelsVolatilityStrategy
from src.strategies.indicator_based.volume_indicators.donchian_channel_mean_reversion_strategy import \
    DonchianChannelMeanReversionStrategy
from src.strategies.indicator_based.volume_indicators.on_balance_volume_strategy import OnBalanceVolumeStrategy
from src.strategies.performance_matrix_based.benchmark_relative_metrics.calmar_ratio_strategy import CalmarRatioStrategy
from src.strategies.performance_matrix_based.benchmark_relative_metrics.gain_to_pain_ratio_strategy import \
    GainToPainRatioStrategy
from src.strategies.performance_matrix_based.benchmark_relative_metrics.sharpe_ratio_strategy import SharpeRatioStrategy
from src.strategies.performance_matrix_based.benchmark_relative_metrics.sortino_ratio_strategy import \
    SortinoRatioStrategy
from src.strategies.performance_matrix_based.distribution_metrics.loss_rate_strategy import LossRateStrategy
from src.strategies.performance_matrix_based.distribution_metrics.profit_factor_strategy import ProfitFactorStrategy
from src.strategies.performance_matrix_based.distribution_metrics.win_rate_strategy import WinRateStrategy
from src.strategies.performance_matrix_based.return_matrix.annualized_return_strategy import AnnualizedReturnStrategy
from src.strategies.performance_matrix_based.return_matrix.average_daily_return_strategy import \
    AverageDailyReturnStrategy
from src.strategies.performance_matrix_based.return_matrix.cumulative_return_strategy import CumulativeReturnStrategy
from src.strategies.performance_matrix_based.risk_metrics.conditional_value_at_risk_strategy import \
    ConditionalValueAtRiskStrategy
from src.strategies.performance_matrix_based.risk_metrics.maximum_drawdown_strategy import MaximumDrawdownStrategy
from src.strategies.performance_matrix_based.risk_metrics.ulcerIndex_strategy import UlcerIndexStrategy
from src.strategies.performance_matrix_based.risk_metrics.value_at_risk_strategy import ValueAtRiskStrategy
from src.strategies.performance_matrix_based.risk_metrics.volatility_strategy import VolatilityStrategy

# Static mapping of strategies to their required data type
STRATEGY_DATA_MAPPING = {
    # Indicator-based strategies
    "AroonTrendStrategy": "indicator",
    "ExponentialMovingAverageTrendStrategy": "indicator",
    "MACDCrossoverStrategy": "indicator",
    "MovingAverageTrendStrategy": "indicator",
    "BollingerBandsVolatilityStrategy": "indicator",
    "ChaikinMoneyFlowStrategy": "indicator",
    "KeltnerChannelsVolatilityStrategy": "indicator",
    "DonchianChannelMeanReversionStrategy": "indicator",
    "OnBalanceVolumeStrategy": "indicator",
    "CCIMomentumStrategy": "indicator",
    "PriceRateOfChangeMomentumStrategy": "indicator",
    "RSIMomentumStrategy": "indicator",
    "WilliamsRMomentumStrategy": "indicator",
    "BollingerBandsMeanReversionStrategy": "indicator",
    "KeltnerChannelsMeanReversionStrategy": "indicator",

    # Performance-based strategies
    "CalmarRatioStrategy": "performance",
    "GainToPainRatioStrategy": "performance",
    "SharpeRatioStrategy": "performance",
    "SortinoRatioStrategy": "performance",
    "LossRateStrategy": "performance",
    "ProfitFactorStrategy": "performance",
    "WinRateStrategy": "performance",
    "AnnualizedReturnStrategy": "performance",
    "AverageDailyReturnStrategy": "performance",
    "CumulativeReturnStrategy": "performance",
    "ConditionalValueAtRiskStrategy": "performance",
    "MaximumDrawdownStrategy": "performance",
    "UlcerIndexStrategy": "performance",
    "ValueAtRiskStrategy": "performance",
    "VolatilityStrategy": "performance",
}


class StrategyFactory:
    def __init__(self, indicators_df: pd.DataFrame, performance_df: pd.DataFrame, top_n: int = 15):
        """
        Initialize with indicator and performance data.
        """
        self.indicators_df = indicators_df
        self.performance_df = performance_df
        self.top_n = top_n

        # Store strategies in a dictionary
        self.strategies = {
            # Indicator-based strategies
            "AroonTrendStrategy": AroonTrendStrategy(indicators_df, top_n),
            "ExponentialMovingAverageTrendStrategy": ExponentialMovingAverageTrendStrategy(indicators_df, top_n),
            "MACDCrossoverStrategy": MACDCrossoverStrategy(indicators_df, top_n),
            "MovingAverageTrendStrategy": MovingAverageTrendStrategy(indicators_df, top_n),
            "BollingerBandsVolatilityStrategy": BollingerBandsVolatilityStrategy(indicators_df, top_n),
            "ChaikinMoneyFlowStrategy": ChaikinMoneyFlowStrategy(indicators_df, top_n),
            "KeltnerChannelsVolatilityStrategy": KeltnerChannelsVolatilityStrategy(indicators_df, top_n),
            "DonchianChannelMeanReversionStrategy": DonchianChannelMeanReversionStrategy(indicators_df, top_n),
            "OnBalanceVolumeStrategy": OnBalanceVolumeStrategy(indicators_df, top_n),
            "CCIMomentumStrategy": CCIMomentumStrategy(indicators_df, top_n),
            "PriceRateOfChangeMomentumStrategy": PriceRateOfChangeMomentumStrategy(indicators_df, top_n),
            "RSIMomentumStrategy": RSIMomentumStrategy(indicators_df, top_n),
            "WilliamsRMomentumStrategy": WilliamsRMomentumStrategy(indicators_df, top_n),
            "BollingerBandsMeanReversionStrategy": BollingerBandsMeanReversionStrategy(indicators_df, top_n),
            "KeltnerChannelsMeanReversionStrategy": KeltnerChannelsMeanReversionStrategy(indicators_df, top_n),

            # Performance-based strategies
            "CalmarRatioStrategy": CalmarRatioStrategy(performance_df, top_n),
            "GainToPainRatioStrategy": GainToPainRatioStrategy(performance_df, top_n),
            "SharpeRatioStrategy": SharpeRatioStrategy(performance_df, top_n),
            "SortinoRatioStrategy": SortinoRatioStrategy(performance_df, top_n),
            "LossRateStrategy": LossRateStrategy(performance_df, top_n),
            "ProfitFactorStrategy": ProfitFactorStrategy(performance_df, top_n),
            "WinRateStrategy": WinRateStrategy(performance_df, top_n),
            "AnnualizedReturnStrategy": AnnualizedReturnStrategy(performance_df, top_n),
            "AverageDailyReturnStrategy": AverageDailyReturnStrategy(performance_df, top_n),
            "CumulativeReturnStrategy": CumulativeReturnStrategy(performance_df, top_n),
            "ConditionalValueAtRiskStrategy": ConditionalValueAtRiskStrategy(performance_df, top_n),
            "MaximumDrawdownStrategy": MaximumDrawdownStrategy(performance_df, top_n),
            "UlcerIndexStrategy": UlcerIndexStrategy(performance_df, top_n),
            "ValueAtRiskStrategy": ValueAtRiskStrategy(performance_df, top_n),
            "VolatilityStrategy": VolatilityStrategy(performance_df, top_n),
        }

    def run_strategy(self, strategy_name: str) -> List[str]:
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy '{strategy_name}' not found.")

        strategy = self.strategies[strategy_name]
        sorted_df = strategy.run()

        # Log the type and contents of sorted_df
        logging.info(f"Strategy: {strategy_name}")
        logging.info(f"Type of sorted_df: {type(sorted_df)}")
        logging.info(f"Contents of sorted_df: {sorted_df}")

        return strategy.get_tickers(sorted_df)

    def run_all_strategies(self) -> Dict[str, List[str]]:
        results = {}
        for name in self.strategies:
            try:
                results[name] = self.run_strategy(name)
            except Exception as ex:
                logging.error(str(ex))
        return results

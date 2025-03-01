import logging
from typing import Optional, List

import pandas as pd
from tabulate import tabulate

from src.cache.cache_factory import CacheFactory
from src.indicators.close_price_indicators.bollinger_bands import BollingerBands
from src.indicators.close_price_indicators.exponential_moving_average import ExponentialMovingAverage
from src.indicators.close_price_indicators.force_index import ForceIndex
from src.indicators.close_price_indicators.money_flow_index import MoneyFlowIndex
from src.indicators.close_price_indicators.moving_average import MovingAverageIndicator
from src.indicators.close_price_indicators.price_rate_of_change import PriceRateOfChange
from src.indicators.close_price_indicators.relative_strength_index import RelativeStrengthIndex
from src.indicators.close_price_indicators.vortex_indicator import VortexIndicator
from src.indicators.close_price_indicators.williams_percentage_R import WilliamsR
from src.indicators.historical_price_indicators.accumulation_distribution_line import ADLine
from src.indicators.historical_price_indicators.aroon import Aroon
from src.indicators.historical_price_indicators.chaikin_money_flow import ChaikinMoneyFlow
from src.indicators.historical_price_indicators.commodity_channel_index import CommodityChannelIndex
from src.indicators.historical_price_indicators.donchian_channel import DonchianChannel
from src.indicators.historical_price_indicators.ease_of_movement import EaseOfMovement
from src.indicators.historical_price_indicators.keltner_channel import KeltnerChannel
from src.indicators.historical_price_indicators.moving_average_convergence_divergence import \
    MovingAverageConvergenceDivergence
from src.indicators.historical_price_indicators.on_balance_volume import OnBalanceVolume
from src.service.main import dataFetcher
from src.utils.constants import CacheType


class IndicatorFactory:
    """Factory class to calculate multiple indicators."""

    # Group indicators based on close price or historical prices (Open, High, Low, Volume)
    CLOSE_PRICE_INDICATORS = {
        'BollingerBands': BollingerBands,
        'ExponentialMovingAverage': ExponentialMovingAverage,
        'MovingAverage': MovingAverageIndicator,
        'RelativeStrengthIndex': RelativeStrengthIndex,
        'PriceRateOfChange': PriceRateOfChange,
    }

    HISTORICAL_PRICE_INDICATORS = {
        'AccumulationDistributionLine': ADLine,
        'Aroon': Aroon,
        'ChaikinMoneyFlow': ChaikinMoneyFlow,
        'CommodityChannelIndex': CommodityChannelIndex,
        'DonchianChannel': DonchianChannel,
        'EaseOfMovement': EaseOfMovement,
        'ForceIndex': ForceIndex,
        'KeltnerChannel': KeltnerChannel,
        'MovingAverageConvergenceDivergence': MovingAverageConvergenceDivergence,
        'MoneyFlowIndex': MoneyFlowIndex,
        'OnBalanceVolume': OnBalanceVolume,
        'VortexIndicator': VortexIndicator,
        'WilliamsR': WilliamsR
    }

    def __init__(self, stock_data: pd.DataFrame, period: int = 14):
        """Initialize with period and stock data."""
        self.period = period
        self.stock_data = stock_data
        self.cache = CacheFactory.get_cache(CacheType.PANDAS)

    def calculate_close_price_indicators(self, ticker: str, selected_indicators: Optional[List[str]] = None) -> dict:
        """
        Calculate indicators that use only the close price.
        """
        close_price_results = {}
        close_price_indicators_group = IndicatorFactory.CLOSE_PRICE_INDICATORS

        for indicator_name in close_price_indicators_group:
            if selected_indicators is None or indicator_name in selected_indicators:
                indicator_class = close_price_indicators_group[indicator_name]
                indicator = indicator_class(self.period)
                logging.info("calculating close price indicator={} for ticker={}".format(indicator_name, ticker))
                result = indicator.calculate(self.stock_data['Close'])
                close_price_results[indicator_name] = result

        return close_price_results

    def calculate_historical_price_indicators(self, ticker: str,
                                              selected_indicators: Optional[List[str]] = None) -> dict:
        """
        Calculate indicators that use historical price data (Open, High, Low, Volume).
        """
        historical_price_results = {}
        historical_price_indicators_group = IndicatorFactory.HISTORICAL_PRICE_INDICATORS

        for indicator_name in historical_price_indicators_group:
            if selected_indicators is None or indicator_name in selected_indicators:
                indicator_class = historical_price_indicators_group[indicator_name]
                indicator = indicator_class(self.period)
                logging.info("calculating historic indicator={} for ticker={}".format(indicator_name, ticker))
                result = indicator.calculate(self.stock_data)
                historical_price_results[indicator_name] = result

        return historical_price_results

    def calculate_all_indicators(self, ticker: str, start_date: str, end_date: str,
                                 selected_indicators: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate all indicators for both close price and historical price data and combine them with caching.
        """
        cache_key = f"indicator_{ticker}_{start_date}_{end_date}"
        cached_data = self.cache.get(cache_key)

        if cached_data is not None:
            logging.info(f"Returning cached indicators data for {ticker} ({start_date} to {end_date})")
            return cached_data

        # Calculate indicators
        close_price_results = self.calculate_close_price_indicators(ticker, selected_indicators)
        historical_price_results = self.calculate_historical_price_indicators(ticker, selected_indicators)

        all_results = {**close_price_results, **historical_price_results}
        # Step 1: Create a DataFrame with the Ticker column
        df = pd.DataFrame({'Ticker': [ticker]})
        df.set_index('Ticker', inplace=True)
        # Step 2: Add the dictionary data to the DataFrame
        # Convert the dictionary to a DataFrame and concatenate it with the existing DataFrame
        df = pd.concat([df, pd.DataFrame([all_results])], axis=1)
        # Store result in cache
        self.cache.set(cache_key, df)
        return df


def get_indicator(ticker: str, start_date: str, end_date: str):
    data = dataFetcher.get_close_price_service(ticker=ticker, start_date=start_date, end_date=end_date)
    factory = IndicatorFactory(data)
    # Calculate all indicators for the current symbol
    all_indicators_df = factory.calculate_all_indicators(ticker=ticker, start_date=start_date, end_date=end_date)
    return all_indicators_df

import logging
from typing import List, Optional

import pandas as pd

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
from src.utils.constants import CacheType, GLobalColumnName
from src.utils.utils import to_dataframe


class IndicatorFactory:
    """Factory class to calculate multiple indicators."""

    def __init__(self, period: int = 14):
        """Initialize with period and stock data."""
        self.cache = CacheFactory.get_cache(CacheType.PANDAS)
        self.period = period

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

    def _calculate_close_price_indicators(self,
                                          ticker_data_df: pd.DataFrame,
                                          ticker: str,
                                          selected_indicators: Optional[List[str]] = None) -> dict:
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
                result = indicator.calculate(ticker_data_df['Close'])
                close_price_results[indicator_name] = result

        return close_price_results

    def _calculate_historical_price_indicators(self,
                                               ticker_data_df: pd.DataFrame,
                                               ticker: str,
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
                result = indicator.calculate(ticker_data_df)
                historical_price_results[indicator_name] = result

        return historical_price_results

    def calculate_all_indicators(self,
                                 ticker_data_df: pd.DataFrame,
                                 ticker: str,
                                 start_date: str,
                                 end_date: str,
                                 selected_indicators: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate all indicators for both close price and historical price data and combine them with caching.
        """
        cache_key = f"indicator_{ticker}_{start_date}_{end_date}"
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            logging.info(f"Returning cached indicators data for {ticker} ({start_date} to {end_date})")
            return cached_data

        logging.info(f" calculating indicators data for {ticker} ({start_date} to {end_date})")
        # Calculate indicators
        close_price_results = self._calculate_close_price_indicators(ticker_data_df, ticker, selected_indicators)
        historical_price_results = self._calculate_historical_price_indicators(ticker_data_df, ticker,
                                                                               selected_indicators)

        all_results = {**close_price_results, **historical_price_results}
        return to_dataframe(column_name=GLobalColumnName.TICKER, column_value=ticker, results=all_results)


def get_indicator(ticker_data_df: pd.DataFrame,
                  ticker: str,
                  start_date: str,
                  end_date: str,
                  selected_indicators: Optional[List[str]] = None):
    all_indicators_df = IndicatorFactory().calculate_all_indicators(ticker_data_df=ticker_data_df,
                                                                    ticker=ticker,
                                                                    start_date=start_date,
                                                                    end_date=end_date,
                                                                    selected_indicators=selected_indicators)
    return all_indicators_df


def get_indicator_bulk(
        ticker_data_df: pd.DataFrame,
        ticker_list: Optional[List[str]],
        start_date=None,
        end_date=None,
        selected_indicators: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calculate indicators for multiple tickers using pre-fetched close price data.

    :param selected_indicators:
    :param ticker_data_df:
    :param ticker_list: Optional list of stock ticker symbols. If None, fetch all available stocks.
    :param start_date: Start date of the data.
    :param end_date: End date of the data.
    :return: DataFrame containing indicators for the provided tickers.
    """
    ticker_len = len(ticker_list)

    cache_key = "all_indicator_{ticker_len}_{start_date}_{end_date}".format(ticker_len=ticker_len,
                                                                            start_date=start_date,
                                                                            end_date=end_date)

    cache = CacheFactory.get_cache(CacheType.PANDAS)
    cached_results = cache.get(cache_key)
    if cached_results is not None:
        logging.info("Returning cached data to indicator for all stocks key={}".format(cache_key))
        return cached_results

    # Initialize the indicator factory and compute indicators for each ticker
    indicators_list = []
    for ticker in ticker_list:
        ticker_data = ticker_data_df[ticker_data_df[GLobalColumnName.TICKER] == ticker]
        if ticker_data.empty:
            logging.warning(f"get_indicator_bulk No data found for ticker={ticker}. Skipping.")
            continue

        indicators_df = get_indicator(ticker_data_df=ticker_data,
                                      ticker=ticker,
                                      start_date=start_date,
                                      end_date=end_date,
                                      selected_indicators=selected_indicators)
        indicators_list.append(indicators_df)

    # Concatenate results if available
    if indicators_list:
        df = pd.concat(indicators_list, ignore_index=True)
    else:
        logging.warning("No indicators were computed for the provided tickers.")
        df = pd.DataFrame()
    cache.set(cache_key, df)
    return df

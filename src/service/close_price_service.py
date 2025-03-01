import logging

import pandas as pd

from .base_service import BaseService
from ..cache.base_cache import BaseCache
from ..providers.provider_factory import ProviderFactory
from ..utils.constants import Providers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClosePriceService(BaseService):
    def __init__(self, provider: str = Providers.YAHOO, cache: BaseCache = None, **kwargs):
        """
        Initialize the service with a specific provider and cache.
        :param provider: Name of the provider (e.g., "yahoo", "jugad", "custom").
        :param db_session: SQLAlchemy database session for caching.
        :param kwargs: Additional arguments for the provider (e.g., file_path for CustomProvider).
        """
        self.provider = ProviderFactory.get_provider(provider, **kwargs)
        self.cache = cache

    @staticmethod
    def _get_cache_key(ticker: str, **kwargs) -> str:
        """
        Generate a unique cache key based on the ticker and additional performance_matrix.
        :param ticker: Stock ticker (e.g., "AAPL").
        :param kwargs: Additional arguments (e.g., start_date, end_date).
        :return: Cache key as a string.
        """
        start_date = kwargs.get("start_date")
        end_date = kwargs.get("end_date")
        return f"close_price_{ticker}_{start_date}_{end_date}"

    def _handle_cache(self, cache_key: str, fetch_data_func, **kwargs) -> pd.DataFrame:
        """
        Common logic for handling cache: check cache, fetch data if not cached, and update cache.
        :param cache_key: Unique cache key.
        :param fetch_data_func: Function to fetch data from the provider.
        :param kwargs: Additional arguments for the fetch function.
        :return: Pandas DataFrame containing the data.
        """
        # Check if data is cached
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            logging.info(f"Returning cached data for {cache_key}")
            return cached_data

        # Fetch data from the provider
        logging.info(f"Fetching data from provider for {cache_key}")
        data = fetch_data_func(**kwargs)

        # Cache the data
        self.cache.set(cache_key, data, ttl=86400)  # Cache for 1 day (86400 seconds)
        return data

    def get_data(self, ticker: str, **kwargs) -> pd.DataFrame:
        """
        Get close prices for a given ticker.
        :param ticker: Stock ticker (e.g., "AAPL").
        :param kwargs: Additional arguments (e.g., start_date, end_date).
        :return: Pandas DataFrame containing close prices.
        """
        cache_key = self._get_cache_key(ticker, **kwargs)
        return self._handle_cache(cache_key, self.provider.download_historical_data, ticker=ticker, **kwargs)

    async def get_data_async(self, ticker: str, **kwargs) -> pd.DataFrame:
        """
        Asynchronously get close prices for a given ticker.
        :param ticker: Stock ticker (e.g., "AAPL").
        :param kwargs: Additional arguments (e.g., start_date, end_date).
        :return: Pandas DataFrame containing close prices.
        """
        cache_key = self._get_cache_key(ticker, **kwargs)
        return await self._handle_cache(cache_key, self.provider.download_close_prices_async, ticker=ticker, **kwargs)

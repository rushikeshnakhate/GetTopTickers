import logging
import pandas as pd

from .base_service import BaseService
from ..cache.base_cache import BaseCache
from ..providers.provider_factory import ProviderFactory
from ..utils.constants import Providers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BalanceSheetService(BaseService):
    def __init__(self, provider: str = Providers.YAHOO_FOR_BALANCE_SHEET, cache: BaseCache = None, **kwargs):
        """
        Initialize the service with a specific provider and cache.
        """
        self.provider = ProviderFactory.get_provider(provider, **kwargs)
        self.cache = cache

    @staticmethod
    def _get_cache_key(ticker: str) -> str:
        """
        Generate a unique cache key based on the ticker.
        """
        return f"balance_sheet_{ticker}"

    def _handle_cache(self, cache_key: str, fetch_data_func, **kwargs) -> pd.DataFrame:
        """
        Common logic for handling cache: check cache, fetch data if not cached, and update cache.
        """
        # Check if data is cached
        cached_data = self.cache.get(cache_key) if self.cache else None
        if cached_data is not None:
            logger.info(f"Returning cached data for {cache_key}")
            return cached_data

        # Fetch data from the provider
        logger.info(f"Fetching data from provider for {cache_key}")
        data = fetch_data_func(**kwargs)

        # Cache the data
        if self.cache:
            logger.info(f"cache the data from provider for {cache_key}")
            self.cache.set(cache_key, data, ttl=86400)  # Cache for 1 day
        return data

    def get_data(self, ticker: str, **kwargs) -> pd.DataFrame:
        """
        Get balance sheet data for a given ticker.
        """
        cache_key = self._get_cache_key(ticker)
        return self._handle_cache(cache_key, self.provider.download_balance_sheet, ticker=ticker, **kwargs)

    async def get_data_async(self, ticker: str, **kwargs) -> pd.DataFrame:
        """
        Asynchronously get balance sheet data for a given ticker.
        """
        cache_key = self._get_cache_key(ticker)
        return await self._handle_cache(cache_key, self.provider.download_balance_sheet_async, ticker=ticker, **kwargs)

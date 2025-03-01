import pandas as pd
import typing
from .base_service import BaseService
from ..cache.base_cache import BaseCache
from ..providers.provider_factory import ProviderFactory
from ..utils.constants import Providers


class StockListService(BaseService):
    def __init__(self, provider: str = Providers.CUSTOM, cache: BaseCache = None, **kwargs):
        """
        Initialize the service with a specific provider.
        """
        self.provider = ProviderFactory.get_provider(provider, **kwargs)

    def get_data(self, ticker: typing.Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Get stock listing data.
        """
        return self.provider.download_historical_data(ticker, **kwargs)

    async def get_data_async(self, ticker: typing.Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Asynchronously get stock listing data.
        """
        return await self.provider.download_historical_data_async(ticker, **kwargs)

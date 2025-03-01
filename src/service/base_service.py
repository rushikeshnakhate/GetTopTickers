from abc import ABC, abstractmethod

import pandas as pd

from src.cache.base_cache import BaseCache


class BaseService(ABC):
    @abstractmethod
    def get_data(self, ticker: str, cache: BaseCache, **kwargs) -> pd.DataFrame:
        """
        Get data for a given ticker.
        :param cache:
        :param ticker: Stock ticker ticker.
        :param kwargs: Additional performance_matrix (e.g., start_date, end_date).
        :return: DataFrame containing the requested data.
        """
        pass

    @abstractmethod
    async def get_data_async(self, ticker: str, **kwargs) -> pd.DataFrame:
        """
        Asynchronously get data for a given ticker.
        :param ticker: Stock ticker ticker.
        :param kwargs: Additional performance_matrix (e.g., start_date, end_date).
        :return: DataFrame containing the requested data.
        """
        pass

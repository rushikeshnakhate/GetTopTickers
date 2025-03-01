from abc import ABC, abstractmethod
from typing import Dict, Optional, List

import pandas as pd


class BaseProvider(ABC):
    @abstractmethod
    def download_historical_data(self,
                                 ticker: str,
                                 start_date: Optional[str] = None,
                                 end_date: Optional[str] = None,
                                 interval: str = "1d",
                                 ) -> pd.DataFrame:
        """
        Download historical stock data for a given ticker.
        """
        pass

    @abstractmethod
    async def download_historical_data_async(self,
                                             ticker: str,
                                             start_date: Optional[str] = None,
                                             end_date: Optional[str] = None,
                                             interval: str = "1d",
                                             ) -> pd.DataFrame:
        """
        Asynchronously download historical stock data for a given ticker.
        """
        pass

    @abstractmethod
    def download_close_prices(self,
                              ticker: str,
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None,
                              interval: str = "1d",
                              ) -> pd.Series:
        """
        Download close prices for a given ticker.
        """
        pass

    @abstractmethod
    async def download_close_prices_async(self,
                                          ticker: str,
                                          start_date: Optional[str] = None,
                                          end_date: Optional[str] = None,
                                          interval: str = "1d",
                                          ) -> pd.Series:
        """
        Asynchronously download close prices for a given ticker.
        """
        pass

    @abstractmethod
    def download_multiple_symbols(self,
                                  symbols: List[str],
                                  start_date: Optional[str] = None,
                                  end_date: Optional[str] = None,
                                  interval: str = "1d",
                                  ) -> Dict[str, pd.DataFrame]:
        """
        Download historical data for multiple symbols.
        """
        pass

    @abstractmethod
    async def download_multiple_symbols_async(self,
                                              symbols: List[str],
                                              start_date: Optional[str] = None,
                                              end_date: Optional[str] = None,
                                              interval: str = "1d",
                                              ) -> Dict[str, pd.DataFrame]:
        """
        Asynchronously download historical data for multiple symbols.
        """
        pass

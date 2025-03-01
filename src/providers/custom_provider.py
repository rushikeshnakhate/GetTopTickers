import pandas as pd
from typing import Dict, Optional, List
import asyncio
from .base_provider import BaseProvider


class CustomProvider(BaseProvider):
    def __init__(self, file_path: str):
        """
        Initialize the CustomProvider with a static CSV file.
        :param file_path: Path to the CSV file.
        """
        self.file_path = file_path
        self.data = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        """
        Load data from the CSV file into a DataFrame.
        """
        return pd.read_csv(self.file_path)

    def download_historical_data(self,
                                 ticker: Optional[str] = None,
                                 start_date: Optional[str] = None,
                                 end_date: Optional[str] = None,
                                 interval: str = "1d",
                                 ) -> pd.DataFrame:
        """
        Download historical stock data from the static CSV file.
        :param ticker: Filter by ticker (optional).
        :param start_date: Filter by start date (optional).
        :param end_date: Filter by end date (optional).
        :param interval: Not used for static data.
        :return: Filtered DataFrame.
        """
        data = self.data

        # # Filter by ticker
        # if ticker:
        #     data = data[data["SYMBOL"] == ticker]
        #
        # # Filter by date range (if DATE OF LISTING column exists)
        # if "DATE OF LISTING" in data.columns and start_date and end_date:
        #     data = data[
        #         (data["DATE OF LISTING"] >= start_date) & (data["DATE OF LISTING"] <= end_date)
        #         ]

        return data

    async def download_historical_data_async(self,
                                             ticker: Optional[str] = None,
                                             start_date: Optional[str] = None,
                                             end_date: Optional[str] = None,
                                             interval: str = "1d",
                                             ) -> pd.DataFrame:
        """
        Asynchronously download historical stock data from the static CSV file.
        """
        # Simulate async behavior using asyncio.sleep (optional)
        await asyncio.sleep(0.1)  # Simulate async delay
        return self.download_historical_data(ticker, start_date, end_date, interval)

    def download_close_prices(
            self,
            ticker: Optional[str] = None,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            interval: str = "1d",
    ) -> pd.Series:
        """
        Download close prices from the static CSV file.
        Note: Since the CSV does not contain close prices, this method is not applicable.
        """
        raise NotImplementedError("Close prices not available in static data.")

    async def download_close_prices_async(self,
                                          ticker: Optional[str] = None,
                                          start_date: Optional[str] = None,
                                          end_date: Optional[str] = None,
                                          interval: str = "1d",
                                          ) -> pd.Series:
        """
        Asynchronously download close prices from the static CSV file.
        Note: Since the CSV does not contain close prices, this method is not applicable.
        """
        raise NotImplementedError("Close prices not available in static data.")

    def download_multiple_symbols(self,
                                  symbols: List[str],
                                  start_date: Optional[str] = None,
                                  end_date: Optional[str] = None,
                                  interval: str = "1d",
                                  ) -> Dict[str, pd.DataFrame]:
        """
        Download historical data for multiple symbols from the static CSV file.
        """
        result = {}
        for ticker in symbols:
            result[ticker] = self.download_historical_data(ticker, start_date, end_date, interval)
        return result

    async def download_multiple_symbols_async(self,
                                              symbols: List[str],
                                              start_date: Optional[str] = None,
                                              end_date: Optional[str] = None,
                                              interval: str = "1d",
                                              ) -> Dict[str, pd.DataFrame]:
        """
        Asynchronously download historical data for multiple symbols from the static CSV file.
        """
        # Simulate async behavior using asyncio.sleep (optional)
        await asyncio.sleep(0.1)  # Simulate async delay
        return self.download_multiple_symbols(symbols, start_date, end_date, interval)

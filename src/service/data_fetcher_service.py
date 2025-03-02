import logging
import os
from typing import List

import pandas as pd

from src.cache.cache_factory import CacheFactory
from src.service.balance_sheet_service import BalanceSheetService
from src.service.close_price_service import ClosePriceService
from src.service.stock_list_service import StockListService
from src.utils.constants import Providers, CacheType, GLobalColumnName, GlobalStockData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFetcherService:
    def __init__(self):
        self.cache = CacheFactory.get_cache(CacheType.PANDAS)
        csv_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../customData/EQUITY_L.csv"))
        self.stock_list_service = StockListService(provider=Providers.CUSTOM, cache=self.cache, file_path=csv_file_path)
        self.close_price_service = ClosePriceService(provider=Providers.YAHOO, cache=self.cache)
        self.historical_price_service = ClosePriceService(provider=Providers.YAHOO, cache=self.cache)
        self.balance_sheet_service = BalanceSheetService(provider=Providers.YAHOO_FOR_BALANCE_SHEET, cache=self.cache)

    def get_stock_list_service_data(self) -> pd.DataFrame:
        return self.stock_list_service.get_data()

    def get_balance_sheet_service(self, ticker: str) -> pd.DataFrame:
        logger.info("get_balance_sheet_service price for ticker={}".format(ticker))
        return self.balance_sheet_service.get_data(ticker)

    def get_close_price_service(self, ticker: str, start_date=None, end_date=None) -> pd.DataFrame:
        try:
            df = self.close_price_service.get_data(ticker=ticker, start_date=start_date, end_date=end_date)
            logger.info("closing price for ticker={},start_date={}, end_date={}".format(ticker, start_date, end_date))
            return df

        except Exception as e:
            logging.info(f" closing failed for {ticker}: {e}")
            return None

    def get_close_price_service_bulk(self, start_date, end_date, ticker_list: List[str] = None) -> pd.DataFrame:
        """
        Fetch close price data for multiple tickers.
        :param ticker_list: List of stock ticker symbols.
        :param start_date: Start date of the data (optional).
        :param end_date: End date of the data (optional).
        :return: Combined DataFrame containing close price data for all tickers.
        """
        ticker_len = len(ticker_list)
        cache_key = "{ticker_len}_{start_date}_{end_date}".format(ticker_len=ticker_len,
                                                                  start_date=start_date,
                                                                  end_date=end_date)
        cached_results = self.cache.get(cache_key)
        if cached_results is not None:
            logging.info("Returning cached data to close_price for all stocks key={}".format(cache_key))
            return cached_results

        logging.info("downloading close_price data for ticker_len={} stocks key={}".format(ticker_len, cache_key))
        all_data = []
        for index, ticker in enumerate(ticker_list, start=1):
            remaining = ticker_len - index  # Calculate remaining symbols
            # Fetch data for the current ticker
            logger.info(f"Downloading data for {index}/{ticker_len}: {ticker} | Remaining: {remaining}")
            df = self.get_close_price_service(ticker=ticker, start_date=start_date, end_date=end_date)

            # Check if data is valid
            if df is not None and not df.empty:
                # Add ticker, start date, and end date as columns
                df[GLobalColumnName.TICKER] = ticker
                df[GlobalStockData.START_DATE] = start_date
                df[GlobalStockData.END_DATE] = end_date
                all_data.append(df)

        # Combine all DataFrames into one
        if all_data:
            df = pd.concat(all_data, ignore_index=True)
        else:
            df = pd.DataFrame()  # Return an empty Data
        self.cache.set(cache_key, df)
        return df

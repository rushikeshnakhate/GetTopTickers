import logging
import os

import pandas as pd

from src.cache.cache_factory import CacheFactory
from src.service.balance_sheet_service import BalanceSheetService
from src.service.close_price_service import ClosePriceService
from src.service.stock_list_service import StockListService
from src.utils.constants import Providers, CacheType

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

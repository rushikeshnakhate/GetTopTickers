import logging

import pandas as pd

from src.database.sqlite_db_engine import SQLiteDbEngine
from src.providers.yahoo_finance_provider import YahooFinanceProvider
from src.services.balance_sheet_service import BalanceSheetService
from src.services.stock_list_service import StockListService
from src.services.stock_pricing_service import StockPricingService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFetcher:
    def __init__(self, db_engine: SQLiteDbEngine):
        # Initialize the pricing provider and services only once during app start
        self.pricing_provider = YahooFinanceProvider()
        logger.info("Pricing provider initialized.")

        # Initialize services
        self.stock_pricing_service = StockPricingService(db_engine, self.pricing_provider)
        self.balance_sheet_service = BalanceSheetService(db_engine, self.pricing_provider)
        self.stock_list_service = StockListService(db_engine, self.pricing_provider)
        logger.info("Services initialized.")

    def get_data(self, tickers: list, period: str):
        for ticker in tickers:
            ticker = ticker + ".NS"
            pricing_data = self.stock_pricing_service.fetch_pricing_data(ticker, period)
            balance_sheet_data = self.balance_sheet_service.fetch_balance_sheet(ticker, period)

        return pricing_data, balance_sheet_data

    def get_stock_list(self):
        stock_list_data = self.stock_list_service.fetch_stock_data(cache_key="EQUITY_L.csv",
                                                                   csv_path=r"D:\PypPortWorkflow\src\inputData\EQUITY_L.csv")
        stock_list_data_df = pd.DataFrame(stock_list_data)

        tickers = stock_list_data_df['symbol'].unique().tolist()
        return self.get_data(tickers, period="1y")

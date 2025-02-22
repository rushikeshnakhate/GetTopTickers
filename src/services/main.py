import logging
from src.database.sqlite_db_engine import SQLiteDbEngine
from src.providers.yahoo_finance_provider import YahooFinanceProvider
from src.services.balance_sheet_service import BalanceSheetService
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
        logger.info("Services initialized.")

    def get_data(self, ticker: str, period: str):
        # Fetch pricing and balance sheet data
        pricing_data = self.stock_pricing_service.fetch_pricing_data(ticker, period)
        balance_sheet_data = self.balance_sheet_service.fetch_balance_sheet(ticker, period)
        return pricing_data, balance_sheet_data

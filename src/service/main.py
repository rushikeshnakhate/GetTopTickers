import logging
from typing import List

from src.service.data_fetcher_service import DataFetcherService
from src.utils.constants import StockListsColumns, Stocks, GLobalColumnName

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_stocks(tickers: List = None):
    ticker_list = tickers
    if tickers is None:
        df = dataFetcher.get_stock_list_service_data()
        eq_tickers = df[df[StockListsColumns.SERIES] == Stocks.EQUITY][StockListsColumns.SYMBOL]
        # Append 'NS' to each ticker and convert to list
        ticker_list = [tickers + Stocks.NSE_EXTENSION for tickers in eq_tickers]

    total_tickers_count = len(ticker_list)
    logging.info(f"Total tickers to download: {total_tickers_count}")
    return ticker_list, total_tickers_count


dataFetcher = DataFetcherService()


def fetch_market_and_price_data(start_date, end_date, tickers):
    """Fetch market data and close prices for the given date range."""
    logger.info(f"Fetching data for start_date={start_date}, end_date={end_date}")

    market_data = dataFetcher.get_close_price_service(
        ticker=GLobalColumnName.ticker_nifty50, start_date=start_date, end_date=end_date
    )

    ticker_list, _ = get_stocks(tickers)

    close_price = dataFetcher.get_close_price_service_bulk(
        ticker_list=ticker_list, start_date=start_date, end_date=end_date
    )

    return market_data, ticker_list, close_price

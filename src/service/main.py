import logging

from src.service.data_fetcher_service import DataFetcherService
from src.utils.constants import StockListsColumns, Stocks


def get_stocks():
    df = dataFetcher.get_stock_list_service_data()
    eq_tickers = df[df[StockListsColumns.SERIES] == Stocks.EQUITY][StockListsColumns.SYMBOL]
    # Append 'NS' to each ticker and convert to list
    ticker_list = [tickers + Stocks.NSE_EXTENSION for tickers in eq_tickers]
    total_tickers_count = len(ticker_list)
    logging.info(f"Total tickers to download: {total_tickers_count}")
    return ticker_list, total_tickers_count


dataFetcher = DataFetcherService()

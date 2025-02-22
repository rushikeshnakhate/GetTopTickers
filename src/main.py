import logging

from src.database.main import create_db_engine
from src.services.main import DataFetcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    try:
        db_engine = create_db_engine()
        data_fetcher = DataFetcher(db_engine)
        tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        for ticker in tickers:
            pricing_data, balance_sheet_data = data_fetcher.get_data(ticker, "2Y")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()

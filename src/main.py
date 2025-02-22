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
        stock_list_data_df = data_fetcher.get_stock_list()
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()

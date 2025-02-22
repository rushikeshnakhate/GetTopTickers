import logging
import pandas as pd
from sqlalchemy.orm import Session
from src.database.repositories.stock_lists_repository import StockListsRepository
from src.services.base_service import BaseService

logger = logging.getLogger(__name__)


class StockListService(BaseService):
    def get_repository(self, db: Session):
        """Returns the repository for stock listings."""
        return StockListsRepository(db)

    def download_data(self, ticker: str, period: str):
        """Since data comes from CSV, no download logic is required."""
        return None

    def transform_data(self, key, data):
        """
        Transform the data (this could be adjusted depending on the CSV format).
        The data should already be in a form suitable for insertion into the database.
        """
        # For now, we'll assume the data is already transformed into a list of dictionaries
        return data

    def get_stocks_list(self, csv_path: str):
        """Load stock list from CSV into SQLite database and insert cache key."""
        # Read the CSV file into a DataFrame
        logger.info(f"Loading stock data from CSV: {csv_path}")
        df = pd.read_csv(csv_path)

        # Remove spaces at the beginning and end of column names
        df.columns = df.columns.str.strip()

        # Ensure that the CSV has the correct columns
        required_columns = ['SYMBOL',
                            'NAME OF COMPANY',
                            'SERIES',
                            'DATE OF LISTING',
                            'PAID UP VALUE',
                            'MARKET LOT',
                            'ISIN NUMBER',
                            'FACE VALUE']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Missing required column: {col}, available_columns={df.columns}")
                return []

        # Map the column names in the DataFrame to match the model attributes
        column_mapping = {
            'SYMBOL': 'symbol',
            'NAME OF COMPANY': 'company_name',
            'SERIES': 'series',
            'DATE OF LISTING': 'date_of_listing',
            'PAID UP VALUE': 'paid_up_value',
            'MARKET LOT': 'market_lot',
            'ISIN NUMBER': 'isin_number',
            'FACE VALUE': 'face_value'
        }

        # Rename the columns to match the model
        df = df.rename(columns=column_mapping)

        # Transform the DataFrame into a list of dictionaries (each record to be inserted)
        stock_list_data = df.to_dict(orient='records')

        # Insert into the database
        db = self.db_engine.get_session()
        repository = self.get_repository(db)
        for stock in stock_list_data:
            repository.add(**stock)

        db.commit()  # Commit the changes to the database
        db.close()
        logger.info("Stock data successfully loaded into the database.")

        # Return the transformed data for caching and further processing
        return stock_list_data

    def fetch_stock_data(self, cache_key: str, csv_path: str):
        """Fetch the stock data with caching and handle cache keys."""
        # Step 1: Check cache
        cached_data = self.cache.get(cache_key)
        if cached_data:
            logger.info(f"Data found in cache for cache_key={cache_key}")
            return cached_data

        # Step 2: If data is not found in cache, load it from the CSV and insert it into the database
        stock_data = self.get_stocks_list(csv_path)
        if stock_data:
            # Cache the data
            self.cache.set(cache_key, stock_data)
            logger.info(f"Data cached for cache_key={cache_key}")

        return stock_data

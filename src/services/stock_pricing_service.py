import logging

from requests import Session

from src.database.models.stock_models import StockPricing
from .base_service import BaseService
from ..database.repositories.stock_pricing_repository import StockPricingRepository

logger = logging.getLogger(__name__)


class StockPricingService(BaseService):
    def get_repository(self, db: Session):
        return StockPricingRepository(db)

    def download_data(self, ticker: str, period: str):
        return self.pricing_provider.download_pricing_data(ticker, period)

    def transform_data(self, key, data):
        """
        Transform downloaded data into a format suitable for storage.
        """
        transformed_data = []

        # Print column names for debugging
        logger.info(f"Columns in data: {data.columns.tolist()}")

        try:
            for index, row in data.iterrows():
                # Transform the row into a dictionary
                transformed_data.append({
                    "Ticker": key,
                    "Date": index.to_pydatetime(),
                    "Open": row[StockPricing.Open.name],
                    "High": row[StockPricing.High.name],
                    "Low": row[StockPricing.Low.name],
                    "Close": row[StockPricing.Close.name],
                    "Volume": row[StockPricing.Volume.name]
                })
        except Exception as e:
            logger.error(f"Error transforming data: {e}", exc_info=True)
            raise  # Re-raise the exception to stop further processing

        return transformed_data

    def fetch_pricing_data(self, ticker: str, period: str):
        cache_key = f"pricing_{ticker}"
        return self.fetch_data(ticker, cache_key, period)

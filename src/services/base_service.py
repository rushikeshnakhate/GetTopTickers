import json
import logging
from abc import ABC, abstractmethod
from sqlalchemy.orm import Session
from ..cache.base_cache import BaseCache
from ..cache.sqlite_cache import SQLiteCache
from ..providers.pricing_provider_base import PricingProviderBase

logger = logging.getLogger(__name__)


class BaseService(ABC):
    def __init__(self, db_engine, pricing_provider: PricingProviderBase):
        self.cache = SQLiteCache(db_session=db_engine.get_session())
        self.db_engine = db_engine
        self.pricing_provider = pricing_provider

    @abstractmethod
    def get_repository(self, db: Session):
        """Return the repository for the specific table."""
        pass

    @abstractmethod
    def download_data(self, ticker: str, period: str):
        """Download data from the provider."""
        pass

    @abstractmethod
    def transform_data(self, key: str, data):
        """Transform downloaded data into a format suitable for storage."""
        pass

    def fetch_data(self, ticker: str, cache_key: str, period: str):
        """Common logic for fetching data (with caching)."""
        # Step 1: Check Redis cache
        cached_data = self.cache.get(cache_key)
        db = self.db_engine.get_session()
        repository = self.get_repository(db)
        try:
            if cached_data:
                db_data = repository.get_by_ticker(ticker)
                if db_data:
                    logger.info("Data found in Databse cache for ticker={}, cache_key={}".format(ticker, cache_key))
                    # Convert SQLAlchemy objects to dictionaries
                    db_data_dict = [self._model_to_dict(item) for item in db_data]
                    # Store in Redis for future cache hits (serialize to JSON)
                    self.cache.set(cache_key, db_data_dict)
                    return db_data_dict
            else:
                # Step 3: Download data from provider
                logger.info(
                    "Data not found in cache for ticker={}, cache_key={},Downloading now..".format(ticker, cache_key))
                downloaded_data = self.download_data(ticker, period)
                transformed_data = self.transform_data(ticker, downloaded_data)

                # Save to SQLite
                for item in transformed_data:
                    repository.add(**item)  # Pass item as kwargs to the add method

                # Save to Redis (serialize to JSON)
                self.cache.set(cache_key, transformed_data)

                return transformed_data
        finally:
            db.close()

    def _model_to_dict(self, model_instance):
        """
        Convert a SQLAlchemy model instance to a dictionary.
        """
        return {column.name: getattr(model_instance, column.name) for column in model_instance.__table__.columns}

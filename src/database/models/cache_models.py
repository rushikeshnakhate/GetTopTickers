# Model for cache table
from sqlalchemy import Column, String, Integer, Text, DateTime, ForeignKey

from src.database.sqlite_db_engine import Base


class Cache(Base):
    __tablename__ = "cache"

    key = Column(String, primary_key=True)  # Cache key (e.g., "pricing_AAPL")
    value = Column(Text)  # Cached value (stored as JSON string)
    expires_at = Column(DateTime)  # Expiration time for the cache entry

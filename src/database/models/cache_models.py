from sqlalchemy import Column, String, DateTime

from src.database.models.base_models import Base


class Cache(Base):
    __tablename__ = "cache"

    key = Column(String, primary_key=True)  # Cache key
    pd_pickle_file_path = Column(String)  # Store file path as a string
    expires_at = Column(DateTime)  # Expiration time for the cache entry

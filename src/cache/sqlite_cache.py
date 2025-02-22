from datetime import datetime, timedelta
import json
from sqlalchemy.orm import Session
from .base_cache import BaseCache
from ..database.models.cache_models import Cache


class SQLiteCache(BaseCache):
    def __init__(self, db_session: Session):
        self.db_session = db_session

    def get(self, key: str):
        """Get data from the cache."""
        cache_entry = self.db_session.query(Cache).filter(
            Cache.key == key,
            (Cache.expires_at.is_(None) | (Cache.expires_at > datetime.utcnow()))
        ).first()
        if cache_entry:
            # Deserialize the JSON string back into a Python object
            data = json.loads(cache_entry.value)
            # Convert datetime strings back to datetime objects (if needed)
            if isinstance(data, dict) and "date" in data:
                data["date"] = datetime.fromisoformat(data["date"])
            return data
        return None

    def set(self, key: str, value, ttl: int = 3600):
        """Set data in the cache with a TTL (time-to-live)."""
        expires_at = (datetime.utcnow() + timedelta(seconds=ttl)) if ttl else None
        # Serialize the Python object into a JSON string
        value_json = json.dumps(value, default=str)  # Use default=str to handle non-serializable types
        cache_entry = Cache(key=key, value=value_json, expires_at=expires_at)
        self.db_session.merge(cache_entry)  # Insert or update the cache entry
        self.db_session.commit()

    def exists(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        return self.db_session.query(Cache).filter(
            Cache.key == key,
            (Cache.expires_at.is_(None) | (Cache.expires_at > datetime.utcnow()))
        ).first() is not None

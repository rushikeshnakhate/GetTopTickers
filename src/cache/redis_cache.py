import redis
import json
from datetime import timedelta
from .base_cache import BaseCache


class RedisCache(BaseCache):
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.cache = redis.Redis(host=host, port=port, db=db)

    def get(self, key: str):
        """Get data from Redis cache."""
        cached_data = self.cache.get(key)
        return json.loads(cached_data) if cached_data else None

    def set(self, key: str, value, ttl: int = 3600):
        """Set data in Redis cache with a TTL (time-to-live)."""
        self.cache.setex(key, timedelta(seconds=ttl), json.dumps(value))

    def exists(self, key: str) -> bool:
        """Check if a key exists in Redis cache."""
        return self.cache.exists(key) == 1

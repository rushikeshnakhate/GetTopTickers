import redis
import json
import threading
from datetime import timedelta
from .base_cache import BaseCache


class RedisCache(BaseCache):
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, host="localhost", port=6379, db=0):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(RedisCache, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, host="localhost", port=6379, db=0):
        if self._initialized:
            return
        self.cache = redis.Redis(host=host, port=port, db=db)
        self._initialized = True

    def get(self, key: str):
        cached_data = self.cache.get(key)
        return json.loads(cached_data) if cached_data else None

    def set(self, key: str, value, ttl: int = 3600):
        self.cache.setex(key, timedelta(seconds=ttl), json.dumps(value))

    def exists(self, key: str) -> bool:
        return self.cache.exists(key) == 1

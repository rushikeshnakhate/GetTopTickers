import os
import pickle
import threading
from datetime import datetime, timedelta
import logging

from .base_cache import BaseCache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PandasCache(BaseCache):
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, base_path: str = None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(PandasCache, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, base_path: str = None):
        if self._initialized:
            return
        self.base_path = base_path or os.path.join(os.getcwd(), "..", "cache_files")
        os.makedirs(self.base_path, exist_ok=True)
        self._initialized = True

    def _get_file_path(self, key: str) -> str:
        return os.path.join(self.base_path, f"{key}.pkl")

    def get(self, key: str):
        file_path = self._get_file_path(key)
        if not os.path.exists(file_path):
            return None

        with open(file_path, "rb") as f:
            cache_entry = pickle.load(f)

        if cache_entry.get("expires_at") and cache_entry["expires_at"] < datetime.utcnow():
            os.remove(file_path)
            return None

        return cache_entry["value"]

    def set(self, key: str, value, ttl: int = 3600):
        expires_at = datetime.utcnow() + timedelta(seconds=ttl) if ttl else None
        cache_entry = {"value": value, "expires_at": expires_at}
        file_path = self._get_file_path(key)
        with open(file_path, "wb") as f:
            pickle.dump(cache_entry, f)

    def exists(self, key: str) -> bool:
        file_path = self._get_file_path(key)
        if not os.path.exists(file_path):
            return False

        with open(file_path, "rb") as f:
            cache_entry = pickle.load(f)

        if cache_entry.get("expires_at") and cache_entry["expires_at"] < datetime.utcnow():
            os.remove(file_path)
            return False

        return True

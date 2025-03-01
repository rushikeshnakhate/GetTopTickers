from abc import ABC, abstractmethod


class BaseCache(ABC):
    @abstractmethod
    def get(self, key: str):
        """Get data from the cache."""
        pass

    @abstractmethod
    def set(self, key: str, value, ttl: int = 3600):
        """Set data in the cache with a TTL (time-to-live)."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        pass

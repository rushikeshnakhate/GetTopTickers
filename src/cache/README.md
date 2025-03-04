# PyKeyCache

**PyKeyCache** is a flexible, simple, and efficient caching library for Python that supports multiple backend cache
systems: Pandas, Redis, and SQLite. It allows you to manage cache data in a consistent way using a unified API while
supporting various storage backends for different use cases.

---

## Features

- **Multiple Cache Backends**: Supports Pandas (local file-based), Redis (in-memory), and SQLite (database-based)
  caches.
- **TTL Support**: All cache backends support time-to-live (TTL) to automatically expire cache after a specified
  duration.
- **Thread-Safe**: The library is thread-safe for multi-threaded applications.
- **Easy Integration**: Integrate seamlessly with your Python projects for fast and persistent caching solutions.

---

## Supported Cache Types

1. **Pandas Cache**:
    - Stores cached data as pickle files in a local directory.
    - Suitable for local file-based caching.

2. **Redis Cache**:
    - Stores cache in Redis (memory-based caching).
    - Suitable for high-performance, distributed caching in web apps.

3. **SQLite Cache**:
    - Stores cache data in an SQLite database.
    - Suitable for persistent and lightweight database caching.

### Usage 

```
from src.cache.cache_factory import CacheFactory
from src.utils.constants import CacheType

# Get a Pandas cache instance
pandas_cache = CacheFactory.get_cache(CacheType.PANDAS)

# Set a value in the cache
pandas_cache.set("my_key", "my_value", ttl=60)

# Get the value from the cache
value = pandas_cache.get("my_key")
print(value)  # Output: my_value

# Check if a key exists
exists = pandas_cache.exists("my_key")
print(exists) # Output: True

# Get a Redis cache instance
redis_cache = CacheFactory.get_cache(CacheType.REDIS)

# Set a value in redis cache
redis_cache.set("redis_key", "redis_value", ttl=60)

# Get value from redis cache
redis_value = redis_cache.get("redis_key")
print(redis_value) # Output: redis_value

# Get a SQLite cache instance
sqlite_cache = CacheFactory.get_cache(CacheType.SQLITE)

# Set a value in sqlite cache
sqlite_cache.set("sqlite_key", "sqlite_value", ttl=60)

# Get value from sqlite cache
sqlite_value = sqlite_cache.get("sqlite_key")
print(sqlite_value) # Output: sqlite_value
```
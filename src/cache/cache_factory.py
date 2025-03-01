from src.cache.panda_cache import PandasCache
from src.cache.redis_cache import RedisCache
from src.cache.sqlite_cache import SQLiteCache
from src.utils.constants import CacheType


class CacheFactory:
    _cache_instances = {}

    @staticmethod
    def get_cache(cache_type: CacheType):
        if cache_type not in CacheFactory._cache_instances:
            if cache_type == CacheType.PANDAS:
                CacheFactory._cache_instances[cache_type] = PandasCache()
            elif cache_type == CacheType.REDIS:
                CacheFactory._cache_instances[cache_type] = RedisCache()
            elif cache_type == CacheType.SQLITE:
                CacheFactory._cache_instances[cache_type] = SQLiteCache()
            else:
                raise ValueError(f"Unknown cache type: {cache_type}")
        return CacheFactory._cache_instances[cache_type]

import pickle
import threading
import logging
from datetime import datetime, timedelta
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from .base_cache import BaseCache
from ..database.models.base_models import Base
from ..database.models.cache_models import Cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SQLiteCache(BaseCache):
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, db_url="sqlite:///cache.db"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SQLiteCache, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, db_url="sqlite:///cache.db"):
        if self._initialized:
            return
        self.db_url = db_url
        self.engine = create_engine(self.db_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.db_session = self.SessionLocal()
        self.create_tables()
        inspector = inspect(self.engine)
        tables = inspector.get_table_names()
        print(f"Tables in the database: {tables}")
        self._initialized = True

    def get(self, key: str):
        cache_entry = self.db_session.query(Cache).filter(
            Cache.key == key,
            (Cache.expires_at.is_(None) | (Cache.expires_at > datetime.utcnow()))
        ).first()

        if cache_entry:
            return pickle.loads(cache_entry.pd_pickle_file_path)
        return None

    def set(self, key: str, value, ttl: int = 3600):
        expires_at = datetime.utcnow() + timedelta(seconds=ttl) if ttl else None
        value_pkl = pickle.dumps(value)
        cache_entry = Cache(key=key, pd_pickle_file_path=value_pkl, expires_at=expires_at)
        self.db_session.merge(cache_entry)
        self.db_session.commit()

    def exists(self, key: str) -> bool:
        return self.db_session.query(Cache).filter(
            Cache.key == key,
            (Cache.expires_at.is_(None) | (Cache.expires_at > datetime.utcnow()))
        ).first() is not None

    def create_tables(self):
        Base.metadata.create_all(bind=self.engine)

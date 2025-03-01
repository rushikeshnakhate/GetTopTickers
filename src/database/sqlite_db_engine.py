from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from .database_engine import DatabaseEngine
from .models.cache_models import Cache  # Import the Cache model

Base = declarative_base()


class SQLiteDbEngine(DatabaseEngine):
    def __init__(self, db_url: str = "sqlite:///financial_data.db"):
        self.db_url = db_url
        self.engine = create_engine(self.db_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.create_tables()

    def get_session(self) -> Session:
        return self.SessionLocal()

    def create_tables(self):
        # Ensure all models are registered with Base.metadata
        Base.metadata.create_all(bind=self.engine)

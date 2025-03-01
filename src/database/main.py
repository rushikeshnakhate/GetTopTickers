import logging

from sqlalchemy import inspect  # Add this import

from src.cache.sqlite_cache import SQLiteCache
from src.database.sqlite_db_engine import SQLiteDbEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_db_engine():
    # Initialize database engine and create tables
    db_engine = SQLiteDbEngine()
    logger.info("Database engine initialized and tables created.")

    # Initialize SQLite cache
    db_session = db_engine.get_session()
    cache = SQLiteCache(db_session)
    logger.info("SQLite cache initialized.")

    # List all tables in the database
    inspector = inspect(db_engine.engine)  # Use the engine from db_engine
    tables = inspector.get_table_names()
    logger.info(f"Tables in the database: {tables}")
    return db_engine, cache

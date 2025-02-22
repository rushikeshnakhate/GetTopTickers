from sqlite3 import Date

from sqlalchemy import Column, String, Integer

from src.database.sqlite_db_engine import Base


class StockListing(Base):
    __tablename__ = 'stock_listing'  # The table name in the database

    # Defining columns
    symbol = Column('SYMBOL', String, primary_key=True)  # Symbol is the primary key
    company_name = Column('NAME OF COMPANY', String, nullable=False)
    series = Column('SERIES', String, nullable=False)
    date_of_listing = Column('DATE OF LISTING', String, nullable=False)
    paid_up_value = Column('PAID UP VALUE', Integer, nullable=False)
    market_lot = Column('MARKET LOT', Integer, nullable=False)
    isin_number = Column('ISIN NUMBER', String, nullable=False)
    face_value = Column('FACE VALUE', Integer, nullable=False)

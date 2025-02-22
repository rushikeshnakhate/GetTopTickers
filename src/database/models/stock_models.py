from sqlalchemy import Float, Column, String, DateTime
from sqlalchemy import PrimaryKeyConstraint  # Import PrimaryKeyConstraint
from src.database.sqlite_db_engine import Base


# Model for stock pricing data
class StockPricing(Base):
    __tablename__ = "stock_pricing"

    # Composite primary key: ticker + Date
    Ticker = Column(String, primary_key=True)  # Stock ticker (e.g., "AAPL")
    Date = Column(DateTime, primary_key=True)  # Date of the pricing data

    # Other columns
    Open = Column(Float, nullable=False)  # Opening price
    High = Column(Float, nullable=False)  # High price
    Low = Column(Float, nullable=False)  # Low price
    Close = Column(Float, nullable=False)  # Closing price
    Volume = Column(Float, nullable=False)  # Trading volume
    Dividend = Column(Float, nullable=True)  # Dividend amount
    StockSplits = Column(Float, nullable=True)  # Stock splits

    # Optional: Explicitly define the composite primary key
    __table_args__ = (
        PrimaryKeyConstraint('Ticker', 'Date'),
    )

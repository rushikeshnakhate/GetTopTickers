from sqlalchemy import Column, String, Float, PrimaryKeyConstraint

from src.database.sqlite_db_engine import Base


class BalanceSheet(Base):
    __tablename__ = "balance_sheet"

    Ticker = Column(String, primary_key=True)  # Stock ticker (e.g., "AAPL")
    year = Column(String, primary_key=True)  # Year of the balance sheet (e.g., "2024")

    # Assets
    total_assets = Column(Float)
    total_current_assets = Column(Float)
    cash_and_cash_equivalents = Column(Float)
    net_receivables = Column(Float)
    inventory = Column(Float)
    total_non_current_assets = Column(Float)
    property_plant_equipment_net = Column(Float)

    # Liabilities
    total_liabilities = Column(Float)
    total_current_liabilities = Column(Float)
    accounts_payable = Column(Float)
    short_term_debt = Column(Float)
    total_non_current_liabilities = Column(Float)
    long_term_debt = Column(Float)

    # Equity
    total_equity = Column(Float)
    common_stock = Column(Float)
    retained_earnings = Column(Float)
    # Optional: Explicitly define the composite primary key
    __table_args__ = (
        PrimaryKeyConstraint('Ticker', 'year'),
    )

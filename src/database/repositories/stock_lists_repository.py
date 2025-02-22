from sqlalchemy.orm import Session

from .base_repository import BaseRepository
from ..models.stock_lists_model import StockListing


class StockListsRepository(BaseRepository):
    def __init__(self, db: Session):
        """
        Initialize the repository with the BalanceSheet model.
        """
        super().__init__(db, StockListing)

    def get_by_ticker(self, ticker: str):
        """
        Get all balance sheet records for a given ticker.
        """
        return self.db.query(self.model).filter(self.model.Ticker == ticker).all()

    def add_from_dict(self, ticker: str, year: str, data: dict):
        """
        Add a new balance sheet record from a dictionary.
        """
        return self.add(ticker=ticker, year=year, data=str(data))  # Pass ticker, year, and data as kwargs

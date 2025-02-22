from sqlalchemy.orm import Session
from src.database.models.balance_sheet_models import BalanceSheet
from .base_repository import BaseRepository


class BalanceSheetRepository(BaseRepository):
    def __init__(self, db: Session):
        """
        Initialize the repository with the BalanceSheet model.
        """
        super().__init__(db, BalanceSheet)

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

from sqlalchemy.orm import Session
from src.database.models.stock_models import StockPricing
from .base_repository import BaseRepository


class StockPricingRepository(BaseRepository):
    def __init__(self, db: Session):
        """
        Initialize the repository with the StockPricing model.
        """
        super().__init__(db, StockPricing)

    def get_by_ticker(self, ticker: str):
        """
        Get all stock pricing records for a given ticker.
        """
        return self.db.query(self.model).filter(self.model.Ticker == ticker).all()

    def add_from_dict(self, ticker: str, data: dict):
        """
        Add a new stock pricing record from a dictionary.
        """
        return self.add(ticker=ticker, **data)  # Pass ticker and data as kwargs

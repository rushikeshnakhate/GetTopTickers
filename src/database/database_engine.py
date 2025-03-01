from abc import ABC, abstractmethod
from sqlalchemy.orm import Session


class DatabaseEngine(ABC):
    @abstractmethod
    def get_session(self) -> Session:
        """Return a database session."""
        pass

    @abstractmethod
    def create_tables(self):
        """Create database tables."""
        pass

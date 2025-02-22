from typing import List
from sqlalchemy.orm import Session


class BaseRepository:
    def __init__(self, db: Session, model):
        """
        Initialize the repository with a database session and a model class.
        """
        self.db = db
        self.model = model

    def get_by_id(self, id: str):
        """
        Get a record by its ID.
        """
        return self.db.query(self.model).filter(self.model.id == id).first()

    def get_all(self) -> List:
        """
        Get all records for the model.
        """
        return self.db.query(self.model).all()

    def add(self, **kwargs):
        """
        Add a new record to the database.
        """
        entity = self.model(**kwargs)  # Create a model instance using kwargs
        self.db.add(entity)
        self.db.commit()
        self.db.refresh(entity)
        return entity

    def delete(self, id: str):
        """
        Delete a record by its ID.
        """
        entity = self.get_by_id(id)
        if entity:
            self.db.delete(entity)
            self.db.commit()
        return entity

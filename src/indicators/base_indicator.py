from abc import ABC, abstractmethod


class BaseIndicator(ABC):
    """Abstract base class for financial indicators."""

    def __init__(self, period: int):
        """Initialize with the period for the indicator."""
        self.period = period

    @abstractmethod
    def calculate(self, data):
        """Calculate the indicator based on data."""
        pass

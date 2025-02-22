from abc import ABC, abstractmethod
import pandas as pd


class PricingProviderBase(ABC):
    @abstractmethod
    def download_pricing_data(self, ticker: str, period: str) -> pd.DataFrame:
        """Download pricing data for the given ticker and period."""
        pass

    @abstractmethod
    def download_balance_sheet(self, ticker: str, period: str) -> dict:
        """Download balance sheet data for the given ticker."""
        pass

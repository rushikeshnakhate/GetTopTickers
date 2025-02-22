import logging
from datetime import datetime

import pandas as pd
from sqlalchemy.orm import Session
from .base_service import BaseService
from ..database.repositories.balance_sheet_repository import BalanceSheetRepository

logger = logging.getLogger(__name__)


class BalanceSheetService(BaseService):
    def get_repository(self, db: Session):
        """
        Return the repository instance for the BalanceSheet model.
        """
        return BalanceSheetRepository(db)

    def download_data(self, ticker: str, period: str):
        """
        Download balance sheet data from the provider.
        """
        return self.pricing_provider.download_balance_sheet(ticker, period)

    def transform_data(self, key, data):
        """
        Transform downloaded data into a format suitable for storage in the BalanceSheet table.
        Only extract the most important columns and ignore the rest.
        """
        # List of important columns to extract
        important_columns = [
            "Total Assets",
            "Total Current Assets",
            "Cash And Cash Equivalents",
            "Net Receivables",
            "Inventory",
            "Total Non Current Assets",
            "Property Plant Equipment Net",
            "Total Liabilities",
            "Total Current Liabilities",
            "Accounts Payable",
            "Short Term Debt",
            "Total Non Current Liabilities",
            "Long Term Debt",
            "Total Equity",
            "Common Stock",
            "Retained Earnings",
        ]

        transformed_data = []
        try:
            # Print column names for debugging
            logger.info(f"Columns in data: {data}")

            # Iterate through the data and transform it
            for timestamp, values in data.items():
                # Convert Timestamp to string for the 'year' column
                year = timestamp.strftime("%Y-%m-%d")  # Convert to string in YYYY-MM-DD format

                # Create a dictionary for each year's data
                balance_sheet_entry = {
                    "Ticker": key,  # Stock ticker (e.g., "RELIANCE.NS")
                    "year": year,  # Year of the balance sheet as a string
                }

                # Add only the important columns to the dictionary
                for field in important_columns:
                    # Check if the field exists in the data
                    if field in values:
                        # Convert field names to snake_case if necessary
                        field_name = field.lower().replace(" ", "_")

                        # Replace NaN values with None
                        value = values[field]
                        if pd.isna(value):
                            value = None

                        balance_sheet_entry[field_name] = value
                    else:
                        # If the field is not in the data, set it to None
                        field_name = field.lower().replace(" ", "_")
                        balance_sheet_entry[field_name] = None

                transformed_data.append(balance_sheet_entry)

        except Exception as e:
            logger.error(f"Error transforming data: {e}", exc_info=True)
            raise  # Re-raise the exception to stop further processing

        return transformed_data

    def fetch_balance_sheet(self, ticker: str, period: str):
        """
        Fetch balance sheet data for a given ticker.
        """
        cache_key = f"balance_sheet_{ticker}"
        return self.fetch_data(ticker, cache_key, period)

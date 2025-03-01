import asyncio
from typing import Dict, List

import pandas as pd
import yfinance as yf

from src.providers.yahoo_provider import YahooProvider


class YahooProviderBalanceSheet(YahooProvider):
    def download_balance_sheet(self, ticker: str) -> pd.DataFrame:
        """
        Download the balance sheet data for a given ticker using Yahoo Finance.
        """
        ticker = yf.Ticker(ticker)
        balance_sheet_data = ticker.balance_sheet
        return balance_sheet_data.T  # Transpose so that columns are the dates and rows are the account categories

    async def download_balance_sheet_async(self, ticker: str) -> pd.DataFrame:
        """
        Asynchronously download the balance sheet data for a given ticker using Yahoo Finance.
        """
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, self.download_balance_sheet, ticker)
        return data

    def download_multiple_balance_sheets(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Download balance sheet data for multiple symbols using Yahoo Finance.
        """
        balance_sheets = {}
        for ticker in symbols:
            balance_sheets[ticker] = self.download_balance_sheet(ticker)
        return balance_sheets

    async def download_multiple_balance_sheets_async(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Asynchronously download balance sheet data for multiple symbols using Yahoo Finance.
        """
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(
            None, self.download_multiple_balance_sheets, symbols
        )
        return data

import pandas as pd
from typing import Dict, Optional, List
import aiohttp
import asyncio
from .base_provider import BaseProvider


class JugadProvider(BaseProvider):
    def __init__(self, base_url: str = "https://api.jugad.com"):
        self.base_url = base_url

    async def _fetch_data(self, url: str) -> Dict:
        """
        Asynchronously fetch data from the Jugad API.
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                return await response.json()

    async def download_historical_data_async(self,
                                             ticker: str,
                                             start_date: Optional[str] = None,
                                             end_date: Optional[str] = None,
                                             interval: str = "1d",
                                             ) -> pd.DataFrame:
        """
        Asynchronously download historical stock data for a given ticker using Jugad.
        """
        url = f"{self.base_url}/historical?ticker={ticker}&start_date={start_date}&end_date={end_date}&interval={interval}"
        data = await self._fetch_data(url)
        return pd.DataFrame(data)

    async def download_close_prices_async(self,
                                          ticker: str,
                                          start_date: Optional[str] = None,
                                          end_date: Optional[str] = None,
                                          interval: str = "1d",
                                          ) -> pd.Series:
        """
        Asynchronously download close prices for a given ticker using Jugad.
        """
        data = await self.download_historical_data_async(ticker, start_date, end_date, interval)
        return data["close"]

    async def download_multiple_symbols_async(self,
                                              symbols: List[str],
                                              start_date: Optional[str] = None,
                                              end_date: Optional[str] = None,
                                              interval: str = "1d",
                                              ) -> Dict[str, pd.DataFrame]:
        """
        Asynchronously download historical data for multiple symbols using Jugad.
        """
        tasks = [
            self.download_historical_data_async(ticker, start_date, end_date, interval)
            for ticker in symbols
        ]
        results = await asyncio.gather(*tasks)
        return {ticker: df for ticker, df in zip(symbols, results)}

    # Synchronous methods (not implemented for Jugad)
    def download_historical_data(self, *args, **kwargs):
        raise NotImplementedError("Synchronous download not supported for Jugad.")

    def download_close_prices(self, *args, **kwargs):
        raise NotImplementedError("Synchronous download not supported for Jugad.")

    def download_multiple_symbols(self, *args, **kwargs):
        raise NotImplementedError("Synchronous download not supported for Jugad.")

import asyncio
import logging
from typing import Dict, Optional, List

import pandas as pd
import yfinance as yf

from .base_provider import BaseProvider


class YahooProvider(BaseProvider):
    def download_historical_data(self,
                                 ticker: str,
                                 start_date: Optional[str] = None,
                                 end_date: Optional[str] = None,
                                 interval: str = "1d",
                                 ) -> pd.DataFrame:
        """
        Download historical stock data for a given ticker using Yahoo Finance.
        """
        try:
            ticker = yf.Ticker(ticker)
            data = ticker.history(start=start_date, end=end_date, interval=interval)
            return data
        except Exception as e:
            logging.info(f" YahooProvider:Error fetching data for {ticker}: {e}")
            return None

    async def download_historical_data_async(self,
                                             ticker: str,
                                             start_date: Optional[str] = None,
                                             end_date: Optional[str] = None,
                                             interval: str = "1d",
                                             ) -> pd.DataFrame:
        """
        Asynchronously download historical stock data for a given ticker using Yahoo Finance.
        """
        # Use yfinance's synchronous method in a thread pool
        try:
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None, self.download_historical_data, ticker, start_date, end_date, interval
            )
            return data
        except Exception as e:
            logging.info(f"Error fetching data for {ticker}: {e}")
            return None

    def download_close_prices(self,
                              ticker: str,
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None,
                              interval: str = "1d",
                              ) -> pd.Series:
        """
        Download close prices for a given ticker using Yahoo Finance.
        """
        data = self.download_historical_data(ticker, start_date, end_date, interval)
        return data["Close"]

    async def download_close_prices_async(self,
                                          ticker: str,
                                          start_date: Optional[str] = None,
                                          end_date: Optional[str] = None,
                                          interval: str = "1d",
                                          ) -> pd.Series:
        """
        Asynchronously download close prices for a given ticker using Yahoo Finance.
        """
        data = await self.download_historical_data_async(ticker, start_date, end_date, interval)
        return data["Close"]

    def download_multiple_symbols(self,
                                  symbols: List[str],
                                  start_date: Optional[str] = None,
                                  end_date: Optional[str] = None,
                                  interval: str = "1d",
                                  ) -> Dict[str, pd.DataFrame]:
        """
        Download historical data for multiple symbols using Yahoo Finance.
        """
        data = yf.download(symbols, start=start_date, end=end_date, interval=interval, group_by="ticker")
        return {ticker: data[ticker] for ticker in symbols}

    async def download_multiple_symbols_async(self,
                                              symbols: List[str],
                                              start_date: Optional[str] = None,
                                              end_date: Optional[str] = None,
                                              interval: str = "1d",
                                              ) -> Dict[str, pd.DataFrame]:
        """
        Asynchronously download historical data for multiple symbols using Yahoo Finance.
        """
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, self.download_multiple_symbols,
                                          symbols, start_date, end_date, interval)
        return data

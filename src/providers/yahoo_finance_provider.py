import yfinance as yf
import pandas as pd
from .pricing_provider_base import PricingProviderBase


class YahooFinanceProvider(PricingProviderBase):
    def download_pricing_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        stock = yf.Ticker(ticker)
        return stock.history(period=period)

    def download_balance_sheet(self, ticker: str, period: str = "1y") -> dict:
        stock = yf.Ticker(ticker)
        balance_sheet = stock.balance_sheet
        return balance_sheet.to_dict()

import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentals import Fundamentals
from .pricing_provider_base import PricingProviderBase


class AlphaVantageProvider(PricingProviderBase):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def download_pricing_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        ts = TimeSeries(key=self.api_key, output_format='pandas')
        data, _ = ts.get_daily(symbol=ticker, outputsize='full')
        return data

    def download_balance_sheet(self, ticker: str) -> dict:
        fs = Fundamentals(key=self.api_key, output_format='pandas')
        balance_sheet, _ = fs.get_balance_sheet_annual(symbol=ticker)
        return balance_sheet.to_dict()

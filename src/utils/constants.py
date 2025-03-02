import dataclasses
from enum import Enum


@dataclasses.dataclass
class GlobalStockData:
    START_DATE: str = "Start Date"
    END_DATE: str = "End Date"
    LOW: str = "Low"
    CLOSE: str = "Close"
    VOLUME: str = "Volume"
    STOCK_SPLIT: str = "Stock Splits"
    DIVIDENDS: str = "Dividends"


@dataclasses.dataclass
class GLobalColumnName:
    TICKER: str = "Ticker"
    ticker_nifty50 = "^NSEI"  # Nifty 50 Index
    ticker_nifty100 = "^NSE100"  # Nifty 100 Index


class Providers(Enum):
    YAHOO = "yahoo"
    YAHOO_FOR_BALANCE_SHEET = "yahoo_for_balance_sheet"
    JUGAD = "jugad"
    CUSTOM = "custom"


class CacheType(Enum):
    SQLITE = "sqlite"
    PANDAS = "pandas"
    REDIS = "redis"


@dataclasses.dataclass
class StockListsColumns:
    SYMBOL: str = "SYMBOL"
    NAME_OF_COMPANY: str = "NAME OF COMPANY"
    SERIES: str = "SERIES"
    DATE_OF_LISTING: str = " DATE OF LISTING"
    PAID_UP_VALUE: str = "PAID UP VALUE"
    MARKET_LOT: str = "MARKET LOT"
    ISIN_NUMBER: str = "ISIN NUMBER"
    FACE_VALUE: str = "FACE VALUE"


@dataclasses.dataclass
class Stocks:
    EQUITY: str = "EQ"
    NSE_EXTENSION: str = ".NS"

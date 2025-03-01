import dataclasses
from enum import Enum


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

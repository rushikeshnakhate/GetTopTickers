## PyPriceFetcherService

`PyPriceFetcherService` is a Python library designed to fetch and cache stock price data from various sources such as Yahoo and custom providers. The library provides functionality to fetch stock prices, balance sheet data, and more, with caching and error handling support for efficient data retrieval.

### Features

- Fetch stock price data for single or multiple tickers.
- Supports different data sources (Yahoo, Custom).
- Caching mechanism to optimize data retrieval.
- Asynchronous support for fetching data in parallel.
- Get balance sheet data for a given stock ticker.
- Configurable providers and cache types.


```from pyPriceFetcherService import DataFetcherService
data_fetcher = DataFetcherService()
Fetch Stock List Data
# Get stock list data
stock_list = data_fetcher.get_stock_list_service_data()
Fetch Close Price Data for Single Ticker
```

### Get close price for a specific ticker
```ticker = "AAPL"
df = data_fetcher.get_close_price_service(ticker, start_date="2022-01-01", end_date="2022-12-31")
```

### Get close price data for multiple tickers
```
tickers = ["AAPL", "GOOG", "AMZN"]
df = data_fetcher.get_close_price_service_bulk(start_date="2022-01-01", end_date="2022-12-31", ticker_list=tickers)
```
### Fetch Balance Sheet Data
### Get balance sheet data for a specific ticker
```
ticker = "AAPL"
balance_sheet_df = data_fetcher.get_balance_sheet_service(ticker)
```

### Asynchronous Data Fetching
### Asynchronous fetching of close price data
```
df_async = await data_fetcher.get_close_price_service_async(ticker="AAPL", start_date="2022-01-01", end_date="2022-12-31")
```

### Services
- DataFetcherService:Responsible for fetching stock list, close prices, and balance sheet data.
- ClosePriceService:Fetches historical close price data from various sources, supports caching.
- BalanceSheetService:Fetches balance sheet data from Yahoo Finance.
- StockListService: Fetches stock list data from a custom CSV file.

### Caching
The library uses a caching mechanism to store and retrieve data efficiently. The cache can be configured to use different cache types like PANDAS. Cached data is retrieved if available, and new data is fetched and cached for future use.
import logging
from pathlib import Path

from tabulate import tabulate

from src.indicators.main import get_indicator
from src.performance_matrix.main import get_performance_metrics
from src.service.main import get_stocks
from src.utils.logging_config import setup_logging
from src.utils.utils import to_pickled_df

# Setup logging
logger = logging.getLogger(__name__)
project_directory = Path(__file__).resolve().parent.parent
setup_logging(project_directory)

indicator_data = []
performance_data = []
start_date = "2024-01-01"
end_date = "2024-12-31"

ticker_list, total_tickers_count = get_stocks()
ticker_list = ["TCS.NS", "INFY.NS"]
for index, ticker in enumerate(ticker_list, start=1):
    remaining = total_tickers_count - index  # Calculate remaining symbols
    logger.info(f"Downloading data for {index}/{total_tickers_count}: {ticker} | Remaining: {remaining}")
    # indicators_df = get_indicator(ticker=ticker, start_date=start_date, end_date=end_date)
    # indicator_data.append(indicators_df)

    df = get_performance_metrics(ticker=ticker, start_date=start_date, end_date=end_date)
    performance_data.append(df)

# print(performance_data)
# indicator_data_df = to_pickled_df(indicator_data, pkl_file_name="indicators")
performance_data_df = to_pickled_df(performance_data, pkl_file_name="performance")
# print(tabulate(indicator_data_df, headers='keys', tablefmt='psql'))
print(tabulate(performance_data_df, headers='keys', tablefmt='psql'))
#

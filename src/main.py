import logging
from pathlib import Path

from tabulate import tabulate

from src.indicators.main import get_indicator_bulk
from src.performance_matrix.main import get_performance_metrics_bulk
from src.service.main import dataFetcher, get_stocks
from src.strategies.main import StrategyFactory
from src.utils.constants import GLobalColumnName
from src.utils.logging_config import setup_logging

# Setup logging
logger = logging.getLogger(__name__)
project_directory = Path(__file__).resolve().parent.parent
setup_logging(project_directory)

start_date = "2024-01-01"
end_date = "2024-01-31"

market_data = dataFetcher.get_close_price_service(ticker=GLobalColumnName.ticker_nifty50,
                                                  start_date=start_date,
                                                  end_date=end_date)
ticker_list, _ = get_stocks()

close_price = dataFetcher.get_close_price_service_bulk(ticker_list=ticker_list,
                                                       start_date=start_date,
                                                       end_date=end_date)

indicators_df = get_indicator_bulk(ticker_data_df=close_price,
                                   ticker_list=ticker_list,
                                   start_date=start_date,
                                   end_date=end_date)

performance_df = get_performance_metrics_bulk(ticker_data_df=close_price,
                                              ticker_list=ticker_list,
                                              market_data=market_data,
                                              start_date=start_date, end_date=end_date)

df = StrategyFactory(indicators_df=indicators_df, performance_df=performance_df).run_all_strategies()
logging.info(tabulate(df, headers='keys', tablefmt='psql'))

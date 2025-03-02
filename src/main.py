import logging
from pathlib import Path

from tabulate import tabulate

from src.indicators.main import IndicatorFactory
from src.performance_matrix.main import get_performance_metrics_bulk
from src.service.main import dataFetcher, get_stocks
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
ticker_list = ["TCS.NS", "INFY.NS"]
close_price_data_for_tickers_df = dataFetcher.get_close_price_service_bulk(ticker_list=ticker_list,
                                                                           start_date=start_date,
                                                                           end_date=end_date)

indicators_df = IndicatorFactory().get_indicator_bulk(ticker_data_df=close_price_data_for_tickers_df,
                                                      ticker_list=ticker_list,
                                                      start_date=start_date,
                                                      end_date=end_date)
performance_df = get_performance_metrics_bulk(ticker_data_df=close_price_data_for_tickers_df,
                                              ticker_list=ticker_list,
                                              market_data=market_data,
                                              start_date=start_date, end_date=end_date)

logging.info(tabulate(indicators_df, headers='keys', tablefmt='psql'))
logging.info(tabulate(performance_df, headers='keys', tablefmt='psql'))
# Define strategies
# Define strategies
# rsi_strategy = RSIMomentumStrategy(indicator_data_df, top_n=5)
# sharpe_strategy = SharpeRatioStrategy(performance_data_df, top_n=5)
# # max_drawdown_strategy = MaximumDrawdownStrategy(performance_data_df, top_n=5)
#
# # Run strategies
# print("RSI Momentum Tickers:", rsi_strategy.run())
# print("Sharpe Ratio Tickers:", sharpe_strategy.run())
# print("Maximum Drawdown Tickers:", max_drawdown_strategy.run())

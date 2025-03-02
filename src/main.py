import pandas as pd
from tabulate import tabulate
import logging

from src.indicators.main import get_indicator_bulk
from src.performance_matrix.main import get_performance_metrics_bulk
from src.service.main import dataFetcher, get_stocks
from src.strategies.main import StrategyFactory
from src.utils.constants import GLobalColumnName
from src.utils.generate_date_range import generate_month_date_ranges

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Generate date ranges (assuming generate_month_date_ranges is defined)
date_ranges = generate_month_date_ranges(2024)

# Initialize an empty list to store results
results = []

# Loop through date ranges and process the data
for start_date, end_date in date_ranges:
    logging.info("start_date={}, end_date={}".format(start_date, end_date))

    # Fetch market data
    market_data = dataFetcher.get_close_price_service(ticker=GLobalColumnName.ticker_nifty50,
                                                      start_date=start_date,
                                                      end_date=end_date)

    # Get the list of tickers
    ticker_list, _ = get_stocks()

    # Fetch close prices for all tickers
    close_price = dataFetcher.get_close_price_service_bulk(ticker_list=ticker_list,
                                                           start_date=start_date,
                                                           end_date=end_date)

    # Calculate indicators
    indicators_df = get_indicator_bulk(ticker_data_df=close_price,
                                       ticker_list=ticker_list,
                                       start_date=start_date,
                                       end_date=end_date)

    # Calculate performance metrics
    performance_df = get_performance_metrics_bulk(ticker_data_df=close_price,
                                                  ticker_list=ticker_list,
                                                  market_data=market_data,
                                                  start_date=start_date, end_date=end_date)

    # Run all strategies
    strategy_results = StrategyFactory(indicators_df=indicators_df, performance_df=performance_df).run_all_strategies()

    # Log the results for the current date range
    logging.info(tabulate(strategy_results, headers='keys', tablefmt='psql'))

    # Append results to the list
    for strategy_name, tickers in strategy_results.items():
        results.append({
            'start_date': start_date,
            'end_date': end_date,
            'strategy_name': strategy_name,
            'tickers': tickers
        })

# Convert the results list to a DataFrame
results_df = pd.DataFrame(results)

# Log the final DataFrame
logging.info("\nFinal DataFrame:\n{}".format(tabulate(results_df, headers='keys', tablefmt='psql')))

# Save the DataFrame to a CSV file (optional)
results_df.to_csv('strategy_results.csv', index=False)

import logging

import pandas as pd
from tabulate import tabulate

from src.date_generator.date_range_generator import DateRangeGenerator
from src.indicators.main import get_indicator_bulk
from src.performance_matrix.main import get_performance_metrics_bulk
from src.service.main import get_stocks, dataFetcher
from src.strategies.main import StrategyFactory
from src.utils.constants import GLobalColumnName, GlobalConstant

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_pyport_ticker_selector(
        years: list,  # List of years to process (mandatory)
        months: list = None,
        top_n_tickers: int = GlobalConstant.TOP_N_TICKERS,
        tickers: list = None,  # List of tickers(optional)
        rebalancing_days=None,
        rebalancing_months=3,
        indicators: list = None,
        performance_matrix: list = None,
        strategies: list = None
):
    # Generate date ranges (assuming generate_month_date_ranges is defined)

    date_ranges = DateRangeGenerator(years=years, months=months, rebalancing_days=rebalancing_days,
                                     rebalancing_months=rebalancing_months).get_date_range()

    results = []
    # Loop through date ranges and process the data
    for start_date, end_date in date_ranges:
        logging.info("start_date={}, end_date={}".format(start_date, end_date))

        # Fetch market data
        market_data = dataFetcher.get_close_price_service(ticker=GLobalColumnName.ticker_nifty50,
                                                          start_date=start_date,
                                                          end_date=end_date)

        # Get the list of tickers
        ticker_list, _ = get_stocks(tickers)

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
        strategy_results = StrategyFactory(indicators_df=indicators_df,
                                           performance_df=performance_df,
                                           top_n=top_n_tickers).run_all_strategies()

        # Append results to the list
        for strategy_name, tickers in strategy_results.items():
            results.append({
                'start_date': start_date,
                'end_date': end_date,
                'strategy_name': strategy_name,
                'tickers': tickers
            })
        # Log the results for the current date range
        logging.info(tabulate(strategy_results, headers='keys', tablefmt='psql'))
    # Convert the results list to a DataFrame
    results_df = pd.DataFrame(results)

    # Log the final DataFrame
    logging.info("\nFinal DataFrame:\n{}".format(tabulate(results_df, headers='keys', tablefmt='psql')))

    # Save the DataFrame to a CSV file (optional)
    results_df.to_csv('strategy_results.csv', index=False)

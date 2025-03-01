# from src.performance_matrix.base_parameter import BasePerformanceMatrix
#
#
# class PercentageChange(BasePerformanceMatrix):
#     def calculate(self):
#         return self.stock_data.pct_change().dropna()

from src.performance_matrix.base_parameter import BasePerformanceMatrix
import pandas as pd
import logging


class PercentageChange(BasePerformanceMatrix):
    def calculate(self, method: str = 'average', start: str = None, end: str = None, column: str = 'Close') -> float:
        """
        Calculate percentage change for a given period using the specified method. :param method: Method to use for
        calculation. Options: 'first_last', 'mean_first_last', 'cumulative', 'mean_all', 'average'. :param start:
        Start date (inclusive). :param end: End date (inclusive). :param column: Column to use for calculation (
        default: 'Close'). :return: Percentage change as a single value.
        """
        if not isinstance(self.stock_data, pd.DataFrame):
            raise ValueError("stock_data must be a pandas DataFrame.")

        # if start is None or end is None:
        #     raise ValueError("start and end dates must be provided.")

        # Filter data for the specified period
        period_data = self.stock_data.loc[start:end]

        if method == 'first_last':
            return self._percentage_change_first_last(period_data, column)
        elif method == 'mean_first_last':
            return self._percentage_change_mean_first_last(period_data)
        elif method == 'cumulative':
            return self._percentage_change_cumulative(period_data, column)
        elif method == 'mean_all':
            return self._percentage_change_mean_all(period_data, column)
        elif method == 'average':
            return self._percentage_change_average(period_data, column)
        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def _percentage_change_first_last(period_data: pd.DataFrame, column: str) -> float:
        """
        Calculate percentage change between the first and last values in the period.
        :param period_data: Filtered DataFrame for the specified period.
        :param column: Column to use for calculation.
        :return: Percentage change as a single value.
        """
        first_value = period_data[column].iloc[0]
        last_value = period_data[column].iloc[-1]
        return ((last_value - first_value) / first_value) * 100

    @staticmethod
    def _percentage_change_mean_first_last(period_data: pd.DataFrame) -> float:
        """
        Calculate percentage change between the mean of first and last values in the period.
        :param period_data: Filtered DataFrame for the specified period.
        :return: Percentage change as a single value.
        """
        first_mean = (period_data['Open'].iloc[0] + period_data['Close'].iloc[0]) / 2
        last_mean = (period_data['Open'].iloc[-1] + period_data['Close'].iloc[-1]) / 2
        return ((last_mean - first_mean) / first_mean) * 100

    @staticmethod
    def _percentage_change_cumulative(period_data: pd.DataFrame, column: str) -> float:
        """
        Calculate cumulative percentage change over the period.
        :param period_data: Filtered DataFrame for the specified period.
        :param column: Column to use for calculation.
        :return: Cumulative percentage change as a single value.
        """
        daily_pct_change = period_data[column].pct_change().dropna()
        cumulative_pct_change = (daily_pct_change + 1).prod() - 1
        return cumulative_pct_change * 100

    @staticmethod
    def _percentage_change_mean_all(period_data: pd.DataFrame, column: str) -> float:
        """
        Calculate percentage change between the mean of all values in the period.
        :param period_data: Filtered DataFrame for the specified period.
        :param column: Column to use for calculation.
        :return: Percentage change as a single value.
        """
        first_mean = period_data[column].iloc[:2].mean()  # Mean of first 2 values
        last_mean = period_data[column].iloc[-2:].mean()  # Mean of last 2 values
        return ((last_mean - first_mean) / first_mean) * 100

    def _percentage_change_average(self, period_data: pd.DataFrame, column: str) -> float:
        """
        Calculate the average of all four percentage change methods.
        :param period_data: Filtered DataFrame for the specified period.
        :param column: Column to use for calculation.
        :return: Average percentage change as a single value.
        """
        v1 = self._percentage_change_first_last(period_data, column)
        v2 = self._percentage_change_mean_first_last(period_data)
        v3 = self._percentage_change_cumulative(period_data, column)
        v4 = self._percentage_change_mean_all(period_data, column)

        # Calculate the average
        average_pct_change = (v1 + v2 + v3 + v4) / 4
        # Log the values for comparison
        logging.info(
            f"First and Last: {v1:.2f}%, Mean First and Last: {v2:.2f}%, Cumulative: {v3:.2f}%, Mean All: {v4:.2f}% average_pct_change:{average_pct_change}")

        return average_pct_change

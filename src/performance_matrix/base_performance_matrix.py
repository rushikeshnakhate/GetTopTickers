import pandas as pd


class BasePerformanceMatrix:
    def __init__(self, stock_data: pd.Series):
        """
        :param stock_data: Pandas Series containing stock prices or returns.
        """
        self.stock_data = stock_data

    def calculate(self):
        """
        Method to be overridden by child classes to calculate performance metrics.
        """
        raise NotImplementedError("Subclasses must implement this method")

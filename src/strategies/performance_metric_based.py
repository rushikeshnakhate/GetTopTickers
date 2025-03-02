# Risk-Adjusted Return Strategies
from src.strategies.base_strategy import BaseStrategy


class SharpeRatioStrategy(BaseStrategy):
    def run(self):
        sorted_df = self.df.nlargest(self.top_n, 'SharpeRatio')
        return self.get_tickers(sorted_df)


class SortinoRatioStrategy(BaseStrategy):
    def run(self):
        sorted_df = self.df.nlargest(self.top_n, 'SortinoRatio')
        return self.get_tickers(sorted_df)


class CalmarRatioStrategy(BaseStrategy):
    def run(self):
        sorted_df = self.df.nlargest(self.top_n, 'CalmarRatio')
        return self.get_tickers(sorted_df)


# Drawdown and Risk Strategies
class MaximumDrawdownStrategy(BaseStrategy):
    def run(self):
        sorted_df = self.df.nsmallest(self.top_n, 'MaximumDrawdown')
        return self.get_tickers(sorted_df)


class ValueAtRiskStrategy(BaseStrategy):
    def run(self):
        sorted_df = self.df.nsmallest(self.top_n, 'ValueAtRisk')
        return self.get_tickers(sorted_df)


class ConditionalValueAtRiskStrategy(BaseStrategy):
    def run(self):
        sorted_df = self.df.nsmallest(self.top_n, 'ConditionalValueAtRisk')
        return self.get_tickers(sorted_df)


# Return-Based Strategies
class AnnualizedReturnStrategy(BaseStrategy):
    def run(self):
        sorted_df = self.df.nlargest(self.top_n, 'AnnualizedReturn')
        return self.get_tickers(sorted_df)


class CumulativeReturnStrategy(BaseStrategy):
    def run(self):
        sorted_df = self.df.nlargest(self.top_n, 'CumulativeReturn')
        return self.get_tickers(sorted_df)


class AverageDailyReturnStrategy(BaseStrategy):
    def run(self):
        sorted_df = self.df.nlargest(self.top_n, 'AverageDailyReturn')
        return self.get_tickers(sorted_df)


# Volatility and Risk Strategies
class VolatilityStrategy(BaseStrategy):
    def run(self):
        sorted_df = self.df.nsmallest(self.top_n, 'Volatility')
        return self.get_tickers(sorted_df)


class UlcerIndexStrategy(BaseStrategy):
    def run(self):
        sorted_df = self.df.nsmallest(self.top_n, 'UlcerIndex')
        return self.get_tickers(sorted_df)


# Win/Loss Ratio Strategies
class WinRateStrategy(BaseStrategy):
    def run(self):
        sorted_df = self.df.nlargest(self.top_n, 'WinRate')
        return self.get_tickers(sorted_df)


class LossRateStrategy(BaseStrategy):
    def run(self):
        sorted_df = self.df.nsmallest(self.top_n, 'LossRate')
        return self.get_tickers(sorted_df)


class ProfitFactorStrategy(BaseStrategy):
    def run(self):
        sorted_df = self.df.nlargest(self.top_n, 'ProfitFactor')
        return self.get_tickers(sorted_df)


class GainToPainRatioStrategy(BaseStrategy):
    def run(self):
        sorted_df = self.df.nlargest(self.top_n, 'GainToPainRatio')
        return self.get_tickers(sorted_df)

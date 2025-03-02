from src.strategies.base_strategy import BaseStrategy


class OnBalanceVolumeStrategy(BaseStrategy):
    def run(self):
        self.df = self.df[self.df['OnBalanceVolume'] > self.df['OnBalanceVolume'].shift(1)]
        sorted_df = self.df.nlargest(self.top_n, 'OnBalanceVolume')
        return self.get_tickers(sorted_df)


class ChaikinMoneyFlowStrategy(BaseStrategy):
    def run(self):
        self.df = self.df[self.df['ChaikinMoneyFlow'] > 0]
        sorted_df = self.df.nlargest(self.top_n, 'ChaikinMoneyFlow')
        return self.get_tickers(sorted_df)


class AccumulationDistributionLineStrategy(BaseStrategy):
    def run(self):
        self.df = self.df[self.df['AccumulationDistributionLine'] > self.df['AccumulationDistributionLine'].shift(1)]
        sorted_df = self.df.nlargest(self.top_n, 'AccumulationDistributionLine')
        return self.get_tickers(sorted_df)


class MoneyFlowIndexStrategy(BaseStrategy):
    def run(self):
        self.df = self.df[self.df['MoneyFlowIndex'] > 80]
        sorted_df = self.df.nlargest(self.top_n, 'MoneyFlowIndex')
        return self.get_tickers(sorted_df)


class VolumeRateOfChangeStrategy(BaseStrategy):
    def run(self):
        sorted_df = self.df.nlargest(self.top_n, 'VolumeRateOfChange')
        return self.get_tickers(sorted_df)

from src.strategies.base_strategy import BaseStrategy


class RSIMomentumStrategy(BaseStrategy):
    def run(self):
        self.df = self.df[self.df['RelativeStrengthIndex'] < 30]
        sorted_df = self.df.nsmallest(self.top_n, 'RelativeStrengthIndex')
        return self.get_tickers(sorted_df)


class PriceRateOfChangeMomentumStrategy(BaseStrategy):
    def run(self):
        sorted_df = self.df.nlargest(self.top_n, 'PriceRateOfChange')
        return self.get_tickers(sorted_df)


class MACDCrossoverStrategy(BaseStrategy):
    def run(self):
        self.df = self.df[(self.df['MovingAverageConvergenceDivergence']['MACD'] > 0) &
                          (self.df['MovingAverageConvergenceDivergence']['MACD'] >
                           self.df['MovingAverageConvergenceDivergence']['Signal Line'])]
        sorted_df = self.df.nlargest(self.top_n, 'MovingAverageConvergenceDivergence.MACD')
        return self.get_tickers(sorted_df)


class WilliamsRMomentumStrategy(BaseStrategy):
    def run(self):
        self.df = self.df[self.df['WilliamsR'] < -80]
        sorted_df = self.df.nsmallest(self.top_n, 'WilliamsR')
        return self.get_tickers(sorted_df)


class CCIMomentumStrategy(BaseStrategy):
    def run(self):
        self.df = self.df[self.df['CommodityChannelIndex'] < -100]
        sorted_df = self.df.nsmallest(self.top_n, 'CommodityChannelIndex')
        return self.get_tickers(sorted_df)


class ForceIndexMomentumStrategy(BaseStrategy):
    def run(self):
        sorted_df = self.df.nlargest(self.top_n, 'ForceIndex')
        return self.get_tickers(sorted_df)


class EaseOfMovementMomentumStrategy(BaseStrategy):
    def run(self):
        sorted_df = self.df.nlargest(self.top_n, 'EaseOfMovement')
        return self.get_tickers(sorted_df)


class MoneyFlowIndexMomentumStrategy(BaseStrategy):
    def run(self):
        self.df = self.df[self.df['MoneyFlowIndex'] < 20]
        sorted_df = self.df.nsmallest(self.top_n, 'MoneyFlowIndex')
        return self.get_tickers(sorted_df)


class VortexIndicatorMomentumStrategy(BaseStrategy):
    def run(self):
        self.df['VortexPositive'] = self.df['VortexIndicator'].apply(lambda x: x[1])
        sorted_df = self.df.nlargest(self.top_n, 'VortexPositive')
        return self.get_tickers(sorted_df)


class GainToPainRatioMomentumStrategy(BaseStrategy):
    def run(self):
        sorted_df = self.df.nlargest(self.top_n, 'GainToPainRatio')
        return self.get_tickers(sorted_df)

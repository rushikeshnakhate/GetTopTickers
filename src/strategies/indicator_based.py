# Momentum Strategies
from src.strategies.base_strategy import BaseStrategy


class RSIMomentumStrategy(BaseStrategy):
    def run(self):
        self.df = self.df[self.df['RelativeStrengthIndex'] < 30]  # Oversold
        sorted_df = self.df.nsmallest(self.top_n, 'RelativeStrengthIndex')
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
        self.df = self.df[self.df['WilliamsR'] < -80]  # Oversold
        sorted_df = self.df.nsmallest(self.top_n, 'WilliamsR')
        return self.get_tickers(sorted_df)


class CCIMomentumStrategy(BaseStrategy):
    def run(self):
        self.df = self.df[self.df['CommodityChannelIndex'] < -100]  # Oversold
        sorted_df = self.df.nsmallest(self.top_n, 'CommodityChannelIndex')
        return self.get_tickers(sorted_df)


class PriceRateOfChangeMomentumStrategy(BaseStrategy):
    def run(self):
        sorted_df = self.df.nlargest(self.top_n, 'PriceRateOfChange')
        return self.get_tickers(sorted_df)


# Mean Reversion Strategies
class BollingerBandsMeanReversionStrategy(BaseStrategy):
    def run(self):
        self.df['BollingerLower'] = self.df['BollingerBands'].apply(lambda x: x['Lower Band'])
        self.df = self.df[self.df['Close'] <= self.df['BollingerLower']]
        sorted_df = self.df.nsmallest(self.top_n, 'BollingerLower')
        return self.get_tickers(sorted_df)


class KeltnerChannelsMeanReversionStrategy(BaseStrategy):
    def run(self):
        self.df['KeltnerLower'] = self.df['KeltnerChannel'].apply(lambda x: x[1])
        self.df = self.df[self.df['Close'] <= self.df['KeltnerLower']]
        sorted_df = self.df.nsmallest(self.top_n, 'KeltnerLower')
        return self.get_tickers(sorted_df)


class DonchianChannelMeanReversionStrategy(BaseStrategy):
    def run(self):
        self.df['DonchianLower'] = self.df['DonchianChannel'].apply(lambda x: x['Lower Band'])
        self.df = self.df[self.df['Close'] <= self.df['DonchianLower']]
        sorted_df = self.df.nsmallest(self.top_n, 'DonchianLower')
        return self.get_tickers(sorted_df)


# Volume-Based Strategies
class ChaikinMoneyFlowStrategy(BaseStrategy):
    def run(self):
        self.df = self.df[self.df['ChaikinMoneyFlow'] > 0]
        sorted_df = self.df.nlargest(self.top_n, 'ChaikinMoneyFlow')
        return self.get_tickers(sorted_df)


class OnBalanceVolumeStrategy(BaseStrategy):
    def run(self):
        self.df = self.df[self.df['OnBalanceVolume'] > self.df['OnBalanceVolume'].shift(1)]
        sorted_df = self.df.nlargest(self.top_n, 'OnBalanceVolume')
        return self.get_tickers(sorted_df)


# Volatility Strategies
class BollingerBandsVolatilityStrategy(BaseStrategy):
    def run(self):
        self.df['BollingerWidth'] = self.df['BollingerBands'].apply(lambda x: x['Upper Band'] - x['Lower Band'])
        sorted_df = self.df.nsmallest(self.top_n, 'BollingerWidth')
        return self.get_tickers(sorted_df)


class KeltnerChannelsVolatilityStrategy(BaseStrategy):
    def run(self):
        self.df['KeltnerWidth'] = self.df['KeltnerChannel'].apply(lambda x: x[0] - x[1])
        sorted_df = self.df.nsmallest(self.top_n, 'KeltnerWidth')
        return self.get_tickers(sorted_df)


# Trend-Following Strategies
class MovingAverageTrendStrategy(BaseStrategy):
    def run(self):
        self.df = self.df[self.df['Close'] > self.df['MovingAverage']]
        sorted_df = self.df.nlargest(self.top_n, 'MovingAverage')
        return self.get_tickers(sorted_df)


class ExponentialMovingAverageTrendStrategy(BaseStrategy):
    def run(self):
        self.df = self.df[self.df['Close'] > self.df['ExponentialMovingAverage']]
        sorted_df = self.df.nlargest(self.top_n, 'ExponentialMovingAverage')
        return self.get_tickers(sorted_df)


class AroonTrendStrategy(BaseStrategy):
    def run(self):
        self.df['AroonUp'] = self.df['Aroon'].apply(lambda x: x['Aroon Up'])
        sorted_df = self.df.nlargest(self.top_n, 'AroonUp')
        return self.get_tickers(sorted_df)

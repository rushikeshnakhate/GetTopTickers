from src.strategies.base_strategy import BaseStrategy


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


class RSIMeanReversionStrategy(BaseStrategy):
    def run(self):
        self.df = self.df[self.df['RelativeStrengthIndex'] > 70]
        sorted_df = self.df.nlargest(self.top_n, 'RelativeStrengthIndex')
        return self.get_tickers(sorted_df)


class WilliamsRMeanReversionStrategy(BaseStrategy):
    def run(self):
        self.df = self.df[self.df['WilliamsR'] > -20]
        sorted_df = self.df.nlargest(self.top_n, 'WilliamsR')
        return self.get_tickers(sorted_df)


class CCIMeanReversionStrategy(BaseStrategy):
    def run(self):
        self.df = self.df[self.df['CommodityChannelIndex'] > 100]
        sorted_df = self.df.nlargest(self.top_n, 'CommodityChannelIndex')
        return self.get_tickers(sorted_df)


class MovingAverageMeanReversionStrategy(BaseStrategy):
    def run(self):
        self.df = self.df[self.df['Close'] < self.df['MovingAverage']]
        sorted_df = self.df.nsmallest(self.top_n, 'MovingAverage')
        return self.get_tickers(sorted_df)


class AroonMeanReversionStrategy(BaseStrategy):
    def run(self):
        self.df['AroonDown'] = self.df['Aroon'].apply(lambda x: x['Aroon Down'])
        self.df = self.df[self.df['AroonDown'] > 70]
        sorted_df = self.df.nlargest(self.top_n, 'AroonDown')
        return self.get_tickers(sorted_df)


class ChaikinMoneyFlowMeanReversionStrategy(BaseStrategy):
    def run(self):
        self.df = self.df[self.df['ChaikinMoneyFlow'] < 0]
        sorted_df = self.df.nsmallest(self.top_n, 'ChaikinMoneyFlow')
        return self.get_tickers(sorted_df)


class OnBalanceVolumeMeanReversionStrategy(BaseStrategy):
    def run(self):
        self.df = self.df[self.df['OnBalanceVolume'] < self.df['OnBalanceVolume'].shift(1)]
        sorted_df = self.df.nsmallest(self.top_n, 'OnBalanceVolume')
        return self.get_tickers(sorted_df)

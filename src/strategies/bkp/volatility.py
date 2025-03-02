from src.strategies.base_strategy import BaseStrategy


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


class DonchianChannelVolatilityStrategy(BaseStrategy):
    def run(self):
        self.df['DonchianWidth'] = self.df['DonchianChannel'].apply(lambda x: x['Upper Band'] - x['Lower Band'])
        sorted_df = self.df.nsmallest(self.top_n, 'DonchianWidth')
        return self.get_tickers(sorted_df)


class ATRVolatilityStrategy(BaseStrategy):
    def run(self):
        sorted_df = self.df.nlargest(self.top_n, 'AverageTrueRange')
        return self.get_tickers(sorted_df)


class VolatilityBreakoutStrategy(BaseStrategy):
    def run(self):
        self.df['BollingerUpper'] = self.df['BollingerBands'].apply(lambda x: x['Upper Band'])
        self.df = self.df[self.df['Close'] > self.df['BollingerUpper']]
        sorted_df = self.df.nlargest(self.top_n, 'BollingerUpper')
        return self.get_tickers(sorted_df)

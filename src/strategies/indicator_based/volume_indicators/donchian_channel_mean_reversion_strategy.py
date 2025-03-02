from src.strategies.base_strategy import BaseStrategy


class DonchianChannelMeanReversionStrategy(BaseStrategy):
    def run(self):
        self.df['DonchianLower'] = self.df['DonchianChannel'].apply(lambda x: x['Lower Band'])
        self.df = self.df[self.df['Close'] <= self.df['DonchianLower']]
        sorted_df = self.df.nsmallest(self.top_n, 'DonchianLower')
        return self.get_tickers(sorted_df)

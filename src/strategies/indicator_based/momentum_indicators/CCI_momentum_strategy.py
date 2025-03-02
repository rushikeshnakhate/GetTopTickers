from src.strategies.base_strategy import BaseStrategy


class CCIMomentumStrategy(BaseStrategy):
    def run(self):
        self.df = self.df[self.df['CommodityChannelIndex'] < -100]  # Oversold
        sorted_df = self.df.nsmallest(self.top_n, 'CommodityChannelIndex')
        return self.get_tickers(sorted_df)

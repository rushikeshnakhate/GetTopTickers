from src.strategies.base_strategy import BaseStrategy


class CumulativeReturnStrategy(BaseStrategy):
    def run(self):
        sorted_df = self.df.nlargest(self.top_n, 'CumulativeReturn')
        return self.get_tickers(sorted_df)

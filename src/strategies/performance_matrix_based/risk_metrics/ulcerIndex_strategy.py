from src.strategies.base_strategy import BaseStrategy


class UlcerIndexStrategy(BaseStrategy):
    def run(self):
        sorted_df = self.df.nsmallest(self.top_n, 'UlcerIndex')
        return self.get_tickers(sorted_df)

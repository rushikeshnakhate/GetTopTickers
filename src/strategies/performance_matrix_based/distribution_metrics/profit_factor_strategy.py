from src.strategies.base_strategy import BaseStrategy


class ProfitFactorStrategy(BaseStrategy):
    def run(self):
        sorted_df = self.df.nlargest(self.top_n, 'ProfitFactor')
        return self.get_tickers(sorted_df)

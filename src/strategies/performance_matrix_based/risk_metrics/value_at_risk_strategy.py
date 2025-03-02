from src.strategies.base_strategy import BaseStrategy


class ValueAtRiskStrategy(BaseStrategy):
    def run(self):
        sorted_df = self.df.nsmallest(self.top_n, 'ValueAtRisk')
        return self.get_tickers(sorted_df)

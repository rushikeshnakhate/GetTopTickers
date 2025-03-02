from src.strategies.base_strategy import BaseStrategy


class ConditionalValueAtRiskStrategy(BaseStrategy):
    def run(self):
        sorted_df = self.df.nsmallest(self.top_n, 'ConditionalValueAtRisk')
        return self.get_tickers(sorted_df)

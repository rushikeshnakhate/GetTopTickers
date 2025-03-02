from src.strategies.base_strategy import BaseStrategy


# Volatility and Risk Strategies
class VolatilityStrategy(BaseStrategy):
    def run(self):
        sorted_df = self.df.nsmallest(self.top_n, 'Volatility')
        return self.get_tickers(sorted_df)

from src.strategies.base_strategy import BaseStrategy


class MACDCrossoverStrategy(BaseStrategy):
    def run(self):
        self.df = self.df[(self.df['MovingAverageConvergenceDivergence']['MACD'] > 0) &
                          (self.df['MovingAverageConvergenceDivergence']['MACD'] >
                           self.df['MovingAverageConvergenceDivergence']['Signal Line'])]
        sorted_df = self.df.nlargest(self.top_n, 'MovingAverageConvergenceDivergence.MACD')
        return self.get_tickers(sorted_df)

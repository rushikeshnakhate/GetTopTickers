from src.strategies.base_strategy import BaseStrategy


class AroonTrendStrategy(BaseStrategy):
    def run(self):
        self.df['AroonUp'] = self.df['Aroon'].apply(lambda x: x['Aroon Up'])
        sorted_df = self.df.nlargest(self.top_n, 'AroonUp')
        return self.get_tickers(sorted_df)

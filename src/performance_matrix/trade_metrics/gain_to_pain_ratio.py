from src.performance_matrix.base_parameter import BasePerformanceMatrix
from src.performance_matrix.return_matrix.gain import Gain
from src.performance_matrix.return_matrix.loss import Loss


class GainToPainRatio(BasePerformanceMatrix):
    def calculate(self):
        gains = Gain(self.stock_data).calculate()
        losses = Loss(self.stock_data).calculate()
        return gains.sum() / abs(losses.sum())

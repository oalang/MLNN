import numpy as np

from mlnn.engine import MLNNEngine
from mlnn.optimizers import MLNNBacktracking


class MLNN:
    def __init__(self, d=None, mlnn_params=None, line_search_params=None, optimize_params=None):
        self.d = d
        self.mlnn_params = mlnn_params
        self.line_search_params = line_search_params
        self.optimize_params = optimize_params
        self.mlnn = None

    def fit(self, X, Y):
        B = X
        T = np.where(np.equal(Y.reshape(-1, 1), Y.reshape(1, -1)), 1, -1)
        N = np.sum(T == 1, axis=1, keepdims=True) - 1

        self.mlnn = MLNNEngine(B, T, N, mlnn_params=self.mlnn_params)
        optimizer = MLNNBacktracking(
            self.mlnn, d=self.d,
            line_search_params=self.line_search_params,
            optimize_params=self.optimize_params,
        )
        optimizer.minimize()
        optimizer.print_result()
        optimizer.minimize(method='alternating')
        optimizer.print_result()

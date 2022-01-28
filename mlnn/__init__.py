import numpy as np

from optimize import MLNNSteepestDescent


class MLNN:
    def __init__(self, mlnn_params=None, line_search_params=None, optimize_params=None):
        self.mlnn_params = mlnn_params
        self.line_search_params = line_search_params
        self.optimize_params = optimize_params
        self.mlnn_optimizer = None

    def fit(self, X, Y):
        B = X
        T = np.where(np.equal(Y.reshape(-1, 1), Y.reshape(1, -1)), 1, -1)
        N = np.sum(T == 1, axis=1, keepdims=True) - 1
        self.mlnn_optimizer = MLNNSteepestDescent(
            B, T, N,
            mlnn_params=self.mlnn_params,
            line_search_params=self.line_search_params,
            optimize_params=self.optimize_params,
        )
        self.mlnn_optimizer.optimize()
        self.mlnn_optimizer.print_result()

import numpy as np

from loss import SmoothReLU
from mlnn.engine import MLNNEngine
from mlnn.callback import MLNNCallback
from mlnn.optimize import MLNNSteepestDescent, MLNNBFGS


class MLNN:
    def __init__(self, n_components=None, *, init='pca', max_iter=50, tol=1e-05, callback=None, verbose=0,
                 random_state=None, solver='steepest_fixed_backoff'):
        self.n_components = n_components
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.callback = callback
        self.verbose = verbose
        self.random_state = random_state
        self.solver = solver

        self.mlnn_params = None
        self.optimize_params = None
        self.line_search_params = None
        self.transformer = None
        self.epsilon = None

        self.configure_params()

    def configure_params(self):
        loss = SmoothReLU(.5)

        self.mlnn_params = {
            'r': 1,
            's': 0,
            'l': 1,
            'q': 1,
            'inner_loss': loss,
            'outer_loss': loss,
            'kernel': 'linear',
            'a_mode': 'decomposed',
            'e_mode': 'single',
            'keep_a_psd': False,
            'keep_a_centered': False,
            'keep_e_positive': False,
        }

        self.optimize_params = {
            'optimize_method': 'fixed',
            'initialization': self.init,
            'min_delta_F': self.tol,
            'max_steps': self.max_iter,
            'max_time': np.inf,
            'fixed_arguments': 'AE',
            'max_arg_steps': 5,
            'maxcor': None,
            'gtol': None,
            'eps': None,
            'maxfun': None,
            'iprint': None,
            'finite_diff_rel_step': None,
            'optimize_verbose': False,
        }

        self.line_search_params = {
            'line_search_method': 'backtracking',
            'use_prev_f': False,
            'alpha_0': 1e-3,
            'armijo': 1e-06,
            'wolfe': 0.9,
            'rho_lo': 0.1,
            'rho_hi': 0.9,
            'max_ls_iterations': 20,
            'line_search_verbose': False,
        }

        if 'fixed' in self.solver:
            self.optimize_params['optimize_method'] = 'fixed'
        elif 'alternating' in self.solver:
            self.optimize_params['optimize_method'] = 'alternating'

        if 'backtracking' in self.solver:
            self.line_search_params['line_search_method'] = 'backtracking'
        elif 'strong_wolfe' in self.solver:
            self.optimize_params['optimize_method'] = 'strong_wolfe'

    def fit(self, X, y):
        B = X
        T = np.where(np.equal(y.reshape(-1, 1), y.reshape(1, -1)), 1, -1)
        N = np.sum(T == 1, axis=1, keepdims=True) - 1

        mlnn = MLNNEngine(B, T, N, mlnn_params=self.mlnn_params)

        callback = None
        if self.verbose >= 2:
            callback = MLNNCallback(print_stats=True)

        optimizer = None
        if 'steepest' in self.solver:
            optimizer = MLNNSteepestDescent(mlnn, callback, n_components=self.n_components,
                                            line_search_params=self.line_search_params,
                                            optimize_params=self.optimize_params)
        elif 'bfgs' in self.solver:
            optimizer = MLNNBFGS(mlnn, callback, n_components=self.n_components,
                                 line_search_params=self.line_search_params,
                                 optimize_params=self.optimize_params)

        optimizer.minimize()

        if self.verbose >= 1:
            optimizer.report()

        self.transformer = LinearTransformer(optimizer.mlnn.A)
        self.epsilon = np.mean(optimizer.mlnn.E)

        return self

    def transform(self, X):
        return self.transformer.transform(X)

    def fit_transform(self, X, y):
        self.fit(X, y)

        return self.transform(X)


class Transformer:
    def __init__(self, L):
        self.L = L


class LinearTransformer(Transformer):
    def __init__(self, L):
        super().__init__(L)

    def transform(self, X):
        return X @ self.L.T


class RBFTransformer(Transformer):
    def __init__(self, L, Z, sigma2):
        super().__init__(L)

        self.Z = Z
        self.sigma2 = sigma2

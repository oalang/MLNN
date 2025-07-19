import numpy as np

from loss import SmoothReLU1
from mlnn.engine import MLNNEngine
from mlnn.callback import MLNNCallback
from mlnn.optimize import MLNNSteepestDescent, MLNNBFGS


class MLNN:
    def __init__(
            self,
            n_components=None,
            *,
            kernel='linear',
            regularization='auto',
            init='pca',
            max_iter=50,
            max_time=np.inf,
            tol=1e-05,
            callback=None,
            random_state=None,
            solver='steepest_fixed_backtracking',
            backend='numpy',
            mlnn_alpha=1,
            mlnn_beta=0,
            mlnn_gamma=1,
            mlnn_delta=1,
            mlnn_loss='smooth_relu2',
            mlnn_leaky_slope=1e-2,
            mlnn_matrix_mode='decomposed',
            mlnn_matrix_psd=False,
            mlnn_matrix_centered=False,
            mlnn_matrix_init=None,
            mlnn_epsilon_mode='single',
            mlnn_epsilon_positive=False,
            mlnn_epsilon_init=None,
            opt_arguments='AE',
            opt_max_arg_steps=5,
            opt_maxcor=None,
            opt_gtol=None,
            opt_eps=None,
            opt_maxfun=None,
            opt_finite_diff_rel_step=None,
            ls_max_iterations=20,
            ls_use_prev_f=False,
            ls_alpha_0=1e-3,
            ls_armijo=1e-06,
            ls_wolfe=0.9,
            ls_rho_lo=0.1,
            ls_rho_hi=0.9,
            collect_stats=False,
            animate=False,
            verbose=0,
    ):
        self.n_components = n_components
        #self.kernel = kernel
        #self.regularization = regularization
        self.init = init
        self.max_iter = max_iter
        self.max_time = max_time
        self.tol = tol
        self.callback_fun = callback
        self.random_state = random_state
        self.solver = solver
        #self.backend = backend
        self.mlnn_alpha = mlnn_alpha
        self.mlnn_beta = mlnn_beta
        self.mlnn_gamma = mlnn_gamma
        self.mlnn_delta = mlnn_delta
        #self.mlnn_loss = mlnn_loss
        #self.mlnn_leaky_slope = mlnn_leaky_slope
        self.mlnn_matrix_mode = mlnn_matrix_mode
        self.mlnn_matrix_psd = mlnn_matrix_psd
        self.mlnn_matrix_centered = mlnn_matrix_centered
        self.mlnn_matrix_init = mlnn_matrix_init
        self.mlnn_epsilon_mode = mlnn_epsilon_mode
        self.mlnn_epsilon_positive = mlnn_epsilon_positive
        self.mlnn_epsilon_init = mlnn_epsilon_init
        self.opt_arguments = opt_arguments
        self.opt_max_arg_steps = opt_max_arg_steps
        self.opt_maxcor = opt_maxcor
        self.opt_gtol = opt_gtol
        self.opt_eps = opt_eps
        self.opt_maxfun = opt_maxfun
        self.opt_finite_diff_rel_step = opt_finite_diff_rel_step
        self.ls_max_iterations = ls_max_iterations
        self.ls_use_prev_f = ls_use_prev_f
        self.ls_alpha_0 = ls_alpha_0
        self.ls_armijo = ls_armijo
        self.ls_wolfe = ls_wolfe
        self.ls_rho_lo = ls_rho_lo
        self.ls_rho_hi = ls_rho_hi
        #self.collect_stats = collect_stats
        #self.animate = animate
        self.verbose = verbose

        self.mlnn_params = None
        self.optimize_params = None
        self.line_search_params = None
        self.transformer = None
        self.epsilon = None

        self.configure_params()

    def configure_params(self):
        loss = SmoothReLU1(.5)

        self.mlnn_params = {
            'r': self.mlnn_alpha,
            's': self.mlnn_beta,
            'l': self.mlnn_gamma,
            'q': self.mlnn_delta,
            'inner_loss': loss,
            'outer_loss': loss,
            'kernel': 'linear',
            'a_mode': self.mlnn_matrix_mode,
            'e_mode': self.mlnn_epsilon_mode,
            'keep_a_psd': self.mlnn_matrix_psd,
            'keep_a_centered': self.mlnn_matrix_centered,
            'keep_e_positive': self.mlnn_epsilon_positive,
            'reduce_derivative_matrix': False,
            'check_array_equal': False,
        }

        self.optimize_params = {
            'initialization': self.init,
            'min_delta_F': self.tol,
            'max_steps': self.max_iter,
            'max_time': np.inf,
            'optimize_method': 'fixed',
            'opt_arguments': self.opt_arguments,
            'max_arg_steps': self.opt_max_arg_steps,
            'maxcor': self.opt_maxcor,
            'gtol': self.opt_gtol,
            'eps': self.opt_eps,
            'maxfun': self.opt_maxfun,
            'finite_diff_rel_step': self.opt_finite_diff_rel_step,
        }

        self.line_search_params = {
            'max_ls_iterations': self.ls_max_iterations,
            'line_search_method': 'backtracking',
            'alpha_0': self.ls_alpha_0,
            'armijo': self.ls_armijo,
            'rho_lo': self.ls_rho_lo,
            'rho_hi': self.ls_rho_hi,
            'wolfe': self.ls_wolfe,
            'use_prev_f': self.ls_use_prev_f,
        }

        if 'fixed' in self.solver:
            self.optimize_params['optimize_method'] = 'fixed'
        elif 'alternating' in self.solver:
            self.optimize_params['optimize_method'] = 'alternating'

        if 'backtracking' in self.solver:
            self.line_search_params['line_search_method'] = 'backtracking'
        elif 'strong_wolfe' in self.solver:
            self.line_search_params['line_search_method'] = 'strong_wolfe'

    def fit(self, X, y):
        mlnn = MLNNEngine(X, y, mlnn_params=self.mlnn_params)

        callback = MLNNCallback()
        if self.verbose >= 2:
            callback.print_stats = True
        callback.callback_fun = self.callback_fun

        optimizer = None
        if 'steepest' in self.solver:
            optimizer = MLNNSteepestDescent(
                mlnn, callback, self.mlnn_matrix_init, self.mlnn_epsilon_init,
                self.n_components, self.optimize_params, self.line_search_params)
        elif 'bfgs' in self.solver:
            optimizer = MLNNBFGS(
                mlnn, callback, self.mlnn_matrix_init, self.mlnn_epsilon_init,
                self.n_components, self.optimize_params, self.line_search_params)

        optimizer.minimize()

        if self.verbose >= 1:
            optimizer.report()

        self.transformer = LinearTransformation(mlnn.get_transformation_matrix(self.n_components))
        self.epsilon = np.mean(mlnn.E)

        return self

    def transform(self, X):
        return self.transformer.transform(X)

    def fit_transform(self, X, y):
        self.fit(X, y)

        return self.transform(X)


class Transformation:
    def __init__(self, L):
        self.L = L


class LinearTransformation(Transformation):
    def __init__(self, L):
        super().__init__(L)

    def transform(self, X):
        return X @ self.L.T


class RBFTransformation(Transformation):
    def __init__(self, L, Z, sigma2):
        super().__init__(L)

        self.Z = Z
        self.sigma2 = sigma2

    def transform(self, X):
        D = 2 * X @ self.Z.T - np.sum(X ** 2, axis=1).reshape(-1, 1) - np.sum(self.Z ** 2, axis=1).reshape(1, -1)
        B = np.exp(D / (2 * self.sigma2))

        return B @ self.L.T

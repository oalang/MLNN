from functools import partial

import numpy as np

from sklearn.cluster import KMeans
from scipy.optimize import bracket, minimize_scalar

from mlnn.activation import get_activation_function
from mlnn.engine import MLNNEngine
from mlnn.callback import MLNNCallback
from mlnn.optimize import MLNNSteepestDescent, MLNNBFGS


def pairwise_squared_distance(X, L=None):
    if L is None:
        P = X @ X.T
        C = P.diagonal()
        return np.add.outer(C, C) - 2 * P
    else:
        P = X @ L.T
        C = np.einsum('ij,ij->i', X, X)
        D = np.einsum('ij,ij->i', L, L)
        return np.add.outer(C, D) - 2 * P


def generate_landmarks(X, n_landmarks, method='kmeans'):
    if method == 'kmeans':
        kmeans = KMeans(n_landmarks)
        kmeans.fit(X)
        return kmeans.cluster_centers_
    elif method == 'random':
        return X[np.random.choice(X.shape[0], size=n_landmarks, replace=False)]


def negative_kernel_entropy(sigma2, squared_distances, n_bins=10):
    K = np.exp(-squared_distances / (2 * sigma2))
    H, _ = np.histogram(K, bins=n_bins, range=(0, 1))
    H = H / H.sum()
    return np.sum(H * np.log(H + 1e-10))


def maximize_kernel_entropy(squared_distances, n_bins=10):
    fun = partial(negative_kernel_entropy, squared_distances=squared_distances, n_bins=n_bins)
    xa_0 = 1e-10
    xb_0 = np.mean(squared_distances)

    try:
        xa, xb, xc, _, _, _, funcalls = bracket(fun, xa_0, xb_0)
    except Exception as e:
        print("Kernel entropy maximization: Failed to bracket local maximum.")
        print(f"Exception details: {e}")
    else:
        result = minimize_scalar(fun, bracket=(xa, xb, xc))
        if result.success:
            funcalls += result.nfev
            return result.x, -result.fun, funcalls
        else:
            print("Kernel entropy maximization: Optimization failed.")

    print("Setting sigma2 to the mean squared distance.")
    sigma2 = xb_0
    kernel_entropy = -negative_kernel_entropy(sigma2, squared_distances, n_bins)
    return sigma2, kernel_entropy, None


class LinearTransformation:
    def __init__(self, B):
        self.B = B

    def transform(self, X):
        return X @ self.B.T


class RBFTransformation:
    def __init__(self, B, L, sigma2):
        self.B = B
        self.L = L
        self.sigma2 = sigma2

    def transform(self, X):
        D = pairwise_squared_distance(X, self.L)
        K = np.exp(-D / (2 * self.sigma2))

        return K @ self.B.T


class MLNN:
    def __init__(
            self,
            n_components=None,
            *,
            kernel=None,
            rbf_sigma2='auto',
            regularization='auto',
            landmark_selection=None,
            n_landmarks=100,
            init='auto',
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
            mlnn_inner_loss='smooth_relu2',
            mlnn_outer_loss='smooth_relu2',
            mlnn_inner_offset=1,
            mlnn_outer_offset=1,
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
            config=None
    ):
        self.n_components = n_components
        self.kernel = kernel
        self.rbf_sigma2 = rbf_sigma2
        self.regularization = regularization
        self.n_landmarks = n_landmarks
        self.landmark_selection = landmark_selection
        self.init = init
        self.max_iter = max_iter
        self.max_time = max_time
        self.tol = tol
        self.random_state = random_state
        self.solver = solver
        self.mlnn_alpha = mlnn_alpha
        self.mlnn_beta = mlnn_beta
        self.mlnn_gamma = mlnn_gamma
        self.mlnn_delta = mlnn_delta
        self.mlnn_inner_loss = mlnn_inner_loss
        self.mlnn_outer_loss = mlnn_outer_loss
        self.mlnn_inner_offset = mlnn_inner_offset
        self.mlnn_outer_offset = mlnn_outer_offset
        self.mlnn_leaky_slope = mlnn_leaky_slope
        self.mlnn_matrix_mode = mlnn_matrix_mode
        self.mlnn_matrix_psd = mlnn_matrix_psd
        self.mlnn_matrix_centered = mlnn_matrix_centered
        self.mlnn_epsilon_mode = mlnn_epsilon_mode
        self.mlnn_epsilon_positive = mlnn_epsilon_positive
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

        if config is not None:
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        self.config = self.__dict__.copy()

        self.backend = backend
        self.callback_fun = callback
        self.mlnn_matrix_init = mlnn_matrix_init
        self.mlnn_epsilon_init = mlnn_epsilon_init
        self.collect_stats = collect_stats
        self.animate = animate
        self.verbose = verbose

        self.mlnn_params = None
        self.optimize_params = None
        self.line_search_params = None
        self.stats = None
        self.animation = None
        self.transformer = None
        self.epsilon = None

        self.configure_params()

    def configure_params(self):
        if self.kernel is None:
            x_mode = 'raw'
        else:
            x_mode = 'kernel'

        inner_loss = get_activation_function(
            self.mlnn_inner_loss,
            self.mlnn_inner_offset,
            self.mlnn_leaky_slope
        )

        outer_loss = get_activation_function(
            self.mlnn_outer_loss,
            self.mlnn_outer_offset,
            self.mlnn_leaky_slope
        )

        self.mlnn_params = {
            'r': self.mlnn_alpha,
            's': self.mlnn_beta,
            'l': self.mlnn_gamma,
            'q': self.mlnn_delta,
            'inner_loss': inner_loss,
            'outer_loss': outer_loss,
            'x_mode': x_mode,
            'a_mode': self.mlnn_matrix_mode,
            'e_mode': self.mlnn_epsilon_mode,
            'keep_a_psd': self.mlnn_matrix_psd,
            'keep_a_centered': self.mlnn_matrix_centered,
            'keep_e_positive': self.mlnn_epsilon_positive,
            'reduce_derivative_matrix': False,
            'check_array_equal': False,
        }

        initialization = None if self.init == 'auto' or isinstance(self.init, int) else self.init

        if 'fixed' in self.solver:
            optimize_method = 'fixed'
        elif 'alternating' in self.solver:
            optimize_method = 'alternating'

        self.optimize_params = {
            'initialization': initialization,
            'random_state': self.random_state,
            'min_delta_F': self.tol,
            'max_steps': self.max_iter,
            'max_time': np.inf,
            'optimize_method': optimize_method,
            'opt_arguments': self.opt_arguments,
            'max_arg_steps': self.opt_max_arg_steps,
            'maxcor': self.opt_maxcor,
            'gtol': self.opt_gtol,
            'eps': self.opt_eps,
            'maxfun': self.opt_maxfun,
            'finite_diff_rel_step': self.opt_finite_diff_rel_step,
        }

        if 'backtracking' in self.solver:
            line_search_method = 'backtracking'
        elif 'strong_wolfe' in self.solver:
            line_search_method = 'strong_wolfe'

        self.line_search_params = {
            'max_ls_iterations': self.ls_max_iterations,
            'line_search_method': line_search_method,
            'alpha_0': self.ls_alpha_0,
            'armijo': self.ls_armijo,
            'rho_lo': self.ls_rho_lo,
            'rho_hi': self.ls_rho_hi,
            'wolfe': self.ls_wolfe,
            'use_prev_f': self.ls_use_prev_f,
        }

    def fit(self, X, y):
        Y = y.reshape(-1, 1)
        Z = None

        if self.kernel is None:
            K = X
        else:
            if self.landmark_selection is None:
                L = None
            else:
                L = generate_landmarks(X, self.n_landmarks, method=self.landmark_selection)

            if self.kernel == 'rbf':
                D = pairwise_squared_distance(X, L)

                if self.rbf_sigma2 == 'auto':
                    sigma2, _, _ = maximize_kernel_entropy(D)
                else:
                    sigma2 = self.rbf_sigma2

                K = np.exp(-D / (2 * sigma2))

                if self.regularization == 'auto' and L is not None:
                    Z = np.exp(-pairwise_squared_distance(L) / (2 * sigma2))
            elif self.kernel == 'linear':
                if L is None:
                    K = X @ X.T
                else:
                    K = X @ L.T

                if self.regularization == 'auto' and L is not None:
                    Z = L @ L.T

            if L is None:
                L = X

            if self.regularization == 'unweighted':
                self.mlnn_params['x_mode'] = 'raw'

        mlnn = MLNNEngine(K, Y, Z, self.mlnn_params)

        print_stats = self.verbose >= 2
        callback = MLNNCallback(print_stats, self.collect_stats, self.animate, self.callback_fun)

        if (
            self.mlnn_matrix_mode == 'decomposed'
            and isinstance(self.init, int)
            and self.mlnn_matrix_init is None
        ):
            init_mlnn_params = self.mlnn_params.copy()
            init_mlnn_params['a_mode'] = 'full'
            init_mlnn = MLNNEngine(K, Y, Z, init_mlnn_params)

            init_callback = MLNNCallback(print_stats)

            init_optimize_params = self.optimize_params.copy()
            init_optimize_params['max_steps'] = self.init
            init_optimize_params['max_time'] = np.inf
            if 'steepest' in self.solver:
                init_optimizer = MLNNSteepestDescent(
                    init_mlnn, init_callback,
                    optimize_params=init_optimize_params,
                    line_search_params=self.line_search_params)
            elif 'bfgs' in self.solver:
                init_optimizer = MLNNBFGS(
                    init_mlnn, init_callback,
                    optimize_params=init_optimize_params,
                    line_search_params=self.line_search_params)

            init_optimizer.minimize()

            self.mlnn_matrix_init = init_mlnn.get_transformation_matrix(self.n_components)
            if self.mlnn_epsilon_init is None:
                self.mlnn_epsilon_init = init_mlnn.E

        if 'steepest' in self.solver:
            optimizer = MLNNSteepestDescent(
                mlnn, callback, self.mlnn_matrix_init, self.mlnn_epsilon_init,
                self.n_components, self.optimize_params, self.line_search_params)
        elif 'bfgs' in self.solver:
            optimizer = MLNNBFGS(
                mlnn, callback, self.mlnn_matrix_init, self.mlnn_epsilon_init,
                self.n_components, self.optimize_params, self.line_search_params)

        optimizer.minimize()

        if self.collect_stats:
            self.stats = callback.stats
        if self.animate:
            self.animation = callback.ani
        if self.verbose >= 1:
            optimizer.report()

        B = mlnn.get_transformation_matrix(self.n_components)
        if self.kernel is None:
            self.transformer = LinearTransformation(B)
        elif self.kernel == 'rbf':
            self.transformer = RBFTransformation(B, L, sigma2)
        elif self.kernel == 'linear':
            self.transformer = LinearTransformation(B @ L)

        self.epsilon = np.mean(mlnn.E)

        return self

    def transform(self, X):
        return self.transformer.transform(X)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

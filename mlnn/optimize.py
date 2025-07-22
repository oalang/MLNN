import time
import warnings
import numpy as np
from scipy.optimize import line_search as scipy_line_search, minimize as scipy_minimize, Bounds

from mlnn.callback import MLNNCallback


class MLNNOptimizer:
    def __init__(self, mlnn, callback):
        self.mlnn = mlnn
        self.callback = callback

        self.initialization = None
        self.min_delta_F = 1e-06
        self.max_steps = 1000
        self.max_time = np.inf
        self.verbose = False
        self.max_ls_iterations = 20

        self.time_0 = None
        self.run_time = None
        self.steps = None
        self.termination = None
        self.delta_F = None

        self.A_0 = None
        self.E_0 = None
        self.F_0 = None

    @property
    def time(self):
        return time.perf_counter() - self.time_0

    def apply_params(self, params):
        for attr in params:
            if hasattr(self, attr):
                setattr(self, attr, params[attr])

    def initialize_args(self, A_0=None, E_0=None, n_components=None):
        if self.initialization is None:
            if self.mlnn.a_mode in ('full', 'diagonal'):
                self.initialization = 'zero'
            elif self.mlnn.a_mode == 'decomposed':
                self.initialization = 'pca'

        if self.mlnn.a_mode == 'full':
            assert (self.initialization in ('random', 'zero', 'identity', 'centered'))
        elif self.mlnn.a_mode == 'diagonal':
            assert (self.initialization in ('random', 'zero', 'identity'))
        elif self.mlnn.a_mode == 'decomposed':
            assert (self.initialization in ('random', 'pca'))

        if A_0 is None:
            self.A_0 = self.mlnn.compute_A_0(self.initialization, n_components)
        else:
            self.A_0 = A_0

        if self.mlnn.a_mode == 'full':
            assert self.A_0.shape[0] == self.mlnn.m
            assert np.array_equal(self.A_0, self.A_0.T)
        elif self.mlnn.a_mode == 'diagonal':
            assert self.A_0.shape[0] == self.mlnn.m
            assert self.A_0.shape[1] == 1
        elif self.mlnn.a_mode == 'decomposed':
            assert self.A_0.shape[1] == self.mlnn.m

        if E_0 is None:
            self.E_0 = self.mlnn.compute_E_0(self.initialization)
        else:
            self.E_0 = np.atleast_2d(E_0)

        if self.mlnn.e_mode == 'single':
            assert self.E_0.size == 1
        elif self.mlnn.e_mode == 'multiple':
            assert self.E_0.shape[0] == self.mlnn.n
            assert self.E_0.shape[1] == 1

    def initialize_optimizer(self):
        self.time_0 = time.perf_counter()
        self.run_time = None
        self.steps = 0
        self.termination = None

        self.mlnn.F_count = 0
        self.mlnn.dFdA_count = 0
        self.mlnn.dFdE_count = 0
        self.mlnn.eigh_count = 0

        self.mlnn.A = self.A_0
        self.mlnn.E = self.E_0
        self.F_0 = self.mlnn.F

    def report(self):
        if self.termination == 'max_ls_iterations':
            threshold = f" (max_ls_iterations = {self.max_ls_iterations:d})"
        elif self.termination == 'min_delta_F':
            threshold = f" (min_delta_F = {self.min_delta_F:e})"
        elif self.termination == 'max_steps':
            threshold = f" (max_steps = {self.max_steps:d})"
        elif self.termination == 'max_time':
            threshold = f" (max_time = {self.max_time:f})"
        else:
            threshold = ""

        print("")
        print(f"Termination: {self.termination}{threshold}")
        print(f"       F_0 = {self.F_0:f}")
        print(f"         F = {self.mlnn.F:f}")
        print(f"   delta_F = {self.delta_F:e}")
        print(f"     steps = {self.steps:d}")
        print(f"  run_time = {self.run_time:f} seconds")
        print("")
        print(f"   F function calls: {self.mlnn.F_count:d}")
        print(f"  dA function calls: {self.mlnn.dFdA_count:d}")
        print(f"  dE function calls: {self.mlnn.dFdE_count:d}")
        print(f"eigh function calls: {self.mlnn.eigh_count:d}")
        print("")


class MLNNSteepestDescent(MLNNOptimizer):
    def __init__(self, mlnn, callback=None, A_0=None, E_0=None, n_components=None,
                 optimize_params=None, line_search_params=None):
        super().__init__(mlnn, callback)

        self.optimize_method = 'fixed'
        self.opt_arguments = 'AE'
        self.max_arg_steps = 5

        self.line_search_method = 'backtracking'
        self.alpha_0 = 1e-06
        self.armijo = 1e-04
        self.rho_lo = 0.1
        self.rho_hi = 0.9
        self.wolfe = 0.9
        self.use_prev_f = False

        if optimize_params:
            self.apply_params(optimize_params)
        if line_search_params:
            self.apply_params(line_search_params)

        self.initialize_args(A_0, E_0, n_components)

        self.arguments = None
        self.phi = None
        self.alpha = None
        self.ls_iterations = None

    def backtracking_line_search(self, arguments, F_prev, A_prev, E_prev, dA, dE, phi, alpha):
        # Take a step in the direction of steepest descent.
        if dA is not None:
            self.mlnn.update_A(A_prev, dA, alpha)
        if dE is not None:
            self.mlnn.update_E(E_prev, dE, alpha)

        iterations = 1

        # If Armijo's condition for sufficient decrease has been satisfied, the search is complete.
        if self.mlnn.F <= F_prev + self.armijo * alpha * phi:
            self.arguments = arguments
            self.phi = phi
            self.alpha = alpha
            self.ls_iterations = iterations
            self.steps += 1
            self.termination = None
            return True

        # If the maximum number of ls_iterations have been performed, return without taking a step.
        if iterations == self.max_ls_iterations:
            self.mlnn.A = A_prev
            self.mlnn.E = E_prev
            self.mlnn.F = F_prev
            self.termination = 'max_ls_iterations'
            return False

        # Decrease alpha with a quadratic approximation of the minimizer, by interpolating F_prev, phi, and self.F.
        alpha_old = alpha
        alpha = - .5 * phi * alpha ** 2 / (self.mlnn.F - F_prev - phi * alpha)

        # Safeguard the interpolation by making sure alpha is not too small or too close to alpha_old.
        rho = alpha / alpha_old
        if rho < self.rho_lo or rho > self.rho_hi:
            rho = .5
            alpha = rho * alpha_old

        # Take a step in the direction of steepest descent.
        F_old = self.mlnn.F
        if dA is not None:
            self.mlnn.update_A(A_prev, dA, alpha)
        if dE is not None:
            self.mlnn.update_E(E_prev, dE, alpha)

        iterations += 1

        # If Armijo's condition for sufficient decrease has been satisfied, the search is complete.
        if self.mlnn.F <= F_prev + self.armijo * alpha * phi:
            self.arguments = arguments
            self.phi = phi
            self.alpha = alpha
            self.ls_iterations = iterations
            self.steps += 1
            self.termination = None
            return True

        # If the maximum number of ls_iterations have been performed, return without taking a step.
        if iterations == self.max_ls_iterations:
            self.mlnn.A = A_prev
            self.mlnn.E = E_prev
            self.mlnn.F = F_prev
            self.termination = 'max_ls_iterations'
            return False

        while True:
            # Decrease alpha with a cubic approximation of the minimizer, by interpolating F_prev, phi, self.F, and F_old.
            ab = (np.array([[alpha_old ** 2, - alpha ** 2], [- alpha_old ** 3, alpha ** 3]]) @
                  np.array([[self.mlnn.F - F_prev - phi * alpha], [F_old - F_prev - phi * alpha_old]]) /
                  (alpha_old ** 2 * alpha ** 2 * (alpha - alpha_old)))
            a = ab[0].item()
            b = ab[1].item()
            c = b ** 2 - 3 * a * phi
            alpha_old = alpha
            if c < 0:
                alpha = .5 * alpha_old
            else:
                alpha = (-b + c ** .5) / (3 * a)

            # Safeguard the interpolation by making sure alpha is not too small or too close to alpha_old.
            rho = alpha / alpha_old
            if not (self.rho_lo <= rho <= self.rho_hi):
                rho = .5
                alpha = rho * alpha_old

            # Take a step in the direction of steepest descent.
            F_old = self.mlnn.F
            if dA is not None:
                self.mlnn.update_A(A_prev, dA, alpha)
            if dE is not None:
                self.mlnn.update_E(E_prev, dE, alpha)

            iterations += 1

            # If Armijo's condition for sufficient decrease has been satisfied, the search is complete.
            if self.mlnn.F <= F_prev + self.armijo * alpha * phi:
                self.arguments = arguments
                self.phi = phi
                self.alpha = alpha
                self.ls_iterations = iterations
                self.steps += 1
                self.termination = None
                return True

            # If the maximum number of ls_iterations have been performed, return without taking a step.
            if iterations == self.max_ls_iterations:
                self.mlnn.A = A_prev
                self.mlnn.E = E_prev
                self.mlnn.F = F_prev
                self.termination = 'max_ls_iterations'
                return False

    def strong_wolfe_line_search(self, arguments, F_prev, A_prev, E_prev, dA, dE, phi, alpha, F_prev_prev=None):
        i = 0
        size = 0
        if 'A' in arguments:
            i = A_prev.size
            size += A_prev.size
        if 'E' in arguments:
            size += E_prev.size

        xk = np.empty(size)
        gfk = np.empty(size)
        if 'A' in arguments:
            xk[0:i] = A_prev.ravel()
            if dA is None:
                gfk[0:i] = np.zeros(A_prev.size)
            else:
                gfk[0:i] = dA.ravel()
        if 'E' in arguments:
            xk[i:] = E_prev.ravel()
            if dE is None:
                gfk[i:] = np.zeros(E_prev.size)
            else:
                gfk[i:] = dE.ravel()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            alpha, fc, _, new_fval, _, new_slope = scipy_line_search(
                self.mlnn.fun, self.mlnn.jac, xk, -gfk, gfk, F_prev, F_prev_prev,
                (arguments,), self.armijo, self.wolfe, alpha, None, self.max_ls_iterations)

        # If the strong Wolfe conditions are satisfied, the search is complete.
        # Otherwise, the maximum number of ls_iterations have been performed. Return without taking a step.
        if alpha is not None:
            # Take a step in the direction of steepest descent.
            if dA is not None:
                self.mlnn.update_A(A_prev, dA, alpha)
            if dE is not None:
                self.mlnn.update_E(E_prev, dE, alpha)

            self.mlnn.F = new_fval

            i = 0
            if 'A' in arguments:
                self.mlnn.dFdA = new_slope[0:self.A_0.size].reshape(self.A_0.shape)
                i = self.A_0.size
            if 'E' in arguments:
                self.mlnn.dFdE = new_slope[i:].reshape(self.E_0.shape)

            self.arguments = arguments
            self.phi = phi
            self.alpha = alpha
            self.ls_iterations = fc
            self.steps += 1
            self.termination = None
            return True
        else:
            self.mlnn.A = A_prev
            self.mlnn.E = E_prev
            self.mlnn.F = F_prev
            self.ls_iterations = fc
            self.termination = 'scipy_line_search() did not converge'
            return False

    def take_step(self, arguments='AE', alpha_0=None, F_prev_prev=None, method=None):
        if alpha_0 is None:
            alpha_0 = self.alpha_0

        if method is None:
            method = self.line_search_method

        A_prev = self.mlnn.A
        E_prev = self.mlnn.E
        F_prev = self.mlnn.F
        arguments_prev = self.arguments
        phi_prev = self.phi
        alpha_prev = self.alpha

        # Compute phi, the gradient of F with respect to the step size, alpha.
        phi = 0
        if 'A' in arguments and self.mlnn.phiA:
            dA = self.mlnn.dFdA
            phi += self.mlnn.phiA
        else:
            dA = None
        if 'E' in arguments and self.mlnn.phiE:
            dE = self.mlnn.dFdE
            phi += self.mlnn.phiE
        else:
            dE = None

        # If phi == 0, F is at a minimum or a saddle point. Return without taking a step.
        if phi == 0:
            self.termination = 'zero_phi'
            return False

        # If a previous step was taken with the same arguments, assume that this step's first-order change in
        # F will be the same (i.e. alpha * phi = alpha_prev * phi_prev). Otherwise, set alpha to alpha_0.
        if arguments == arguments_prev:
            pi = phi_prev / phi

            # Safeguard the initialization by ensuring alpha is larger than alpha_prev.
            if pi <= 1:
                pi = 1.1

            alpha = pi * alpha_prev
        else:
            pi = None
            alpha = alpha_0

        if method == 'backtracking':
            return self.backtracking_line_search(arguments, F_prev, A_prev, E_prev, dA, dE, phi, alpha)
        elif method == 'strong_wolfe':
            return self.strong_wolfe_line_search(arguments, F_prev, A_prev, E_prev, dA, dE, phi, alpha, F_prev_prev)

    def optimize_fixed(self, arguments='AE', alpha_0=None, min_delta_F=None, max_steps=None, max_time=None, verbose=None):
        if alpha_0 is None:
            alpha_0 = self.alpha_0

        if min_delta_F is None:
            min_delta_F = self.min_delta_F

        if max_steps is None:
            max_steps = self.max_steps

        if max_time is None:
            max_time = self.max_time

        if verbose is None:
            verbose = self.verbose

        if verbose:
            if self.callback is None:
                self.callback = MLNNCallback()
            self.callback.print_stats = True

        self.initialize_optimizer()

        if self.callback is not None:
            self.callback.start(self)

        self.arguments = None
        self.phi = None
        self.alpha = None
        self.ls_iterations = None
        self.delta_F = None

        F_prev = self.mlnn.F
        F_prev_prev = None

        while self.take_step(arguments, alpha_0, F_prev_prev):
            self.delta_F = F_prev - self.mlnn.F
            if self.use_prev_f:
                F_prev_prev = F_prev
            F_prev = self.mlnn.F

            if self.callback is not None:
                self.callback.iterate()

            if self.delta_F <= min_delta_F:
                self.termination = 'min_delta_F'
                break

            if self.steps == max_steps:
                self.termination = 'max_steps'
                break

            if self.time >= max_time:
                self.termination = 'max_time'
                break

        self.run_time = self.time

        if self.callback is not None:
            self.callback.end()

    def optimize_alternating(self, max_arg_steps=5, alpha_0=None, min_delta_F=None, max_steps=None, max_time=None, verbose=None):
        if alpha_0 is None:
            alpha_0 = self.alpha_0

        if min_delta_F is None:
            min_delta_F = self.min_delta_F

        if max_steps is None:
            max_steps = self.max_steps

        if max_time is None:
            max_time = self.max_time

        if verbose is None:
            verbose = self.verbose

        if verbose:
            if self.callback is None:
                self.callback = MLNNCallback()
            self.callback.print_stats = True

        self.initialize_optimizer()

        if self.callback is not None:
            self.callback.start(self)

        self.arguments = None
        self.phi = None
        self.alpha = None
        self.ls_iterations = None
        self.delta_F = None

        F_prev = self.mlnn.F
        F_prev_prev = None

        arguments = 'AE'
        arg_steps = 0
        arg_alpha_0 = {'AE': alpha_0, 'A': alpha_0, 'E': alpha_0}
        arg_terminated = {'AE': False, 'A': False, 'E': False}

        while True:
            if self.take_step(arguments, arg_alpha_0[arguments], F_prev_prev):
                arg_steps += 1
                arg_alpha_0[arguments] = self.alpha

                self.delta_F = F_prev - self.mlnn.F
                if self.use_prev_f:
                    F_prev_prev = F_prev
                F_prev = self.mlnn.F

                if self.callback is not None:
                    self.callback.iterate()

                if self.delta_F <= min_delta_F:
                    self.termination = 'min_delta_F'

            if self.termination is None:
                arg_terminated[arguments] = False
            else:
                arg_terminated[arguments] = True

                if all(arg_terminated.values()):
                    self.termination = 'All minimization modes terminated'
                    break

            if self.steps == max_steps:
                self.termination = 'max_steps'
                break

            if self.time >= max_time:
                self.termination = 'max_time'
                break

            if arg_terminated[arguments] or (arg_steps == max_arg_steps):
                if arguments == 'AE':
                    arguments = 'A'
                elif arguments == 'A':
                    arguments = 'E'
                elif arguments == 'E':
                    arguments = 'AE'

                arg_steps = 0

        self.run_time = self.time

        if self.callback is not None:
            self.callback.end()

    def minimize(self, method=None, arguments=None, max_arg_steps=None,
                 alpha_0=None, min_delta_F=None, max_steps=None, max_time=None, verbose=None):
        if method is None:
            method = self.optimize_method

        if arguments is None:
            arguments = self.opt_arguments

        if max_arg_steps is None:
            max_arg_steps = self.max_arg_steps

        if method == 'fixed':
            self.optimize_fixed(
                arguments=arguments, alpha_0=alpha_0, min_delta_F=min_delta_F,
                max_steps=max_steps, max_time=max_time, verbose=verbose
            )
        elif method == 'alternating':
            self.optimize_alternating(
                max_arg_steps=max_arg_steps, alpha_0=alpha_0, min_delta_F=min_delta_F,
                max_steps=max_steps, max_time=max_time, verbose=verbose
            )


class MLNNBFGS(MLNNOptimizer):
    def __init__(self, mlnn, callback=None, A_0=None, E_0=None, n_components=None,
                 optimize_params=None, line_search_params=None):
        super().__init__(mlnn, callback)

        self.maxcor = None
        self.gtol = None
        self.eps = None
        self.maxfun = None
        self.finite_diff_rel_step = None

        if optimize_params:
            self.apply_params(optimize_params)
        if line_search_params:
            self.apply_params(line_search_params)

        self.initialize_args(A_0, E_0, n_components)

        self.options = None
        self.bounds = None
        self.result = None

    def set_options(self):
        self.options = {}

        if self.max_steps is not None:
            self.options['maxiter'] = self.max_steps

        if self.min_delta_F is not None:
            self.options['ftol'] = self.min_delta_F

        if self.maxcor is not None:
            self.options['maxcor'] = self.maxcor

        if self.gtol is not None:
            self.options['gtol'] = self.gtol

        if self.eps is not None:
            self.options['eps'] = self.eps

        if self.maxfun is not None:
            self.options['maxfun'] = self.maxfun

        if self.finite_diff_rel_step is not None:
            self.options['finite_diff_rel_step'] = self.finite_diff_rel_step

        if self.max_ls_iterations is not None:
            self.options['maxls'] = self.max_ls_iterations

    def set_bounds(self, arguments):
        i = 0
        size = 0
        if 'A' in arguments:
            i = self.A_0.size
            size += self.A_0.size
        if 'E' in arguments:
            size += self.E_0.size

        lb = np.empty(size)
        if 'A' in arguments:
            if self.mlnn.keep_a_psd:
                if self.mlnn.a_mode == 'full':
                    raise RuntimeError('L-BFGS-B cannot impose the PSD constraint on a full matrix')
                elif self.mlnn.a_mode == 'diagonal':
                    lb[0:i] = np.zeros(self.A_0.size)
                elif self.mlnn.a_mode == 'decomposed':
                    lb[0:i] = np.full(self.A_0.size, -np.inf)
            else:
                lb[0:i] = np.full(self.A_0.size, -np.inf)
            if self.mlnn.keep_a_centered:
                if self.mlnn.a_mode in ('full', 'decomposed'):
                    raise RuntimeError('L-BFGS-B cannot impose the centering constraint')
        if 'E' in arguments:
            if self.mlnn.keep_e_positive:
                lb[i:] = np.zeros(self.E_0.size)
            else:
                lb[i:] = np.full(self.E_0.size, -np.inf)

        self.bounds = Bounds(lb, np.inf)
        self.mlnn.keep_a_psd = False
        self.mlnn.keep_a_centered = False
        self.mlnn.keep_e_positive = False

    def read_result(self, arguments):
        i = 0
        if 'A' in arguments:
            self.mlnn.A = self.result.x[0:self.A_0.size].reshape(self.A_0.shape)
            i = self.A_0.size
        if 'E' in arguments:
            self.mlnn.E = self.result.x[i:].reshape(self.E_0.shape)

        self.mlnn.F = self.result.fun
        self.steps = self.result.nit
        self.termination = self.result.message
        self.delta_F = self.callback.delta_F

    def minimize(self, arguments='AE', min_delta_F=None, max_steps=None, verbose=None):
        if min_delta_F is not None:
            self.min_delta_F = min_delta_F

        if max_steps is not None:
            self.max_steps = max_steps

        if verbose is None:
            verbose = self.verbose

        if self.callback is None:
            self.callback = MLNNCallback()

        if verbose:
            self.callback.print_stats = True

        self.set_options()
        self.set_bounds(arguments)
        self.initialize_optimizer()
        self.callback.start(self)

        x0_parts = []
        if 'A' in arguments:
            x0_parts.append(self.mlnn.A.ravel())
        if 'E' in arguments:
            x0_parts.append(self.mlnn.E.ravel())
        x0 = np.concatenate(x0_parts)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.result = scipy_minimize(self.mlnn.fun, x0, (arguments,), 'L-BFGS-B', self.mlnn.jac,
                                         bounds=self.bounds, options=self.options, callback=self.callback.iterate)

        self.read_result(arguments)

        self.run_time = self.time

        if self.callback is not None:
            self.callback.end()

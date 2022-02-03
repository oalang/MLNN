import time
import warnings
import numpy as np
from scipy.optimize import line_search, minimize, Bounds


class MLNNOptimizer:
    def __init__(self, mlnn, callback, A_0, E_0, d, optimize_params, line_search_params):
        self.mlnn = mlnn
        self.callback = callback

        if optimize_params:
            self.apply_params(optimize_params)
        if line_search_params:
            self.apply_params(line_search_params)

        if self.i_mode is None:
            if self.mlnn.a_mode == 'full' or self.mlnn.a_mode == 'diagonal':
                self.i_mode = 'zero'
            elif self.mlnn.a_mode == 'decomposed':
                self.i_mode = 'pca'

        if self.mlnn.a_mode == 'full':
            assert (self.i_mode == 'random' or self.i_mode == 'zero' or
                    self.i_mode == 'identity' or self.i_mode == 'centered')
        elif self.mlnn.a_mode == 'diagonal':
            assert (self.i_mode == 'random' or self.i_mode == 'zero' or
                    self.i_mode == 'identity')
        elif self.mlnn.a_mode == 'decomposed':
            assert (self.i_mode == 'random' or self.i_mode == 'pca')

        if A_0 is None:
            self.A_0 = mlnn.compute_A_0(self.i_mode, d)
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
            self.E_0 = mlnn.compute_E_0(self.i_mode)
        else:
            self.E_0 = E_0

        if self.mlnn.e_mode == 'single':
            assert np.isscalar(self.E_0)
        elif self.mlnn.e_mode == 'multiple':
            assert self.E_0.shape[0] == self.mlnn.n
            assert self.E_0.shape[1] == 1

        self.time_0 = None
        self.run_time = None
        self.steps = None
        self.F_0 = None
        self.termination = None

    @property
    def time(self):
        return time.perf_counter() - self.time_0

    def apply_params(self, params):
        for attr in params:
            if hasattr(self, attr):
                setattr(self, attr, params[attr])
    
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
        if hasattr(self, 'delta_F') and self.delta_F is not None:
            print(f"   delta_F = {self.delta_F:e}")
        print(f"     steps = {self.steps:d}")
        print(f"  run_time = {self.run_time:f} seconds")
        print("")
        print(f"   F function calls: {self.mlnn.F_count:d}")
        print(f"  dA function calls: {self.mlnn.dFdA_count:d}")
        print(f"  dE function calls: {self.mlnn.dFdE_count:d}")
        print(f"eigh function calls: {self.mlnn.eigh_count:d}")
        print("")


class MLNNBacktracking(MLNNOptimizer):
    def __init__(self, mlnn, callback=None, A_0=None, E_0=None, d=None, optimize_params=None, line_search_params=None):
        self.i_mode = None
        self.max_steps = 1000
        self.min_delta_F = 1e-6
        self.verbose_optimize = False
        self.max_time = 10
        self.method = 'fixed'

        self.max_ls_iterations = 20
        self.alpha_0 = 1e-6
        self.armijo = 1e-6
        self.rho_lo = .1
        self.rho_hi = .9
        self.verbose_line_search = False

        super().__init__(mlnn, callback, A_0, E_0, d, optimize_params, line_search_params)

        self.delta_F = None
        self.arguments = None
        self.phi = None
        self.alpha = None
        self.ls_iterations = None

    def initialize(self):
        self.time_0 = time.perf_counter()
        self.run_time = None

        self.steps = 0
        self.mlnn.F_count = 0
        self.mlnn.dFdA_count = 0
        self.mlnn.dFdE_count = 0
        self.mlnn.eigh_count = 0

        self.F_0 = None
        self.delta_F = None

        self.arguments = None
        self.phi = None
        self.alpha = None
        self.ls_iterations = None
        self.termination = None

        self.mlnn.A = self.A_0
        self.mlnn.E = self.E_0

    def take_step(self, arguments='AE', alpha_0=None, verbose=None):
        if alpha_0 is None:
            alpha_0 = self.alpha_0

        if verbose is None:
            verbose = self.verbose_line_search

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

        # Commence line search.
        ls_iterations = 0

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

        # Take a step in the direction of steepest descent.
        if dA is not None:
            self.mlnn.update_A(A_prev, alpha, dA)
        if dE is not None:
            self.mlnn.update_E(E_prev, alpha, dE)

        # If Armijo's condition for sufficient decrease has been satisfied, the search is complete.
        if self.mlnn.F <= F_prev + self.armijo * alpha * phi:
            self.arguments = arguments
            self.phi = phi
            self.alpha = alpha
            self.ls_iterations = ls_iterations
            self.termination = None
            self.steps += 1
            return True

        # If the maximum number of ls_iterations have been performed, return without taking a step.
        if ls_iterations == self.max_ls_iterations:
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
            self.mlnn.update_A(A_prev, alpha, dA)
        if dE is not None:
            self.mlnn.update_E(E_prev, alpha, dE)

        ls_iterations += 1

        # If Armijo's condition for sufficient decrease has been satisfied, the search is complete.
        if self.mlnn.F <= F_prev + self.armijo * alpha * phi:
            self.arguments = arguments
            self.phi = phi
            self.alpha = alpha
            self.ls_iterations = ls_iterations
            self.termination = None
            self.steps += 1
            return True

        # If the maximum number of ls_iterations have been performed, return without taking a step.
        if ls_iterations == self.max_ls_iterations:
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
                self.mlnn.update_A(A_prev, alpha, dA)
            if dE is not None:
                self.mlnn.update_E(E_prev, alpha, dE)

            ls_iterations += 1

            # If Armijo's condition for sufficient decrease has been satisfied, the search is complete.
            if self.mlnn.F <= F_prev + self.armijo * alpha * phi:
                self.arguments = arguments
                self.phi = phi
                self.alpha = alpha
                self.ls_iterations = ls_iterations
                self.termination = None
                self.steps += 1
                return True

            # If the maximum number of ls_iterations have been performed, return without taking a step.
            if ls_iterations == self.max_ls_iterations:
                self.mlnn.A = A_prev
                self.mlnn.E = E_prev
                self.mlnn.F = F_prev
                self.termination = 'max_ls_iterations'
                return False

    def minimize(self, method=None, **kwargs):
        if method is None:
            method = self.method

        alpha_0 = kwargs['alpha_0'] if 'alpha_0' in kwargs else None
        min_delta_F = kwargs['min_delta_F'] if 'min_delta_F' in kwargs else None
        max_steps = kwargs['max_steps'] if 'max_steps' in kwargs else None
        max_time = kwargs['max_time'] if 'max_time' in kwargs else None
        verbose = kwargs['verbose'] if 'verbose' in kwargs else None

        if method == 'fixed':
            arguments = kwargs['arguments'] if 'arguments' in kwargs else 'AE'

            self.minimize_fixed(
                arguments=arguments, alpha_0=alpha_0, min_delta_F=min_delta_F,
                max_steps=max_steps, max_time=max_time, verbose=verbose
            )
        elif method == 'alternating':
            max_arg_steps = kwargs['max_arg_steps'] if 'max_arg_steps' in kwargs else 5

            self.minimize_alternating(
                max_arg_steps=max_arg_steps, alpha_0=alpha_0, min_delta_F=min_delta_F,
                max_steps=max_steps, max_time=max_time, verbose=verbose
            )

    def minimize_fixed(self, arguments='AE', alpha_0=None, min_delta_F=None, max_steps=None, max_time=None, verbose=None):
        if alpha_0 is None:
            alpha_0 = self.alpha_0

        if min_delta_F is None:
            min_delta_F = self.min_delta_F

        if max_steps is None:
            max_steps = self.max_steps

        if max_time is None:
            max_time = self.max_time

        if verbose is None:
            verbose = self.verbose_optimize

        if verbose:
            if self.callback is None:
                self.callback = MLNNCallback()
            self.callback.print_stats = True

        self.initialize()

        if self.callback is not None:
            self.callback.start(self)

        self.F_0 = F_prev = self.mlnn.F

        while self.take_step(arguments, alpha_0):
            self.delta_F = F_prev - self.mlnn.F
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

    def minimize_alternating(self, max_arg_steps=5, alpha_0=None, min_delta_F=None, max_steps=None, max_time=None, verbose=None):
        if alpha_0 is None:
            alpha_0 = self.alpha_0

        if min_delta_F is None:
            min_delta_F = self.min_delta_F

        if max_steps is None:
            max_steps = self.max_steps

        if max_time is None:
            max_time = self.max_time

        if verbose is None:
            verbose = self.verbose_optimize

        if verbose:
            if self.callback is None:
                self.callback = MLNNCallback()
            self.callback.print_stats = True

        self.initialize()

        if self.callback is not None:
            self.callback.start(self)

        arguments = 'AE'
        arg_alpha_0 = {'AE': alpha_0, 'A': alpha_0, 'E': alpha_0}
        arg_terminated = {'AE': False, 'A': False, 'E': False}
        arg_steps = 0

        self.F_0 = F_prev = self.mlnn.F

        while True:
            if self.take_step(arguments, arg_alpha_0[arguments]):
                arg_alpha_0[arguments] = self.alpha
                arg_steps += 1

                self.delta_F = F_prev - self.mlnn.F
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
                    self.termination = 'convergence'
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
                else:
                    arguments = 'AE'

                arg_steps = 0

        self.run_time = self.time

        if self.callback is not None:
            self.callback.end()







class MLNNStrongWolfe(MLNNOptimizer):
    def __init__(self, mlnn, callback=None, A_0=None, E_0=None, d=None, optimize_params=None, line_search_params=None):
        self.i_mode = None
        self.max_steps = 1000
        self.min_delta_F = 1e-6
        self.verbose_optimize = False
        self.max_time = 10
        self.method = 'fixed'

        self.max_ls_iterations = 20
        self.alpha_0 = 1e-6
        self.armijo = 1e-6
        self.wolfe = .9
        self.verbose_line_search = False

        super().__init__(mlnn, callback, A_0, E_0, d, optimize_params, line_search_params)

        self.delta_F = None
        self.arguments = None
        self.phi = None
        self.alpha = None
        self.ls_iterations = None

    def initialize(self):
        self.time_0 = time.perf_counter()
        self.run_time = None

        self.steps = 0
        self.mlnn.F_count = 0
        self.mlnn.dFdA_count = 0
        self.mlnn.dFdE_count = 0
        self.mlnn.eigh_count = 0

        self.F_0 = None
        self.delta_F = None

        self.arguments = None
        self.phi = None
        self.alpha = None
        self.ls_iterations = None
        self.termination = None

        self.mlnn.A = self.A_0
        self.mlnn.E = self.E_0

    def take_step(self, arguments='AE', alpha_0=None, verbose=None):
        if alpha_0 is None:
            alpha_0 = self.alpha_0

        if verbose is None:
            verbose = self.verbose_line_search

        A_prev = self.mlnn.A
        E_prev = self.mlnn.E
        F_prev = self.mlnn.F
        arguments_prev = self.arguments
        phi_prev = self.phi
        alpha_prev = self.alpha

        # Compute phi, the gradient of F with respect to the step size, alpha.
        phi = 0
        xk = np.empty(0)
        gfk = np.empty(0)
        if 'A' in arguments:
            dA = self.mlnn.dFdA
            phi += self.mlnn.phiA
            xk = np.append(xk, A_prev)
            gfk = np.append(gfk, dA)
        if 'E' in arguments:
            dE = self.mlnn.dFdE
            phi += self.mlnn.phiE
            xk = np.append(xk, E_prev)
            gfk = np.append(gfk, dE)

        # If phi == 0, F is at a minimum or a saddle point. Return without taking a step.
        if phi == 0:
            self.termination = 'zero_phi'
            return False

        # Commence line search.
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

        with warnings.catch_warnings():
            #warnings.simplefilter("ignore")
            alpha, fc, gc, new_fval, old_fval, new_slope = line_search(
                self.mlnn.fun, self.mlnn.jac, xk, -gfk, gfk, F_prev, None,
                (arguments,), self.armijo, self.wolfe, alpha, None, self.max_ls_iterations)

        # If the strong Wolfe conditions are satisfied, the search is complete.
        # Otherwise, the maximum number of ls_iterations have been performed. Return without taking a step.
        if alpha is not None:
            # Take a step in the direction of steepest descent.
            if 'A' in arguments:
                self.mlnn.update_A(A_prev, alpha, dA)
            if 'E' in arguments:
                self.mlnn.update_E(E_prev, alpha, dE)

            self.mlnn.F = new_fval

            i = 0
            if 'A' in arguments:
                self.mlnn.dFdA = new_slope[0:self.mlnn.dFdA.size].reshape(self.mlnn.dFdA.shape)
                i = self.mlnn.dFdA.size
            if 'E' in arguments:
                if self.mlnn.e_mode == 'single':
                    self.mlnn.dFdE = new_slope[-1]
                elif self.mlnn.e_mode == 'multiple':
                    self.mlnn.dFdE = new_slope[i:].reshape(self.mlnn.dFdE.shape)

            self.arguments = arguments
            self.phi = phi
            self.alpha = alpha
            self.ls_iterations = fc - 1
            self.termination = None
            self.steps += 1
            return True
        else:
            self.mlnn.A = A_prev
            self.mlnn.E = E_prev
            self.mlnn.F = F_prev
            self.ls_iterations = fc - 1
            self.termination = 'max_ls_iterations'
            return False

    def minimize(self, method=None, **kwargs):
        if method is None:
            method = self.method

        alpha_0 = kwargs['alpha_0'] if 'alpha_0' in kwargs else None
        min_delta_F = kwargs['min_delta_F'] if 'min_delta_F' in kwargs else None
        max_steps = kwargs['max_steps'] if 'max_steps' in kwargs else None
        max_time = kwargs['max_time'] if 'max_time' in kwargs else None
        verbose = kwargs['verbose'] if 'verbose' in kwargs else None

        if method == 'fixed':
            arguments = kwargs['arguments'] if 'arguments' in kwargs else 'AE'

            self.minimize_fixed(
                arguments=arguments, alpha_0=alpha_0, min_delta_F=min_delta_F,
                max_steps=max_steps, max_time=max_time, verbose=verbose
            )
        elif method == 'alternating':
            max_arg_steps = kwargs['max_arg_steps'] if 'max_arg_steps' in kwargs else 5

            self.minimize_alternating(
                max_arg_steps=max_arg_steps, alpha_0=alpha_0, min_delta_F=min_delta_F,
                max_steps=max_steps, max_time=max_time, verbose=verbose
            )

    def minimize_fixed(self, arguments='AE', alpha_0=None, min_delta_F=None, max_steps=None, max_time=None, verbose=None):
        if alpha_0 is None:
            alpha_0 = self.alpha_0

        if min_delta_F is None:
            min_delta_F = self.min_delta_F

        if max_steps is None:
            max_steps = self.max_steps

        if max_time is None:
            max_time = self.max_time

        if verbose is None:
            verbose = self.verbose_optimize

        if verbose:
            if self.callback is None:
                self.callback = MLNNCallback()
            self.callback.print_stats = True

        self.initialize()

        if self.callback is not None:
            self.callback.start(self)

        self.F_0 = F_prev = self.mlnn.F

        while self.take_step(arguments, alpha_0):
            self.delta_F = F_prev - self.mlnn.F
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

    def minimize_alternating(self, max_arg_steps=5, alpha_0=None, min_delta_F=None, max_steps=None, max_time=None, verbose=None):
        if alpha_0 is None:
            alpha_0 = self.alpha_0

        if min_delta_F is None:
            min_delta_F = self.min_delta_F

        if max_steps is None:
            max_steps = self.max_steps

        if max_time is None:
            max_time = self.max_time

        if verbose is None:
            verbose = self.verbose_optimize

        if verbose:
            if self.callback is None:
                self.callback = MLNNCallback()
            self.callback.print_stats = True

        self.initialize()

        if self.callback is not None:
            self.callback.start(self)

        arguments = 'AE'
        arg_alpha_0 = {'AE': alpha_0, 'A': alpha_0, 'E': alpha_0}
        arg_terminated = {'AE': False, 'A': False, 'E': False}
        arg_steps = 0

        self.F_0 = F_prev = self.mlnn.F

        while True:
            if self.take_step(arguments, arg_alpha_0[arguments]):
                arg_alpha_0[arguments] = self.alpha
                arg_steps += 1

                self.delta_F = F_prev - self.mlnn.F
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
                    self.termination = 'convergence'
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
                else:
                    arguments = 'AE'

                arg_steps = 0

        self.run_time = self.time

        if self.callback is not None:
            self.callback.end()








class MLNNBFGS(MLNNOptimizer):
    def __init__(self, mlnn, callback=None, A_0=None, E_0=None, d=None, optimize_params=None, line_search_params=None):
        self.i_mode = None
        self.max_steps = 1000
        self.min_delta_F = 1e-6
        self.verbose_optimize = False
        self.maxcor = None
        self.gtol = None
        self.eps = None
        self.maxfun = None
        self.iprint = None
        self.finite_diff_rel_step = None

        self.max_ls_iterations = 20

        super().__init__(mlnn, callback, A_0, E_0, d, optimize_params, line_search_params)

        self.options = None
        self.bounds = None
        self.result = None

    def set_options(self):
        self.options = {}

        if self.max_steps is not None:
            self.options['maxiter'] = self.max_steps

        if self.min_delta_F is not None:
            self.options['ftol'] = self.min_delta_F

        if self.verbose_optimize is not None:
            if self.verbose_optimize:
                self.options['disp'] = 1
            else:
                self.options['disp'] = 0

        if self.maxcor is not None:
            self.options['maxcor'] = self.maxcor

        if self.gtol is not None:
            self.options['gtol'] = self.gtol

        if self.eps is not None:
            self.options['eps'] = self.eps

        if self.maxfun is not None:
            self.options['maxfun'] = self.maxfun

        if self.iprint is not None:
            self.options['iprint'] = self.iprint

        if self.finite_diff_rel_step is not None:
            self.options['finite_diff_rel_step'] = self.finite_diff_rel_step

        if self.max_ls_iterations is not None:
            self.options['maxls'] = self.max_ls_iterations

    def set_bounds(self, arguments):
        lb = np.empty(0)

        if 'A' in arguments:
            if self.mlnn.keep_a_psd:
                if self.mlnn.a_mode == 'full':
                    raise RuntimeError('L-BFGS-B cannot impose a PSD constraint on a full matrix')
                elif self.mlnn.a_mode == 'diagonal':
                    lb = np.append(lb, np.zeros(self.A_0.size))
                elif self.mlnn.a_mode == 'decomposed':
                    lb = np.append(lb, np.full(self.A_0.size, -np.inf))
            else:
                lb = np.append(lb, np.full(self.A_0.size, -np.inf))

            if self.mlnn.keep_a_centered:
                if self.mlnn.a_mode == 'full' or self.mlnn.a_mode == 'decomposed':
                    raise RuntimeError('L-BFGS-B cannot impose centering constraint')

        if 'E' in arguments:
            if self.mlnn.keep_e_positive:
                if self.mlnn.e_mode == 'single':
                    lb = np.append(lb, 0)
                elif self.mlnn.e_mode == 'multiple':
                    lb = np.append(lb, np.zeros(self.E_0.size))
            else:
                if self.mlnn.e_mode == 'single':
                    lb = np.append(lb, -np.inf)
                elif self.mlnn.e_mode == 'multiple':
                    lb = np.append(lb, np.full(self.E_0.size, -np.inf))

        self.bounds = Bounds(lb, np.inf)
        self.mlnn.keep_a_psd = False
        self.mlnn.keep_a_centered = False
        self.mlnn.keep_e_positive = False

    def initialize(self):
        self.time_0 = time.perf_counter()
        self.run_time = None

        self.mlnn.A = self.A_0
        self.mlnn.E = self.E_0
        self.F_0 = self.mlnn.F

        self.steps = 0
        self.mlnn.F_count = 0
        self.mlnn.dFdA_count = 0
        self.mlnn.dFdE_count = 0

        self.termination = None

    def read_result(self, arguments):
        i = 0
        if 'A' in arguments:
            self.mlnn.A = self.result.x[0:self.A_0.size].reshape(self.A_0.shape)
            i = self.A_0.size
        if 'E' in arguments:
            if self.mlnn.e_mode == 'single':
                self.mlnn.E = self.result.x[-1]
            elif self.mlnn.e_mode == 'multiple':
                self.mlnn.E = self.result.x[i:].reshape(self.E_0.shape)

        self.steps = self.result.nit
        self.termination = self.result.message

    def minimize(self, arguments='AE', min_delta_F=None, max_steps=None, verbose=None):
        if min_delta_F is not None:
            self.min_delta_F = min_delta_F

        if max_steps is not None:
            self.max_steps = max_steps

        if verbose is not None:
            self.verbose_optimize = verbose

        self.set_options()
        self.set_bounds(arguments)
        self.initialize()

        iterate = None
        if self.callback is not None:
            self.callback.start(self)
            iterate = self.callback.iterate

        x0 = np.empty(0)
        if 'A' in arguments:
            x0 = np.append(x0, self.mlnn.A)
        if 'E' in arguments:
            x0 = np.append(x0, self.mlnn.E)

        self.result = minimize(self.mlnn.fun, x0, (arguments,), 'L-BFGS-B', self.mlnn.jac,
                               bounds=self.bounds, options=self.options, callback=iterate)

        self.run_time = self.time

        self.read_result(arguments)

        if self.callback is not None:
            self.callback.end()


class MLNNCallback:
    def __init__(self, print_stats=False, collect_stats=False, show_figures=False):
        self.print_stats = print_stats
        self.collect_stats = collect_stats
        self.show_figures = show_figures

        self.optimizer = None
        self.iter = None
        self.F_prev = None
        self.delta_F = None

    def start(self, optimizer):
        self.optimizer = optimizer
        self.iter = 0
        self.F_prev = self.optimizer.mlnn.F

        if self.print_stats:
            self._print_stats_start()

        if self.collect_stats:
            self._collect_stats_start()

        if self.show_figures:
            self._show_figures_start()

    def iterate(self, _=None):
        self.iter += 1
        self.delta_F = self.F_prev - self.optimizer.mlnn.F
        self.F_prev = self.optimizer.mlnn.F

        if self.print_stats:
            self._print_stats_iterate()

        if self.collect_stats:
            self._collect_stats_iterate()

        if self.show_figures:
            self._show_figures_iterate()

    def end(self):
        if self.print_stats:
            self._print_stats_end()

        if self.collect_stats:
            self._collect_stats_end()

        if self.show_figures:
            self._show_figures_end()

    def _print_stats_start(self):
        self._print_optimize_header()
        self._print_optimize_row()

    def _print_stats_iterate(self):
        self._print_optimize_row()

    def _print_stats_end(self):
        pass

    def _collect_stats_start(self):
        pass

    def _collect_stats_iterate(self):
        pass

    def _collect_stats_end(self):
        pass

    def _show_figures_start(self):
        pass

    def _show_figures_iterate(self):
        pass

    def _show_figures_end(self):
        pass

    def _print_optimize_header(self):
        steps = f"{'step':^5s}"
        arguments = f"{'args':^4s}" if hasattr(self.optimizer, 'arguments') else ""
        ls_iterations = f"{'iter':^4s}" if hasattr(self.optimizer, 'ls_iterations') else ""
        alpha = f"{'alpha':^10s}" if hasattr(self.optimizer, 'alpha') else ""
        phi = f"{'phi':^10s}" if hasattr(self.optimizer, 'phi') else ""
        delta_F = f"{'delta_F':^10s}"
        F = f"{'F':^10s}"
        R = f"{'R':^10s}"
        S = f"{'S':^10s}"
        L = f"{'L':^10s}"
        mean_E = f"{'mean_E':^10s}"
        actv_rows = f"{'actv_rows':^9s}"
        actv_cols = f"{'actv_cols':^9s}"
        actv_data = f"{'actv_data':^9s}"

        print(" ".join((steps, arguments, ls_iterations, alpha, phi, delta_F, F, R, S, L, mean_E, actv_rows, actv_cols, actv_data)))

    def _print_optimize_row(self):
        steps = f"{self.iter:5d}" if self.iter is not None else f"{'-':^5s}"
        arguments = ((f"{self.optimizer.arguments:^4s}" if self.optimizer.arguments is not None else f"{'-':^4s}")
                     if hasattr(self.optimizer, 'arguments') else "")
        ls_iterations = ((f"{self.optimizer.ls_iterations:4d}" if self.optimizer.ls_iterations is not None else f"{'-':^4s}")
                      if hasattr(self.optimizer, 'ls_iterations') else "")
        alpha = ((f"{self.optimizer.alpha:10.3e}" if self.optimizer.alpha is not None else f"{'-':^10s}")
                 if hasattr(self.optimizer, 'alpha') else "" if hasattr(self.optimizer, 'alpha') else "")
        phi = ((f"{self.optimizer.phi:10.3e}" if self.optimizer.phi is not None else f"{'-':^10s}")
               if hasattr(self.optimizer, 'phi') else "")
        delta_F = f"{self.delta_F:10.3e}" if self.delta_F is not None else f"{'-':^10s}"
        F = f"{self.optimizer.mlnn.F:10.3e}" if self.optimizer.mlnn.F is not None else f"{'-':^10s}"
        R = f"{self.optimizer.mlnn.R:10.3e}" if self.optimizer.mlnn.R is not None else f"{'-':^10s}"
        S = f"{self.optimizer.mlnn.S:10.3e}" if self.optimizer.mlnn.S is not None else f"{'-':^10s}"
        L = f"{self.optimizer.mlnn.L:10.3e}" if self.optimizer.mlnn.L is not None else f"{'-':^10s}"
        mean_E = f"{np.mean(self.optimizer.mlnn.E):10.3e}" if self.optimizer.mlnn.E is not None else f"{'-':^10s}"
        actv_rows = (f"{self.optimizer.mlnn.subset_active_rows.size:9d}"
                     if self.optimizer.mlnn.subset_active_rows.size is not None else f"{'-':^9s}")
        actv_cols = (f"{self.optimizer.mlnn.subset_active_cols.size:9d}"
                     if self.optimizer.mlnn.subset_active_cols.size is not None else f"{'-':^9s}")
        actv_data = (f"{self.optimizer.mlnn.subset_active_data.size:9d}"
                     if self.optimizer.mlnn.subset_active_data.size is not None else f"{'-':^9s}")

        print(" ".join((steps, arguments, ls_iterations, alpha, phi, delta_F, F, R, S, L, mean_E, actv_rows, actv_cols, actv_data)))

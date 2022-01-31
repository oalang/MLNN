import time
import numpy as np


class MLNNSteepestDescent:
    def __init__(self, mlnn, A_0=None, E_0=None, d=None, optimize_params=None, line_search_params=None):
        self.mlnn = mlnn

        self.i_mode = None
        self.max_steps = 1000
        self.max_time = 10
        self.min_delta_F = 1e-6
        self.verbose_optimize = False

        self.alpha_0 = 1e-6
        self.armijo = 1e-6
        self.rho_lo = .1
        self.rho_hi = .9
        self.max_backtracks = 20
        self.verbose_line_search = False

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

    def apply_params(self, params):
        for attr in params:
            setattr(self, attr, params[attr])

    @property
    def time(self):
        return time.perf_counter() - self.time_0

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
        self.backtracks = None
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
        backtracks = 0

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
            self.backtracks = backtracks
            self.termination = None
            self.steps += 1
            return True

        # If the maximum number of backtracks have been performed, return without taking a step.
        if backtracks == self.max_backtracks:
            self.mlnn.A = A_prev
            self.mlnn.E = E_prev
            self.mlnn.F = F_prev
            self.termination = 'max_backtracks'
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

        backtracks += 1

        # If Armijo's condition for sufficient decrease has been satisfied, the search is complete.
        if self.mlnn.F <= F_prev + self.armijo * alpha * phi:
            self.arguments = arguments
            self.phi = phi
            self.alpha = alpha
            self.backtracks = backtracks
            self.termination = None
            self.steps += 1
            return True

        # If the maximum number of backtracks have been performed, return without taking a step.
        if backtracks == self.max_backtracks:
            self.mlnn.A = A_prev
            self.mlnn.E = E_prev
            self.mlnn.F = F_prev
            self.termination = 'max_backtracks'
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

            backtracks += 1

            # If Armijo's condition for sufficient decrease has been satisfied, the search is complete.
            if self.mlnn.F <= F_prev + self.armijo * alpha * phi:
                self.arguments = arguments
                self.phi = phi
                self.alpha = alpha
                self.backtracks = backtracks
                self.termination = None
                self.steps += 1
                return True

            # If the maximum number of backtracks have been performed, return without taking a step.
            if backtracks == self.max_backtracks:
                self.mlnn.A = A_prev
                self.mlnn.E = E_prev
                self.mlnn.F = F_prev
                self.termination = 'max_backtracks'
                return False

    def optimize(self, arguments='AE', alpha_0=None, min_delta_F=None, max_steps=None, max_time=None, verbose=None):
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

        self.initialize()

        if verbose:
            self._print_optimize_header()
            self._print_optimize_row()

        self.F_0 = F_prev = self.mlnn.F

        while self.take_step(arguments, alpha_0):
            self.delta_F = F_prev - self.mlnn.F
            F_prev = self.mlnn.F

            if verbose:
                self._print_optimize_row()

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

        if verbose:
            self._print_optimize_header()
            self.print_result()

    def minimize(self, max_arg_steps=5, alpha_0=None, min_delta_F=None, max_steps=None, max_time=None, verbose=None):
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

        self.initialize()

        if verbose:
            self._print_optimize_header()
            self._print_optimize_row()

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

                if verbose:
                    self._print_optimize_row()

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

        if verbose:
            self._print_optimize_header()
            self.print_result()

    @staticmethod
    def _print_optimize_header():
        steps = f"{'step':^5s}"
        arguments = f"{'args':^4s}"
        backtracks = f"{'bktr':^4s}"
        alpha = f"{'alpha':^10s}"
        phi = f"{'phi':^10s}"
        delta_F = f"{'delta_F':^10s}"
        F = f"{'F':^10s}"
        R = f"{'R':^10s}"
        S = f"{'S':^10s}"
        L = f"{'L':^10s}"
        mean_E = f"{'mean_E':^10s}"
        actv_rows = f"{'actv_rows':^9s}"
        actv_cols = f"{'actv_cols':^9s}"
        actv_data = f"{'actv_data':^9s}"

        print(" ".join((steps, arguments, backtracks, alpha, phi, delta_F, F, R, S, L, mean_E, actv_rows, actv_cols, actv_data)))

    def _print_optimize_row(self):
        steps = f"{self.steps:5d}" if self.steps is not None else f"{'-':^5s}"
        arguments = f"{self.arguments:^4s}" if self.arguments is not None else f"{'-':^4s}"
        backtracks = f"{self.backtracks:4d}" if self.backtracks is not None else f"{'-':^4s}"
        alpha = f"{self.alpha:10.3e}" if self.alpha is not None else f"{'-':^10s}"
        phi = f"{self.phi:10.3e}" if self.phi is not None else f"{'-':^10s}"
        delta_F = f"{self.delta_F:10.3e}" if self.delta_F is not None else f"{'-':^10s}"
        F = f"{self.F:10.3e}" if self.mlnn.F is not None else f"{'-':^10s}"
        R = f"{self.R:10.3e}" if self.mlnn.R is not None else f"{'-':^10s}"
        S = f"{self.S:10.3e}" if self.mlnn.S is not None else f"{'-':^10s}"
        L = f"{self.L:10.3e}" if self.mlnn.L is not None else f"{'-':^10s}"
        mean_E = f"{np.mean(self.mlnn.E):10.3e}" if self.E is not None else f"{'-':^10s}"
        actv_rows = f"{self.mlnn.subset_active_rows.size:9d}" if self.mlnn.subset_active_rows.size is not None else f"{'-':^9s}"
        actv_cols = f"{self.mlnn.subset_active_cols.size:9d}" if self.mlnn.subset_active_cols.size is not None else f"{'-':^9s}"
        actv_data = f"{self.mlnn.subset_active_data.size:9d}" if self.mlnn.subset_active_data.size is not None else f"{'-':^9s}"

        print(" ".join((steps, arguments, backtracks, alpha, phi, delta_F, F, R, S, L, mean_E, actv_rows, actv_cols, actv_data)))

    def print_result(self):
        if self.termination == 'max_backtracks':
            threshold = f" (max_backtracks = {self.max_backtracks:d})"
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
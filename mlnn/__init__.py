import time
import numpy as np
from loss import ReLU


class MLNN:
    def __init__(self, B, T, N, A_0=None, E_0=None, mlnn_params=None, line_search_params=None, optimize_params=None):
        self.r = 0
        self.s = 0
        self.l = 1
        self.inner = ReLU(1)
        self.outer = ReLU(1)
        self.a_mode = 'WX'
        self.e_mode = 'single'

        self.alpha_0 = 1e-6
        self.armijo = 1e-6
        self.rho_lo = .1
        self.rho_hi = .9
        self.max_backtracks = 20
        self.verbose_line_search = False

        self.max_steps = 1000
        self.max_time = 10
        self.min_delta_F = 1e-6
        self.verbose_optimize = False

        if mlnn_params:
            self.apply_params(mlnn_params)
        if line_search_params:
            self.apply_params(line_search_params)
        if optimize_params:
            self.apply_params(optimize_params)

        assert self.l > 0

        self.B = B
        self.T = T
        self.N = N

        self.n = self.B.shape[0]
        self.d = self.B.shape[1]

        if A_0 is not None:
            self.A_0 = A_0
        elif 'W' in self.a_mode:
            self.A_0 = np.zeros((self.d, self.d))
        else:
            self.A_0 = np.zeros((self.n, self.n))

        ### self.a_mode logic ###

        if E_0 is not None:
            self.E_0 = E_0
        elif self.e_mode == 'single':
            self.E_0 = np.zeros(1)
        elif self.e_mode == 'multiple':
            self.E_0 = np.zeros(self.n)

        if self.E_0.size == 1:
            self.e_mode = 'single'
        else:
            self.e_mode = 'multiple'

    def apply_params(self, params):
        for attr in params:
            setattr(self, attr, params[attr])

    def initialize(self):
        self.A = self.A_0
        self.E = self.E_0

        self.arguments = None
        self.phi = None
        self.alpha = None
        self.backtracks = None
        self.termination = None

        self.F_0 = None
        self.delta_F = None

        self.steps = 0
        self.F_count = 0
        self.dFdA_count = 0
        self.dFdE_count = 0

        self.time_0 = time.perf_counter()
        self.run_time = None

        if not self.r:
            self.R = 0
            self.dRdA = 0
        if not self.s:
            self.S = 0
            self.dSdE = 0

    @property
    def time(self):
        return time.perf_counter() - self.time_0

    @property
    def B(self):
        return self._B

    @B.setter
    def B(self, B):
        self._B = B

        if self.a_mode == 'MX' or self.a_mode == 'MXX' or self.a_mode == 'MG':
            self.C = None
        self.P = None
        if self.a_mode == 'WX' or self.a_mode == 'MG':
            self.dLdA = None
        if self.a_mode == 'MX' or self.a_mode == 'MXX':
            self.gFgA = None

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, T):
        self._T = T

        self.I = None
        self.V = None

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, N):
        self._N = N

        self.O = None

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, A):
        self._A = A

        self.C = None
        if self.r and (self.a_mode == 'WX' or self.a_mode == 'MX' or self.a_mode == 'MXX'):
            self.dRdA = None

    @property
    def E(self):
        return self._E

    @E.setter
    def E(self, E):
        self._E = E

        if self.s:
            self.S = None
        self.I = None
        if self.s:
            self.dSdE = None

    @property
    def C(self):
        if self._C is None:
            self._compute_C()
        return self._C

    @C.setter
    def C(self, C):
        self._C = C

        if self.r:
            self.R = None
        self.P = None

    @property
    def R(self):
        if self._R is None:
            self._compute_R()
        return self._R

    @R.setter
    def R(self, R):
        self._R = R

        self.F = None

    @property
    def S(self):
        if self._S is None:
            self._compute_S()
        return self._S

    @S.setter
    def S(self, S):
        self._S = S

        self.F = None

    @property
    def P(self):
        if self._P is None:
            self._compute_P()
        return self._P

    @P.setter
    def P(self, P):
        self._P = P

        self.D = None
        if self.r and self.a_mode == 'MG':
            self.dRdA = None

    @property
    def D(self):
        if self._D is None:
            self._compute_D()
        return self._D

    @D.setter
    def D(self, D):
        self._D = D

        self.I = None

    @property
    def I(self):
        if self._I is None:
            self._compute_I()
        return self._I

    @I.setter
    def I(self, I):
        self._I = I

        self.O = None
        self.V = None

    @property
    def O(self):
        if self._O is None:
            self._compute_O()
        return self._O

    @O.setter
    def O(self, O):
        self._O = O

        self.L = None
        self.V = None

    @property
    def L(self):
        if self._L is None:
            self._compute_L()
        return self._L

    @L.setter
    def L(self, L):
        self._L = L

        self.F = None

    @property
    def F(self):
        if self._F is None:
            self._compute_F()
        return self._F

    @F.setter
    def F(self, F):
        self._F = F

    @property
    def V(self):
        if self._V is None:
            self._compute_V()
        return self._V

    @V.setter
    def V(self, V):
        self._V = V

        self.dLdA = None
        self.dLdE = None

    @property
    def dRdA(self):
        if self._dRdA is None:
            self._compute_dRdA()
        return self._dRdA

    @dRdA.setter
    def dRdA(self, dRdA):
        self._dRdA = dRdA

        self.dFdA = None

    @property
    def dLdA(self):
        if self._dLdA is None:
            self._compute_dLdA()
        return self._dLdA

    @dLdA.setter
    def dLdA(self, dLdA):
        self._dLdA = dLdA

        self.dFdA = None

    @property
    def dFdA(self):
        if self._dFdA is None:
            self._compute_dFdA()
        return self._dFdA

    @dFdA.setter
    def dFdA(self, dFdA):
        self._dFdA = dFdA

        self.gFgA = None
        self.phiA = None

    @property
    def gFgA(self):
        if self._gFgA is None:
            self._compute_gFgA()
        return self._gFgA

    @gFgA.setter
    def gFgA(self, gFgA):
        self._gFgA = gFgA

        self.phiA = None

    @property
    def phiA(self):
        if self._phiA is None:
            self._compute_phiA()
        return self._phiA

    @phiA.setter
    def phiA(self, phiA):
        self._phiA = phiA

    @property
    def dSdE(self):
        if self._dSdE is None:
            self._compute_dSdE()
        return self._dSdE

    @dSdE.setter
    def dSdE(self, dSdE):
        self._dSdE = dSdE

        self.dFdE = None

    @property
    def dLdE(self):
        if self._dLdE is None:
            self._compute_dLdE()
        return self._dLdE

    @dLdE.setter
    def dLdE(self, dLdE):
        self._dLdE = dLdE

        self.dFdE = None

    @property
    def dFdE(self):
        if self._dFdE is None:
            self._compute_dFdE()
        return self._dFdE

    @dFdE.setter
    def dFdE(self, dFdE):
        self._dFdE = dFdE

        self.gFgE = None
        self.phiE = None

    @property
    def gFgE(self):
        if self._gFgE is None:
            self._compute_gFgE()
        return self._gFgE

    @gFgE.setter
    def gFgE(self, gFgE):
        self._gFgE = gFgE

        self.phiE = None

    @property
    def phiE(self):
        if self._phiE is None:
            self._compute_phiE()
        return self._phiE

    @phiE.setter
    def phiE(self, phiE):
        self._phiE = phiE

    def _compute_C(self):
        if self.a_mode == 'WX':
            self.C = self.A
        elif self.a_mode == 'MX':
            self.C = self.B.T @ self.A @ self.B
        elif self.a_mode == 'MXX':
            self.C = self.A @ self.B.T
        elif self.a_mode == 'MG':
            self.C = self.A @ self.B.T

    def _compute_R(self):
        self.R = self.r * .5 * np.dot(self.C.T.ravel(), self.C.ravel())

    def _compute_S(self):
        self.S = self.s * .5 * np.sum(np.square(self.E - 1))

    def _compute_P(self):
        if self.a_mode == 'WX':
            self.P = self.B @ self.C @ self.B.T
        elif self.a_mode == 'MX':
            self.P = self.B @ self.C @ self.B.T
        elif self.a_mode == 'MXX':
            self.P = self.B @ self.C
        elif self.a_mode == 'MG':
            self.P = self.B @ self.C

    def _compute_D(self):
        self.D = self.P.diagonal().reshape(-1, 1) + self.P.diagonal().reshape(1, -1) - 2 * self.P

    def _compute_I(self):
        self.I = self.T * (self.D - self.E.reshape(-1, 1))

    def _compute_O(self):
        self.O = np.sum(self.inner.func(self.I), axis=1) - self.N

    def _compute_L(self):
        self.L = self.l * np.sum(self.outer.func(self.O))

    def _compute_F(self):
        self.F = self.R + self.S + self.L
        self.F_count += 1

    def _compute_V(self):
        self.V = self.l * self.outer.grad(self.O).reshape(-1, 1) * self.inner.grad(self.I) * self.T

    def _compute_dRdA(self):
        if self.a_mode == 'WX':
            self.dRdA = self.r * self.A
        elif self.a_mode == 'MX':
            self.dRdA = self.r * self.A
        elif self.a_mode == 'MXX':
            self.dRdA = self.r * self.A
        elif self.a_mode == 'MG':
            self.dRdA = self.r * self.P

    def _compute_dLdA(self):
        Z = self.V + self.V.T
        U = np.diag(np.sum(Z, axis=0)) - Z

        if self.a_mode == 'WX':
            self.dLdA = self.B.T @ U @ self.B
        elif self.a_mode == 'MX':
            self.dLdA = U
        elif self.a_mode == 'MXX':
            self.dLdA = U
        elif self.a_mode == 'MG':
            self.dLdA = self.B.T @ U @ self.B

    def _compute_dFdA(self):
        self.dFdA = self.dRdA + self.dLdA
        self.dFdA_count += 1

    def _compute_gFgA(self):
        if self.a_mode == 'WX':
            self.gFgA = self.dFdA
        elif self.a_mode == 'MX':
            self.gFgA = self.B @ self.B.T @ self.dFdA @ self.B @ self.B.T
        elif self.a_mode == 'MXX':
            self.gFgA = self.B.T @ self.dFdA @ self.B
        elif self.a_mode == 'MG':
            self.gFgA = self.dFdA

    def _compute_phiA(self):
        self.phiA = -np.dot(self.dFdA.ravel(), self.gFgA.ravel())

    def _compute_dSdE(self):
        self.dSdE = self.s * (self.E - 1)

    def _compute_dLdE(self):
        if self.e_mode == 'single':
            self.dLdE = -np.sum(self.V)
        elif self.e_mode == 'multiple':
            self.dLdE = -np.sum(self.V, axis=1)

    def _compute_dFdE(self):
        self.dFdE = self.dSdE + self.dLdE
        self.dFdE_count += 1

    def _compute_gFgE(self):
        self.gFgE = self.dFdE

    def _compute_phiE(self):
        self.phiE = -np.dot(self.dFdE, self.gFgE)

    def take_step(self, arguments='AE', alpha_0=None, verbose=None):
        if alpha_0 is None:
            alpha_0 = self.alpha_0

        if verbose is None:
            verbose = self.verbose_line_search

        A_prev = self.A
        E_prev = self.E
        F_prev = self.F
        arguments_prev = self.arguments
        phi_prev = self.phi
        alpha_prev = self.alpha

        # Compute phi, the gradient of F with respect to the step size, alpha.
        phi = 0
        if 'A' in arguments:
            dA = self.dFdA
            phi += self.phiA
        else:
            dA = None
        if 'E' in arguments:
            dE = self.dFdE
            phi += self.phiE
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
        if (dA is not None) and (not np.all(dA == 0)):
            self.A = A_prev - alpha * dA
        if (dE is not None) and (not np.all(dE == 0)):
            self.E = E_prev - alpha * dE

        # If Armijo's condition for sufficient decrease has been satisfied, the search is complete.
        if self.F <= F_prev + self.armijo * alpha * phi:
            self.arguments = arguments
            self.phi = phi
            self.alpha = alpha
            self.backtracks = backtracks
            self.termination = None
            self.steps += 1
            return True

        # If the maximum number of backtracks have been performed, return without taking a step.
        if backtracks == self.max_backtracks:
            self.A = A_prev
            self.E = E_prev
            self.F = F_prev
            self.termination = 'max_backtracks'
            return False

        # Decrease alpha with a quadratic approximation of the minimizer, by interpolating F_prev, phi, and self.F.
        alpha_old = alpha
        alpha = - .5 * phi * alpha ** 2 / (self.F - F_prev - phi * alpha)

        # Safeguard the interpolation by making sure alpha is not too small or too close to alpha_old.
        rho = alpha / alpha_old
        if rho < self.rho_lo or rho > self.rho_hi:
            rho = .5
            alpha = rho * alpha_old

        # Take a step in the direction of steepest descent.
        F_old = self.F
        if (dA is not None) and (not np.all(dA == 0)):
            self.A = A_prev - alpha * dA
        if (dE is not None) and (not np.all(dE == 0)):
            self.E = E_prev - alpha * dE

        backtracks += 1

        # If Armijo's condition for sufficient decrease has been satisfied, the search is complete.
        if self.F <= F_prev + self.armijo * alpha * phi:
            self.arguments = arguments
            self.phi = phi
            self.alpha = alpha
            self.backtracks = backtracks
            self.termination = None
            self.steps += 1
            return True

        # If the maximum number of backtracks have been performed, return without taking a step.
        if backtracks == self.max_backtracks:
            self.A = A_prev
            self.E = E_prev
            self.F = F_prev
            self.termination = 'max_backtracks'
            return False

        while True:
            # Decrease alpha with a cubic approximation of the minimizer, by interpolating F_prev, phi, self.F, and F_old.
            ab = (np.array([[alpha_old ** 2, - alpha ** 2], [- alpha_old ** 3, alpha ** 3]]) @
                  np.array([[self.F - F_prev - phi * alpha], [F_old - F_prev - phi * alpha_old]]) /
                  (alpha_old ** 2 * alpha ** 2 * (alpha - alpha_old)))
            a = ab[0].item()
            b = ab[1].item()
            alpha_old = alpha
            alpha = (- b + (b ** 2 - 3 * a * phi) ** .5) / (3 * a)

            # Safeguard the interpolation by making sure alpha is not too small or too close to alpha_old.
            rho = alpha / alpha_old
            if rho < self.rho_lo or rho > self.rho_hi:
                rho = .5
                alpha = rho * alpha_old

            # Take a step in the direction of steepest descent.
            F_old = self.F
            if (dA is not None) and (not np.all(dA == 0)):
                self.A = A_prev - alpha * dA
            if (dE is not None) and (not np.all(dE == 0)):
                self.E = E_prev - alpha * dE

            backtracks += 1

            # If Armijo's condition for sufficient decrease has been satisfied, the search is complete.
            if self.F <= F_prev + self.armijo * alpha * phi:
                self.arguments = arguments
                self.phi = phi
                self.alpha = alpha
                self.backtracks = backtracks
                self.termination = None
                self.steps += 1
                return True

            # If the maximum number of backtracks have been performed, return without taking a step.
            if backtracks == self.max_backtracks:
                self.A = A_prev
                self.E = E_prev
                self.F = F_prev
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

        self.F_0 = F_prev = self.F

        while self.take_step(arguments, alpha_0):
            self.delta_F = F_prev - self.F
            F_prev = self.F

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

        self.F_0 = F_prev = self.F

        while True:
            if self.take_step(arguments, arg_alpha_0[arguments]):
                arg_alpha_0[arguments] = self.alpha
                arg_steps += 1

                self.delta_F = F_prev - self.F
                F_prev = self.F

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

    def _print_optimize_header(self):
        steps = f"{'step':^4s}"
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

        print(" ".join((steps, arguments, backtracks, alpha, phi, delta_F, F, R, S, L, mean_E)))

    def _print_optimize_row(self):
        steps = f"{self.steps:4d}" if self.steps is not None else f"{'-':^4s}"
        arguments = f"{self.arguments:^4s}" if self.arguments is not None else f"{'-':^4s}"
        backtracks = f"{self.backtracks:4d}" if self.backtracks is not None else f"{'-':^4s}"
        alpha = f"{self.alpha:10.3e}" if self.alpha is not None else f"{'-':^10s}"
        phi = f"{self.phi:10.3e}" if self.phi is not None else f"{'-':^10s}"
        delta_F = f"{self.delta_F:10.3e}" if self.delta_F is not None else f"{'-':^10s}"
        F = f"{self.F:10.3e}" if self.F is not None else f"{'-':^10s}"
        R = f"{self.R:10.3e}" if self.R is not None else f"{'-':^10s}"
        S = f"{self.S:10.3e}" if self.S is not None else f"{'-':^10s}"
        L = f"{self.L:10.3e}" if self.L is not None else f"{'-':^10s}"
        mean_E = f"{np.mean(self.E):10.3e}" if self.E is not None else f"{'-':^10s}"

        print(" ".join((steps, arguments, backtracks, alpha, phi, delta_F, F, R, S, L, mean_E)))

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
        print(f"         F = {self.F:f}")
        print(f"   delta_F = {self.delta_F:e}")
        print(f"     steps = {self.steps:d}")
        print(f"  run_time = {self.run_time:f} seconds")
        print("")
        print(f" F function calls: {self.F_count:d}")
        print(f"dA function calls: {self.dFdA_count:d}")
        print(f"dE function calls: {self.dFdE_count:d}")
        print("")

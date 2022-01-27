import time
import numpy as np
from sklearn.decomposition import PCA, KernelPCA

from loss import ReLU


class MLNN:
    def __init__(self, B, T, N, C=None, A_0=None, E_0=None, mlnn_params=None, line_search_params=None, optimize_params=None):
        self.r = 0
        self.s = 0
        self.l = 1
        self.q = 1
        self.inner = ReLU(1)
        self.outer = ReLU(1)
        self.k_mode = None
        self.a_mode = 'full'
        self.e_mode = 'single'
        self.i_mode = None
        self.keep_a_psd = None
        self.keep_a_centered = None
        self.keep_e_positive = None
        self.d = 2

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

        if self.k_mode is None:
            if C is None:
                self.k_mode = 'linear'
            else:
                self.k_mode = 'nonlinear'
        if self.i_mode is None:
            if self.a_mode == 'full' or self.a_mode == 'diagonal':
                self.i_mode = 'zero'
            elif self.a_mode == 'decomposed':
                self.i_mode = 'pca'

        if self.a_mode == 'full':
            assert self.i_mode == 'zero' or self.i_mode == 'random' or self.i_mode == 'identity' or self.i_mode == 'centered'
        elif self.a_mode == 'diagonal':
            assert self.i_mode == 'zero' or self.i_mode == 'random' or self.i_mode == 'identity'
        elif self.a_mode == 'decomposed':
            assert self.i_mode == 'random' or self.i_mode == 'pca'

        if self.keep_a_psd is None:
            self.keep_a_psd = True
        if self.keep_a_centered is None:
            if self.k_mode == 'nonlinear' and self.a_mode == 'full' and self.keep_a_psd:
                self.keep_a_centered = True
            else:
                self.keep_a_centered = False
        if self.keep_e_positive is None:
            self.keep_e_positive = self.keep_a_psd

        assert self.r >= 0
        assert self.s >= 0
        assert self.l > 0
        assert self.q > 0

        self.B = B
        self.T = T
        self.N = self.q * N

        assert self.n > 1
        assert self.m > 0

        assert self.T.shape[0] == self.n
        assert self.T.shape[1] == self.n
        assert self.N.shape[0] == self.n
        assert self.N.shape[1] == 1

        if self.k_mode == 'linear':
            assert C is None
        elif self.k_mode == 'nonlinear':
            if C is None:
                self.C = self.B
            else:
                self.C = C

            assert self.C.shape[0] == self.m
            assert np.array_equal(self.C, self.C.T)

        if A_0 is not None:
            self.A_0 = A_0
            if self.a_mode == 'decompose':
                self.d = self.A_0.shape[0]

        if self.a_mode == 'full':
            assert self.A_0.shape[0] == self.m
            assert np.array_equal(self.A_0, self.A_0.T)
        elif self.a_mode == 'diagonal':
            assert self.A_0.shape[0] == self.m
            assert self.A_0.shape[1] == 1
        elif self.a_mode == 'decomposed':
            assert self.A_0.shape[0] == self.d
            assert self.A_0.shape[1] == self.m

        if E_0 is not None:
            self.E_0 = E_0

        if self.e_mode == 'single':
            assert np.isscalar(self.E_0)
        elif self.e_mode == 'multiple':
            assert self.E_0.shape[0] == self.n
            assert self.E_0.shape[1] == 1

    def apply_params(self, params):
        for attr in params:
            setattr(self, attr, params[attr])

    def initialize(self):
        self.time_0 = time.perf_counter()
        self.run_time = None

        self.steps = 0
        self.F_count = 0
        self.dFdA_count = 0
        self.dFdE_count = 0
        self.eigh_count = 0

        self.F_0 = None
        self.delta_F = None

        self.arguments = None
        self.phi = None
        self.alpha = None
        self.backtracks = None
        self.termination = None

        self.A = self.A_0
        self.E = self.E_0

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
        self.n = None
        self.m = None
        self.I = None

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, T):
        self._T = T
        self.Q = None
        self.I = None

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, N):
        self._N = N
        self.I = None

    @property
    def C(self):
        return self._C

    @C.setter
    def C(self, C):
        self._C = C
        self.J = None

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, A):
        self._A = A
        self.I = None
        self.J = None
        self.A_is_psd = None
        self.eigenvalues = None
        self.eigenvectors = None

        if self.keep_a_psd and not self.A_is_psd:
            self._A = self.A_psd_projection()
            self.A_is_psd = True

        if self.keep_a_centered:
            self._A = self.A_center_projection()

    @property
    def E(self):
        return self._E

    @E.setter
    def E(self, E):
        self._E = E
        self.I = None
        if self.s:
            self.S = None
            self.dSdE = None

        if self.keep_e_positive:
            self._E = self.E_positive_projection()

    @property
    def n(self):
        if self._n is None:
            self._compute_n()
        return self._n

    @n.setter
    def n(self, n):
        self._n = n
        self.E_0 = None

    @property
    def m(self):
        if self._m is None:
            self._compute_m()
        return self._m

    @m.setter
    def m(self, m):
        self._m = m
        self.A_0 = None

    @property
    def Q(self):
        if self._Q is None:
            self._compute_Q()
        return self._Q

    @Q.setter
    def Q(self, Q):
        self._Q = Q
        self.O = None

    @property
    def A_0(self):
        if self._A_0 is None:
            self._compute_A_0()
        return self._A_0

    @A_0.setter
    def A_0(self, A_0):
        self._A_0 = A_0

    @property
    def E_0(self):
        if self._E_0 is None:
            self._compute_E_0()
        return self._E_0

    @E_0.setter
    def E_0(self, E_0):
        self._E_0 = E_0

    @property
    def J(self):
        if self._J is None:
            self._compute_J()
        return self._J

    @J.setter
    def J(self, J):
        self._J = J
        self.K = None

    @property
    def K(self):
        if self._K is None:
            self._compute_K()
        return self._K

    @K.setter
    def K(self, K):
        self._K = K
        if self.r:
            self.R = None
            self.dRdA = None

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
    def I(self):
        if self._I is None:
            self._compute_I()
        return self._I

    @I.setter
    def I(self, I):
        self._I = I
        self.O = None

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
        self.subset_active_rows = None
        self.subset_active_cols = None
        self.subset_active_data = None

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
    def subset_active_data(self):
        if self._subset_active_data is None:
            self._compute_V()
        return self._subset_active_data

    @subset_active_data.setter
    def subset_active_data(self, subset_active_data):
        self._subset_active_data = subset_active_data
        self.dLdA = None
        self.dLdE = None

    @property
    def subset_active_rows(self):
        if self._subset_active_rows is None:
            self._compute_V()
        return self._subset_active_rows

    @subset_active_rows.setter
    def subset_active_rows(self, subset_active_rows):
        self._subset_active_rows = subset_active_rows

    @property
    def subset_active_cols(self):
        if self._subset_active_cols is None:
            self._compute_V()
        return self._subset_active_cols

    @subset_active_cols.setter
    def subset_active_cols(self, subset_active_cols):
        self._subset_active_cols = subset_active_cols

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
        self.phiE = None

    @property
    def phiE(self):
        if self._phiE is None:
            self._compute_phiE()
        return self._phiE

    @phiE.setter
    def phiE(self, phiE):
        self._phiE = phiE

    @property
    def A_is_psd(self):
        if self._A_is_psd is None:
            self._compute_A_is_psd()
        return self._A_is_psd

    @A_is_psd.setter
    def A_is_psd(self, A_is_psd):
        self._A_is_psd = A_is_psd

    @property
    def eigenvalues(self):
        if self._eigenvalues is None:
            self._compute_eigh()
        return self._eigenvalues

    @eigenvalues.setter
    def eigenvalues(self, eigenvalues):
        self._eigenvalues = eigenvalues

    @property
    def eigenvectors(self):
        if self._eigenvectors is None:
            self._compute_eigh()
        return self._eigenvectors

    @eigenvectors.setter
    def eigenvectors(self, eigenvectors):
        self._eigenvectors = eigenvectors

    def _compute_n(self):
        self.n = self.B.shape[0]

    def _compute_m(self):
        self.m = self.B.shape[1]

    def _compute_Q(self):
        self.Q = np.where(self.T == 1, self.q, 1)

    def _compute_A_0(self):
        if self.i_mode == 'random':
            rng = np.random.Generator(np.random.PCG64(12345))

        if self.a_mode == 'full':
            if self.i_mode == 'zero':
                A = np.zeros((self.m, self.m))
            else:
                if self.i_mode == 'random':
                    A = rng.standard_normal(self.m * self.m).reshape(self.m, self.m) / self.m ** .5
                    A = A.T @ A
                elif self.i_mode == 'identity':
                    A = np.diag(np.ones(self.m) / self.m ** .5)
                elif self.i_mode == 'centered':
                    U = np.identity(self.n) - 1 / self.n
                    A = self.B.T @ U @ self.B
                    A = (A + A.T) / 2

                if self.k_mode == 'linear':
                    K = A
                elif self.k_mode == 'nonlinear':
                    K = A @ self.C
                A /= np.dot(K.T.ravel(), K.ravel()) ** .5
        elif self.a_mode == 'diagonal':
            if self.i_mode == 'zero':
                A = np.zeros(self.m).reshape(self.m, 1)
            else:
                if self.i_mode == 'random':
                    A = rng.standard_normal(self.m).reshape(self.m, 1) ** 2
                elif self.i_mode == 'identity':
                    A = np.ones(self.m).reshape(self.m, 1) / self.m ** .5

                if self.k_mode == 'linear':
                    K = A
                elif self.k_mode == 'nonlinear':
                    K = A * self.C
                A /= np.dot(K.T.ravel(), K.ravel()) ** .5
        elif self.a_mode == 'decomposed':
            if self.i_mode == 'random':
                A = rng.standard_normal(self.d * self.m).reshape(self.d, self.m) / self.d ** .5
            elif self.i_mode == 'pca':
                if self.k_mode == 'linear':
                    pca = PCA(n_components=self.d)
                    pca.fit(self.B)
                    A = pca.components_ / self.d ** .5
                elif self.k_mode == 'nonlinear':
                    kpca = KernelPCA(n_components=self.d, kernel='precomputed')
                    kpca.fit(self.C)
                    A = kpca.eigenvectors_.T / self.d ** .5

            if self.k_mode == 'linear':
                K = A @ A.T
            elif self.k_mode == 'nonlinear':
                K = A @ self.C @ A.T
            A /= np.dot(K.T.ravel(), K.ravel()) ** .25

        self.A_0 = A

    def _compute_E_0(self):
        if self.i_mode == 'random':
            rng = np.random.Generator(np.random.PCG64(12345))

        if self.e_mode == 'single':
            if self.i_mode == 'zero':
                E = 0
            elif self.i_mode == 'random':
                E = rng.standard_normal(1).item() ** 2
            elif self.i_mode == 'centered' or self.i_mode == 'identity' or self.i_mode == 'pca':
                E = 1
        elif self.e_mode == 'multiple':
            if self.i_mode == 'zero':
                E = np.zeros(self.n).reshape(self.n, 1)
            elif self.i_mode == 'random':
                E = rng.standard_normal(self.n).reshape(self.n, 1) ** 2
            elif self.i_mode == 'centered' or self.i_mode == 'identity' or self.i_mode == 'pca':
                E = np.ones(self.n).reshape(self.n, 1)

        self.E_0 = E

    def _compute_J(self):
        if self.k_mode == 'linear':
            self.J = self.A
        elif self.k_mode == 'nonlinear':
            if self.a_mode == 'full' or self.a_mode == 'decomposed':
                self.J = self.A @ self.C
            elif self.a_mode == 'diagonal':
                self.J = self.A * self.C

    def _compute_K(self):
        if self.a_mode == 'full' or self.a_mode == 'diagonal':
            self.K = self.J
        elif self.a_mode == 'decomposed':
            self.K = self.A @ self.J.T

    def _compute_R(self):
        self.R = self.r * .5 * np.dot(self.K.T.ravel(), self.K.ravel())

    def _compute_S(self):
        self.S = self.s * .5 * np.sum(np.square(self.E - 1))

    def _compute_I(self):
        if self.a_mode == 'full':
            P = self.B @ self.A @ self.B.T
        elif self.a_mode == 'diagonal':
            P = self.B @ (self.A * self.B.T)
        elif self.a_mode == 'decomposed':
            Z = self.A @ self.B.T
            P = Z.T @ Z

        self.I = self.T * ((P.diagonal().reshape(-1, 1) + P.diagonal().reshape(1, -1) - 2 * P) - self.E)

    def _compute_O(self):
        if self.q == 1:
            self.O = np.sum(self.inner.func(self.I), axis=1, keepdims=True) - self.N
        else:
            self.O = np.sum(self.Q * self.inner.func(self.I), axis=1, keepdims=True) - self.N

    def _compute_L(self):
        self.L = self.l * np.sum(self.outer.func(self.O))

    def _compute_F(self):
        self.F = self.R + self.S + self.L
        self.F_count += 1

    def _compute_V(self):
        if self.q == 1:
            V = self.l * self.outer.grad(self.O) * self.inner.grad(self.I) * self.T
        else:
            V = self.l * self.outer.grad(self.O) * self.inner.grad(self.I) * self.Q * self.T
        is_active_row = np.any(V, axis=1)
        is_active_col = np.any(V, axis=0)
        is_active = np.logical_or(is_active_row, is_active_col)
        self.subset_active_rows = np.argwhere(is_active_row).flatten()
        self.subset_active_cols = np.argwhere(is_active_col).flatten()
        self.subset_active_data = np.argwhere(is_active).flatten()
        self.V = V.take(self.subset_active_data, axis=0).take(self.subset_active_data, axis=1)

    def _compute_dRdA(self):
        if self.a_mode == 'full':
            if self.k_mode == 'linear':
                self.dRdA = self.r * self.K
            elif self.k_mode == 'nonlinear':
                self.dRdA = self.r * self.C @ self.K
        elif self.a_mode == 'diagonal':
            if self.k_mode == 'linear':
                self.dRdA = self.r * self.K
            elif self.k_mode == 'nonlinear':
                self.dRdA = self.r * np.sum(self.C * self.K, axis=1, keepdims=True)
        elif self.a_mode == 'decomposed':
            self.dRdA = self.r * 2 * self.K @ self.J

    def _compute_dLdA(self):
        if self.subset_active_data.size:
            Z = self.V + self.V.T
            U = np.diag(np.sum(Z, axis=0)) - Z
            B = self.B.take(self.subset_active_data, axis=0)

            if self.a_mode == 'full':
                self.dLdA = B.T @ U @ B
            elif self.a_mode == 'diagonal':
                self.dLdA = np.sum(B.T * (U @ B).T, axis=1, keepdims=True)
            elif self.a_mode == 'decomposed':
                self.dLdA = 2 * ((self.A @ B.T) @ U @ B)
        else:
            self.dLdA = np.zeros(1)

    def _compute_dFdA(self):
        self.dFdA = self.dRdA + self.dLdA
        self.dFdA_count += 1

    def _compute_phiA(self):
        self.phiA = -np.dot(self.dFdA.ravel(), self.dFdA.ravel())

    def _compute_dSdE(self):
        self.dSdE = self.s * (self.E - 1)

    def _compute_dLdE(self):
        if self.subset_active_data.size:
            if self.e_mode == 'single':
                self.dLdE = -np.sum(self.V, keepdims=True)
            elif self.e_mode == 'multiple':
                self.dLdE = np.zeros(self.n).reshape(self.n, 1)
                self.dLdE[self.subset_active_data] = -np.sum(self.V, axis=1, keepdims=True)
        else:
            self.dLdE = np.zeros(1)

    def _compute_dFdE(self):
        self.dFdE = self.dSdE + self.dLdE
        self.dFdE_count += 1

    def _compute_phiE(self):
        self.phiE = -np.dot(self.dFdE.ravel(), self.dFdE.ravel())

    def _compute_A_is_psd(self):
        if self.a_mode == 'full':
            try:
                np.linalg.cholesky(self.A)
                self.A_is_psd = True
            except np.linalg.LinAlgError:
                self.A_is_psd = False
        elif self.a_mode == 'diagonal':
            if np.all(self.A >= 0):
                self.A_is_psd = True
            else:
                self.A_is_psd = False
        elif self.a_mode == 'decomposed':
            self.A_is_psd = True

    def _compute_eigh(self):
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.A)
        self.eigh_count += 1

    def A_psd_projection(self, tol=1e-10):
        if self.a_mode == 'full':
            if self.eigenvalues[-1] > tol:
                i = np.argmax(self.eigenvalues > tol)
                return (self.eigenvectors[:, i:] * self.eigenvalues[i:]) @ self.eigenvectors[:, i:].T
            else:
                return np.zeros(self.A.shape)
        elif self.a_mode == 'diagonal':
            return np.maximum(self.A, 0)
        elif self.a_mode == 'decomposed':
            return self.A

    def A_center_projection(self):
        A = self.A
        if self.a_mode == 'full':
            A -= np.sum(A, axis=0, keepdims=True) / self.m
            A -= np.sum(A, axis=1, keepdims=True) / self.m
        elif self.a_mode == 'decomposed':
            A -= np.sum(A, axis=1, keepdims=True) / self.m

        return A

    def E_positive_projection(self):
        return np.maximum(self.E, 0)

    def update_A(self, A_prev, alpha, dA):
        self.A = A_prev - alpha * dA

    def update_E(self, E_prev, alpha, dE):
        self.E = E_prev - alpha * dE

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
        if 'A' in arguments and self.phiA:
            dA = self.dFdA
            phi += self.phiA
        else:
            dA = None
        if 'E' in arguments and self.phiE:
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
        if dA is not None:
            self.update_A(A_prev, alpha, dA)
        if dE is not None:
            self.update_E(E_prev, alpha, dE)

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
        if dA is not None:
            self.update_A(A_prev, alpha, dA)
        if dE is not None:
            self.update_E(E_prev, alpha, dE)

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
            F_old = self.F
            if dA is not None:
                self.update_A(A_prev, alpha, dA)
            if dE is not None:
                self.update_E(E_prev, alpha, dE)

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
        F = f"{self.F:10.3e}" if self.F is not None else f"{'-':^10s}"
        R = f"{self.R:10.3e}" if self.R is not None else f"{'-':^10s}"
        S = f"{self.S:10.3e}" if self.S is not None else f"{'-':^10s}"
        L = f"{self.L:10.3e}" if self.L is not None else f"{'-':^10s}"
        mean_E = f"{np.mean(self.E):10.3e}" if self.E is not None else f"{'-':^10s}"
        actv_rows = f"{self.subset_active_rows.size:9d}" if self.subset_active_rows.size is not None else f"{'-':^9s}"
        actv_cols = f"{self.subset_active_cols.size:9d}" if self.subset_active_cols.size is not None else f"{'-':^9s}"
        actv_data = f"{self.subset_active_data.size:9d}" if self.subset_active_data.size is not None else f"{'-':^9s}"

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
        print(f"         F = {self.F:f}")
        print(f"   delta_F = {self.delta_F:e}")
        print(f"     steps = {self.steps:d}")
        print(f"  run_time = {self.run_time:f} seconds")
        print("")
        print(f"   F function calls: {self.F_count:d}")
        print(f"  dA function calls: {self.dFdA_count:d}")
        print(f"  dE function calls: {self.dFdE_count:d}")
        print(f"eigh function calls: {self.eigh_count:d}")
        print("")

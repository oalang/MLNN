import numpy as np
from sklearn.decomposition import PCA, KernelPCA

from loss import ReLU


class MLNNEngine:
    def __init__(self, B, T, N, C=None, mlnn_params=None):
        self.r = 0
        self.s = 0
        self.l = 1
        self.q = 1
        self.inner_loss = ReLU(1)
        self.outer_loss = ReLU(1)
        self.k_mode = None
        self.a_mode = 'full'
        self.e_mode = 'single'
        self.keep_a_psd = True
        self.keep_a_centered = False
        self.keep_e_positive = None

        if mlnn_params:
            apply_params(self, mlnn_params)

        if self.k_mode is None:
            if C is None:
                self.k_mode = 'linear'
            else:
                self.k_mode = 'nonlinear'

        if self.keep_e_positive is None:
            self.keep_e_positive = self.keep_a_psd

        assert self.r >= 0
        assert self.s >= 0
        assert self.l > 0
        assert self.q > 0

        self.B = B
        self.T = T
        self.N = self.q * N

        self.n = self.B.shape[0]
        self.m = self.B.shape[1]

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

        if not self.r:
            self.R = 0
            self.dRdA = 0
        if not self.s:
            self.S = 0
            self.dSdE = 0

        self.F_count = 0
        self.dFdA_count = 0
        self.dFdE_count = 0
        self.eigh_count = 0

    @property
    def B(self):
        return self._B

    @B.setter
    def B(self, B):
        self._B = B
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
    def Q(self):
        if self._Q is None:
            self._compute_Q()
        return self._Q

    @Q.setter
    def Q(self, Q):
        self._Q = Q
        self.O = None

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

    def _compute_Q(self):
        self.Q = np.where(self.T == 1, self.q, 1)

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
            self.O = np.sum(self.inner_loss.func(self.I), axis=1, keepdims=True)
        else:
            self.O = np.sum(self.Q * self.inner_loss.func(self.I), axis=1, keepdims=True)

        if self.outer_loss is not None:
            self.O -= self.N

    def _compute_L(self):
        if self.outer_loss is not None:
            self.L = self.l * np.sum(self.outer_loss.func(self.O))
        else:
            self.L = self.l * np.sum(self.O)

    def _compute_F(self):
        self.F = self.R + self.S + self.L
        self.F_count += 1

    def _compute_V(self):
        V = self.l * self.inner_loss.grad(self.I) * self.T
        if self.outer_loss is not None:
            V *= self.outer_loss.grad(self.O)
        if self.q != 1:
            V *= self.Q
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

    def update_A(self, A, alpha, dA):
        self.A = A - alpha * dA

    def update_E(self, E, alpha, dE):
        self.E = E - alpha * dE

    def compute_A_0(self, i_mode='random', d=None):
        if i_mode == 'random':
            rng = np.random.Generator(np.random.PCG64(12345))
            
        if self.a_mode == 'full':
            if i_mode == 'zero':
                A = np.zeros((self.m, self.m))
            else:
                if i_mode == 'random':
                    A = rng.standard_normal(self.m * self.m).reshape(self.m, self.m) / self.m ** .5
                    A = A.T @ A
                elif i_mode == 'identity':
                    A = np.diag(np.ones(self.m) / self.m ** .5)
                elif i_mode == 'centered':
                    U = np.identity(self.n) - 1 / self.n
                    A = self.B.T @ U @ self.B
                    A = (A + A.T) / 2

                if self.k_mode == 'linear':
                    K = A
                elif self.k_mode == 'nonlinear':
                    K = A @ self.C
                A /= np.dot(K.T.ravel(), K.ravel()) ** .5
        elif self.a_mode == 'diagonal':
            if i_mode == 'zero':
                A = np.zeros(self.m).reshape(self.m, 1)
            else:
                if i_mode == 'random':
                    A = rng.standard_normal(self.m).reshape(self.m, 1) ** 2
                elif i_mode == 'identity':
                    A = np.ones(self.m).reshape(self.m, 1) / self.m ** .5

                if self.k_mode == 'linear':
                    K = A
                elif self.k_mode == 'nonlinear':
                    K = A * self.C
                A /= np.dot(K.T.ravel(), K.ravel()) ** .5
        elif self.a_mode == 'decomposed':
            if d is None:
                d = self.m
            
            if i_mode == 'random':
                A = rng.standard_normal(d * self.m).reshape(d, self.m) / d ** .5
            elif i_mode == 'pca':
                if self.k_mode == 'linear':
                    pca = PCA(n_components=d)
                    pca.fit(self.B)
                    A = pca.components_ / d ** .5
                elif self.k_mode == 'nonlinear':
                    kpca = KernelPCA(n_components=d, kernel='precomputed')
                    kpca.fit(self.C)
                    A = kpca.eigenvectors_.T / d ** .5

            if self.k_mode == 'linear':
                K = A @ A.T
            elif self.k_mode == 'nonlinear':
                K = A @ self.C @ A.T
            A /= np.dot(K.T.ravel(), K.ravel()) ** .25

        return A

    def compute_E_0(self, i_mode='random'):
        if i_mode == 'random':
            rng = np.random.Generator(np.random.PCG64(12345))

        if self.e_mode == 'single':
            if i_mode == 'zero':
                E = 0
            elif i_mode == 'random':
                E = rng.standard_normal(1).item() ** 2
            elif i_mode == 'centered' or i_mode == 'identity' or i_mode == 'pca':
                E = 1
        elif self.e_mode == 'multiple':
            if i_mode == 'zero':
                E = np.zeros(self.n).reshape(self.n, 1)
            elif i_mode == 'random':
                E = rng.standard_normal(self.n).reshape(self.n, 1) ** 2
            elif i_mode == 'centered' or i_mode == 'identity' or i_mode == 'pca':
                E = np.ones(self.n).reshape(self.n, 1)

        return E

    def fun(self, x, arguments='AE'):
        i = 0
        if 'A' in arguments:
            A = x[0:self.A.size].reshape(self.A.shape)
            i = self.A.size
            if not np.array_equal(A, self.A):
                self.A = A
        if 'E' in arguments:
            if self.e_mode == 'single':
                E = x[i:]
            elif self.e_mode == 'multiple':
                E = x[i:].reshape(self.E.shape)
            if not np.array_equal(E, self.E):
                self.E = E

        return self.F

    def jac(self, x, arguments='AE'):
        i = 0
        if 'A' in arguments:
            A = x[0:self.A.size].reshape(self.A.shape)
            i = self.A.size

            if not np.array_equal(A, self.A):
                self.A = A
        if 'E' in arguments:
            if self.e_mode == 'single':
                E = x[i:]
            elif self.e_mode == 'multiple':
                E = x[i:].reshape(self.E.shape)

            if not np.array_equal(E, self.E):
                self.E = E

        jac = np.empty(0)

        if 'A' in arguments:
            jac = np.append(jac, self.dFdA)

        if 'E' in arguments:
            jac = np.append(jac, self.dFdE)

        return jac


def apply_params(obj, params):
    for attr in params:
        if hasattr(obj, attr):
            setattr(obj, attr, params[attr])


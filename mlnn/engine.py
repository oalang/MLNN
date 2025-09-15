import numpy as np
from sklearn.decomposition import PCA, KernelPCA

from mlnn.activation import get_activation


def compute_T(Y):
    return np.where(np.equal.outer(Y, Y), 1, -1)


def compute_N(T, q):
    return q * (np.sum(T == 1, axis=1, keepdims=True) - 1)


def compute_Q(T, q):
    return np.where(T == 1, q, 1)


# @partial(jax.jit, static_argnames=['x_mode', 'a_mode'])
def compute_J(Z, A, x_mode, a_mode):
    if x_mode == 'raw':
        return A
    elif x_mode == 'kernel':
        if a_mode in ('full', 'decomposed'):
            return A @ Z
        elif a_mode == 'diagonal':
            return A * Z


# @partial(jax.jit, static_argnames=['a_mode'])
def compute_K(A, J, a_mode):
    if a_mode in ('full', 'diagonal'):
        return J
    elif a_mode == 'decomposed':
        return A @ J.T


# @partial(jax.jit, static_argnames=['r'])
def compute_R(K, r):
    return r * 0.5 * np.vdot(K, K)


# @partial(jax.jit, static_argnames=['s'])
def compute_S(E, s):
    return s * 0.5 * np.sum(np.squared(E - 1.0))


# @partial(jax.jit, static_argnames=['a_mode', 'Z_equals_X'])
def compute_D(X, A, J, a_mode, Z_equals_X):
    if a_mode == 'full':
        if Z_equals_X:
            P = X @ J
        else:
            P = X @ A @ X.T
    elif a_mode == 'diagonal':
        if Z_equals_X:
            P = X @ J
        else:
            P = X @ (A * X.T)
    elif a_mode == 'decomposed':
        if Z_equals_X:
            P = J.T @ J
        else:
            B = A @ X.T
            P = B.T @ B
    C = P.diagonal()
    return np.add.outer(C, C) - 2 * P


# @partial(jax.jit, static_argnames=['inner_loss_intr'])
def compute_I(E, T, D, inner_loss_intr):
    return inner_loss_intr((D - E) * T)


# @partial(jax.jit, static_argnames=['q', 'inner_loss_func_intr', 'outer_loss_intr'])
def compute_O(N, Q, I, q, inner_loss_func_intr, outer_loss_intr=None):
    O = inner_loss_func_intr(I)
    if q != 1:
        O *= Q
    if outer_loss_intr is not None:
        O = outer_loss_intr(np.sum(O, axis=1, keepdims=True) - N)
    return O


# @partial(jax.jit, static_argnames=['l', 'outer_loss_func_intr'])
def compute_L(O, l, outer_loss_func_intr=None):
    if outer_loss_func_intr is not None:
        return l * np.sum(outer_loss_func_intr(O))
    else:
        return l * np.sum(O)


# @partial(jax.jit, static_argnames=[])
def compute_F(R, S, L):
    return R + S + L


# @partial(jax.jit, static_argnames=['q', 'inner_loss_grad_intr', 'outer_loss_grad_intr'])
def compute_U(T, Q, I, O, q, inner_loss_grad_intr, outer_loss_grad_intr=None):
    U = inner_loss_grad_intr(I) * T
    if q != 1:
        U *= Q
    if outer_loss_grad_intr is not None:
        U *= outer_loss_grad_intr(O)
    active_rows = np.any(U, axis=1)
    active_cols = np.any(U, axis=0)
    active_data = np.logical_or(active_rows, active_cols)
    none_active = not active_data.any()
    return U, active_rows, active_cols, active_data, none_active


# @partial(jax.jit, static_argnames=['r', 'x_mode', 'a_mode'])
def compute_dRdA(Z, J, K, r, x_mode, a_mode):
    if a_mode == 'full':
        if x_mode == 'raw':
            return r * K
        elif x_mode == 'kernel':
            return r * Z @ K
    elif a_mode == 'diagonal':
        if x_mode == 'raw':
            return r * K
        elif x_mode == 'kernel':
            return r * np.sum(Z * K, axis=1, keepdims=True)
    elif a_mode == 'decomposed':
        return r * 2 * K @ J


# @partial(jax.jit, static_argnames=['l', 'a_mode', 'reduce_derivative_matrix'])
def compute_dLdA(X, A, U, active_data, l, a_mode, reduce_derivative_matrix):
    if reduce_derivative_matrix:
        U = U[np.ix_(active_data, active_data)]
        X = X[active_data]
    W = np.negative(U + U.T)
    np.fill_diagonal(W, np.diagonal(W) - np.sum(W, axis=0))
    if a_mode == 'full':
        return l * (X.T @ W @ X)
    elif a_mode == 'diagonal':
        return l * np.sum(X.T * (W @ X).T, axis=1, keepdims=True)
    elif a_mode == 'decomposed':
        return l * 2 * ((A @ X.T) @ W @ X)


# @partial(jax.jit, static_argnames=[])
def compute_dFdA(dRdA, dLdA):
    return dRdA + dLdA


# @partial(jax.jit, static_argnames=[])
def compute_phiA(dFdA):
    return -np.vdot(dFdA, dFdA)


# @partial(jax.jit, static_argnames=['s'])
def compute_dSdE(E, s):
    return s * (E - 1)


# @partial(jax.jit, static_argnames=[l, e_mode])
def compute_dLdE(U, l, e_mode):
    if e_mode == 'single':
        return l * -np.sum(U, keepdims=True)
    elif e_mode == 'multiple':
        return l * -np.sum(U, axis=1, keepdims=True)


# @partial(jax.jit, static_argnames=[])
def compute_dFdE(dSdE, dLdE):
    return dSdE + dLdE


# @partial(jax.jit, static_argnames=[])
def compute_phiE(dFdE):
    return -np.vdot(dFdE, dFdE)


# @partial(jax.jit, static_argnames=['a_mode'])
def compute_A_is_psd(A, a_mode):
    if a_mode == 'full':
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    elif a_mode == 'diagonal':
        if np.all(A >= 0):
            return True
        else:
            return False
    elif a_mode == 'decomposed':
        return True


# @partial(jax.jit, static_argnames=['a_mode', 'm'])
def compute_eigh(A, a_mode, m):
    if a_mode == 'full':
        eigenvals, eigenvecs = np.linalg.eigh(A)
    elif a_mode == 'diagonal':
        G = A.ravel()
        H = np.argsort(G)
        eigenvals = G[H]
        eigenvecs = np.identity(m)[:, H]
    elif a_mode == 'decomposed':
        eigenvals, eigenvecs = np.linalg.eigh(A.T @ A)
    return eigenvals, eigenvecs


class MLNNEngine:
    def __init__(self, X, Y, Z=None, mlnn_params=None):
        self.r = 0.0
        self.s = 0.0
        self.l = 1.0
        self.q = 1.0
        self.inner_loss = None
        self.outer_loss = None
        self.x_mode = None
        self.a_mode = 'full'
        self.e_mode = 'single'
        self.keep_a_psd = True
        self.keep_a_centered = False
        self.keep_e_positive = None
        self.reduce_derivative_matrix = False
        self.check_array_equal = False

        if mlnn_params is None or 'inner_loss' not in mlnn_params:
            self.inner_loss = get_activation('relu', 1.0)
        if mlnn_params is None or 'outer_loss' not in mlnn_params:
            self.outer_loss = self.inner_loss

        if mlnn_params:
            self.apply_params(mlnn_params)

        if self.x_mode is None:
            if Z is None:
                self.x_mode = 'raw'
            else:
                self.x_mode = 'kernel'

        if self.keep_e_positive is None:
            self.keep_e_positive = self.keep_a_psd

        assert self.r >= 0
        assert self.s >= 0
        assert self.l > 0
        assert self.q > 0

        self.X = X
        self.Y = Y

        self.n = self.X.shape[0]
        self.m = self.X.shape[1]

        assert self.n > 1
        assert self.m > 0
        assert self.Y.size == self.n
        assert self.Y.shape[0] == self.n

        self.Z_equals_X = False
        if self.x_mode == 'raw':
            assert Z is None
            self.Z = None
        elif self.x_mode == 'kernel':
            if Z is None:
                self.Z_equals_X = True
                self.Z = self.X
            else:
                self.Z = Z

            assert self.Z.shape[0] == self.m
            assert np.array_equal(self.Z, self.Z.T)

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

    def apply_params(self, params):
        for attr in params:
            if hasattr(self, attr):
                setattr(self, attr, params[attr])

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, X):
        self._X = X
        self.D = None

    @property
    def Y(self):
        return self._Y

    @Y.setter
    def Y(self, Y):
        self._Y = Y
        self.T = None

    @property
    def Z(self):
        return self._Z

    @Z.setter
    def Z(self, Z):
        self._Z = Z
        self.J = None

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, A):
        if self.a_mode == 'full':
            self._A = (A + A.T) / 2
        else:
            self._A = A
        self.J = None
        self.A_is_psd = None
        self.eigenvals = None
        self.eigenvecs = None

        if self.keep_a_psd and not self.A_is_psd:
            self._A = self._A_psd_projection()
            self.A_is_psd = True

        if self.keep_a_centered:
            self._A = self._A_centered_projection()
            self.eigenvals = None
            self.eigenvecs = None

    @property
    def E(self):
        return self._E

    @E.setter
    def E(self, E):
        self._E = E
        if self.s:
            self.S = None
            self.dSdE = None
        self.I = None

        if self.keep_e_positive:
            self._E = self._E_positive_projection()

    @property
    def T(self):
        if self._T is None:
            self._compute_T()
        return self._T

    @T.setter
    def T(self, T):
        self._T = T
        self.N = None
        self.Q = None
        self.I = None

    @property
    def N(self):
        if self._N is None:
            self._compute_N()
        return self._N

    @N.setter
    def N(self, N):
        self._N = N
        self.O = None

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
        self.D = None

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

    @property
    def O(self):
        if self._O is None:
            self._compute_O()
        return self._O

    @O.setter
    def O(self, O):
        self._O = O
        self.L = None
        self.U = None
        self.active_rows = None
        self.active_cols = None
        self.active_data = None
        self.none_active = None

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
    def U(self):
        if self._U is None:
            self._compute_U()
        return self._U

    @U.setter
    def U(self, U):
        self._U = U
        self.dLdA = None
        self.dLdE = None

    @property
    def active_rows(self):
        if self._active_rows is None:
            self._compute_U()
        return self._active_rows

    @active_rows.setter
    def active_rows(self, active_rows):
        self._active_rows = active_rows

    @property
    def active_cols(self):
        if self._active_cols is None:
            self._compute_U()
        return self._active_cols

    @active_cols.setter
    def active_cols(self, active_cols):
        self._active_cols = active_cols

    @property
    def active_data(self):
        if self._active_data is None:
            self._compute_U()
        return self._active_data

    @active_data.setter
    def active_data(self, active_data):
        self._active_data = active_data

    @property
    def none_active(self):
        if self._none_active is None:
            self._compute_U()
        return self._none_active

    @none_active.setter
    def none_active(self, none_active):
        self._none_active = none_active

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
    def eigenvals(self):
        if self._eigenvals is None:
            self._compute_eigh()
        return self._eigenvals

    @eigenvals.setter
    def eigenvals(self, eigenvals):
        self._eigenvals = eigenvals

    @property
    def eigenvecs(self):
        if self._eigenvecs is None:
            self._compute_eigh()
        return self._eigenvecs

    @eigenvecs.setter
    def eigenvecs(self, eigenvecs):
        self._eigenvecs = eigenvecs

    def _compute_T(self):
        self.T = compute_T(self.Y)

    def _compute_N(self):
        self.N = compute_N(self.T, self.q)

    def _compute_Q(self):
        self.Q = compute_Q(self.T, self.q)

    def _compute_J(self):
        self.J = compute_J(self.Z, self.A, self.x_mode, self.a_mode)

    def _compute_K(self):
        self.K = compute_K(self.A, self.J, self.a_mode)

    def _compute_R(self):
        self.R = compute_R(self.K, self.r)

    def _compute_S(self):
        self.S = compute_S(self.E, self.s)

    def _compute_D(self):
        self.D = compute_D(self.X, self.A, self.J, self.a_mode, self.Z_equals_X)

    def _compute_I(self):
        self.I = compute_I(self.E, self.T, self.D, self.inner_loss.intr)

    def _compute_O(self):
        if self.outer_loss is not None:
            self.O = compute_O(self.N, self.Q, self.I, self.q, self.inner_loss.func_intr,
                               self.outer_loss.intr)
        else:
            self.O = compute_O(self.N, self.Q, self.I, self.q, self.inner_loss.func_intr)

    def _compute_L(self):
        if self.outer_loss is not None:
            self.L = compute_L(self.O, self.l,
                               self.outer_loss.func_intr)
        else:
            self.L = compute_L(self.O, self.l)

    def _compute_F(self):
        self.F = compute_F(self.R, self.S, self.L)
        self.F_count += 1

    def _compute_U(self):
        if self.outer_loss is not None:
            self.U, self.active_rows, self.active_cols, self.active_data, self.none_active = compute_U(
                self.T, self.Q, self.I, self.O, self.q, self.inner_loss.grad_intr,
                self.outer_loss.grad_intr)
        else:
            self.U, self.active_rows, self.active_cols, self.active_data, self.none_active = compute_U(
                self.T, self.Q, self.I, self.O, self.q, self.inner_loss.grad_intr)

    def _compute_dRdA(self):
        self.dRdA = compute_dRdA(self.Z, self.J, self.K, self.r, self.x_mode, self.a_mode)

    def _compute_dLdA(self):
        if self.none_active:
            self.dLdA = 0
        else:
            self.dLdA = compute_dLdA(self.X, self.A, self.U, self.active_data, self.l, self.a_mode,
                                     self.reduce_derivative_matrix)

    def _compute_dFdA(self):
        self.dFdA = compute_dFdA(self.dRdA, self.dLdA)
        self.dFdA_count += 1

    def _compute_phiA(self):
        if np.isscalar(self.dFdA):
            self.phiA = 0.0
        else:
            self.phiA = compute_phiA(self.dFdA)

    def _compute_dSdE(self):
        self.dSdE = compute_dSdE(self.E, self.s)

    def _compute_dLdE(self):
        if self.none_active:
            self.dLdE = 0
        else:
            self.dLdE = compute_dLdE(self.U, self.l, self.e_mode)

    def _compute_dFdE(self):
        self.dFdE = compute_dFdE(self.dSdE, self.dLdE)
        self.dFdE_count += 1

    def _compute_phiE(self):
        if np.isscalar(self.dFdE):
            self.phiE = 0.0
        else:
            self.phiE = compute_phiE(self.dFdE)

    def _compute_A_is_psd(self):
        self.A_is_psd = compute_A_is_psd(self.A, self.a_mode)

    def _compute_eigh(self):
        self.eigenvals, self.eigenvecs = compute_eigh(self.A, self.a_mode, self.m)
        if self.a_mode in ('full', 'decomposed'):
            self.eigh_count += 1

    def _A_psd_projection(self, tol=1e-10):
        if self.a_mode == 'full':
            eigenvals = self.eigenvals
            if eigenvals[-1] > tol:
                i = np.argmax(eigenvals > tol)
                eigenvals[:i] = 0
                A = (self.eigenvecs[:, i:] * eigenvals[i:]) @ self.eigenvecs[:, i:].T
                A = (A + A.T) / 2
            else:
                eigenvals = np.zeros(eigenvals.shape)
                A = np.zeros(self.A.shape)

            self.eigenvals = eigenvals
        elif self.a_mode == 'diagonal':
            A = np.maximum(self.A, 0)
        elif self.a_mode == 'decomposed':
            A = self.A

        return A

    def _A_centered_projection(self):
        A = self.A
        if self.a_mode == 'full':
            A -= np.sum(A, axis=0, keepdims=True) / self.m
            A -= np.sum(A, axis=1, keepdims=True) / self.m
            A = (A + A.T) / 2
        elif self.a_mode == 'decomposed':
            A -= np.sum(A, axis=1, keepdims=True) / self.m

        return A

    def _E_positive_projection(self):
        return np.maximum(self.E, 0)

    def update_A(self, A, dA, alpha):
        self.A = A - alpha * dA

    def update_E(self, E, dE, alpha):
        self.E = E - alpha * dE

    def compute_A_0(self, initialization='random', seed=None, d=None):
        if initialization == 'random':
            rng = np.random.Generator(np.random.PCG64(seed))

        if self.a_mode == 'full':
            if initialization == 'zero':
                A = np.zeros((self.m, self.m))
            else:
                if initialization == 'random':
                    A = rng.standard_normal((self.m, self.m)) / np.sqrt(self.m)
                    A = A.T @ A
                elif initialization == 'identity':
                    A = np.diag(np.ones(self.m) / np.sqrt(self.m))
                elif initialization == 'centered':
                    A = self.X.T @ (np.identity(self.n) - 1 / self.n) @ self.X

                if self.keep_a_centered:
                    A -= np.sum(A, axis=0, keepdims=True) / self.m
                    A -= np.sum(A, axis=1, keepdims=True) / self.m

                if self.x_mode == 'raw':
                    K = A
                elif self.x_mode == 'kernel':
                    K = A @ self.Z
                A /= np.sqrt(np.vdot(K, K))

                A = (A + A.T) / 2
        elif self.a_mode == 'diagonal':
            if initialization == 'zero':
                A = np.zeros((self.m, 1))
            else:
                if initialization == 'random':
                    A = np.square(rng.standard_normal((self.m, 1)))
                elif initialization == 'identity':
                    A = np.ones((self.m, 1)) / np.sqrt(self.m)

                if self.x_mode == 'raw':
                    K = A
                elif self.x_mode == 'kernel':
                    K = A * self.Z
                A /= np.sqrt(np.vdot(K, K))
        elif self.a_mode == 'decomposed':
            if d is None:
                d = self.m

            if initialization == 'random':
                A = rng.standard_normal((d, self.m)) / np.sqrt(d)
            elif initialization == 'pca':
                pca = PCA(n_components=d)
                pca.fit(self.X)
                A = pca.components_ / np.sqrt(d)
            elif initialization == 'kpca':
                if self.x_mode == 'raw':
                    Z = self.X
                    assert np.array_equal(Z, Z.T)
                elif self.x_mode == 'kernel':
                    Z = self.Z

                kpca = KernelPCA(n_components=d, kernel='precomputed')
                kpca.fit(Z)
                A = kpca.eigenvecs_.T / np.sqrt(d)

            if self.keep_a_centered:
                A -= np.sum(A, axis=1, keepdims=True) / self.m

            if self.x_mode == 'raw':
                K = A @ A.T
            elif self.x_mode == 'kernel':
                K = A @ self.Z @ A.T
            A /= np.power(np.vdot(K, K), 0.25)

        return A

    def compute_E_0(self, initialization='random', seed=None):
        if initialization == 'random':
            rng = np.random.Generator(np.random.PCG64(seed))

        if self.e_mode == 'single':
            if initialization == 'zero':
                E = 0.0
            elif initialization == 'random':
                E = float(np.square(rng.standard_normal()))
            elif initialization in ('identity', 'centered', 'pca', 'kpca'):
                E = 1.0
        elif self.e_mode == 'multiple':
            if initialization == 'zero':
                E = np.zeros((self.n, 1))
            elif initialization == 'random':
                E = np.square(rng.standard_normal((self.n, 1)))
            elif initialization in ('identity', 'centered', 'pca', 'kpca'):
                E = np.ones((self.n, 1))

        return np.atleast_2d(E)

    def fun(self, x, arguments='AE'):
        i = 0
        if 'A' in arguments:
            A = x[0:self.A.size].reshape(self.A.shape)
            i = self.A.size
            if not self.check_array_equal or not np.array_equal(A, self.A):
                self.A = A
        if 'E' in arguments:
            E = x[i:].reshape(self.E.shape)
            if not self.check_array_equal or not np.array_equal(E, self.E):
                self.E = E

        return self.F

    def jac(self, x, arguments='AE'):
        i = 0
        size = 0
        if 'A' in arguments:
            i = self.A.size
            size += self.A.size
            A = x[0:i].reshape(self.A.shape)
            if not self.check_array_equal or not np.array_equal(A, self.A):
                self.A = A
        if 'E' in arguments:
            size += self.E.size
            E = x[i:].reshape(self.E.shape)
            if not self.check_array_equal or not np.array_equal(E, self.E):
                self.E = E

        jac = np.empty(size)
        if 'A' in arguments:
            if np.isscalar(self.dFdA):
                jac[0:i] = np.zeros(self.A.size)
            else:
                jac[0:i] = self.dFdA.ravel()
        if 'E' in arguments:
            if np.isscalar(self.dFdE):
                jac[i:] = np.zeros(self.E.size)
            else:
                jac[i:] = self.dFdE.ravel()

        return jac

    def get_transformation_matrix(self, n_components=None):
        if self.a_mode == 'decomposed':
            if n_components is None:
                n_components = self.A.shape[0]

            if n_components == self.A.shape[0]:
                return self.A.copy()

        if n_components is None:
            n_components = self.m

        d = np.minimum(n_components, self.m)
        M = (np.sqrt(np.clip(self.eigenvals[self.m - d:], 0).reshape(d, 1)) *
             self.eigenvecs[:, self.m - d:].T)

        return np.vstack((M, np.zeros((n_components - d, self.m))))

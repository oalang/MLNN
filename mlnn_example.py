import numpy as np
import matplotlib.pyplot as plt
import loss
from mlnn import MLNN

from sklearn.datasets import load_wine
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def main():
    rng = np.random.Generator(np.random.PCG64(12345))
    data = load_wine()

    X_original = np.array(data['data'])
    Y_original = np.array(data['target'], dtype=int)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    for train_index, test_index in split.split(X_original, Y_original):
        X_train, Y_train = X_original[train_index, :], Y_original[train_index]
        X_test, Y_test = X_original[test_index, :], Y_original[test_index]

    X_train, Y_train = X_original, Y_original
    X_test, Y_test = X_original, Y_original

    pipeline = Pipeline([('std_scaler', StandardScaler())])
    X_train_scaled = pipeline.fit_transform(X_train)
    X_test_scaled = pipeline.transform(X_test)

    pca = PCA(n_components=2)
    X_train_2D = pca.fit_transform(X_train_scaled)

    plt.figure(figsize=(4, 4))
    plt.scatter(X_train_2D[:, 0], X_train_2D[:, 1], c=Y_train)

    knn = KNeighborsClassifier(3)
    knn.fit(X_train_scaled, Y_train)
    Y_test_pred = knn.predict(X_test_scaled)
    accuracy = accuracy_score(Y_test, Y_test_pred)
    print(f"accuracy = {accuracy: .3f}")

    config = 1
    e_mode = 'single'    # single/multiple
    m_mode = 'full'      # full/decomposed/diagonal
    i_mode = 'zero'      # zero/random/centered/identity/pca
    m = 2

    r = 1
    s = 0
    l = 1
    inner = loss.SmoothReLU(.5)
    outer = loss.SmoothReLU(.5)

    alpha_0 = 1e-3
    armijo = 1e-6
    max_backtracks = 50

    min_delta_F = 1e-6
    max_steps = 100
    max_time = 1

    X = X_train_scaled
    Y = Y_train
    T = np.where(np.equal(Y.reshape(-1, 1), Y.reshape(1, -1)), 1, -1)
    N = np.sum(T == 1, axis=1) - 1

    sigma2 = 10 ** 1.12
    P = X @ X.T
    D = P.diagonal().reshape(-1, 1) + P.diagonal().reshape(1, -1) - 2 * P
    G = np.exp(D / (-2 * sigma2))

    n, d = X.shape

    if config == 1:
        a_mode = 'WX'
        B = X
    elif config == 2:
        a_mode = 'MX'
        B = X
    elif config == 3:
        a_mode = 'MXX'
        B = P
    elif config == 4:
        a_mode = 'MG'
        B = P
    elif config == 5:
        a_mode = 'MXX'
        B = G
    elif config == 6:
        a_mode = 'MG'
        B = G

    if m_mode == 'decomposed':
        if i_mode == 'random':
            if a_mode == 'WX':
                A = rng.standard_normal(m * d).reshape(m, d) / m ** .5
                C = A.T @ A
            elif a_mode == 'MX':
                A = rng.standard_normal(m * n).reshape(m, n) / m ** .5
                C = B.T @ A.T @ A @ B
            elif a_mode == 'MXX' or a_mode == 'MG':
                A = rng.standard_normal(m * n).reshape(m, n) / m ** .5
                C = A.T @ A @ B

            A /= np.dot(C.T.ravel(), C.ravel()) ** .25
        elif i_mode == 'pca':
            if a_mode == 'WX':
                pca = PCA(n_components=m)
                pca.fit(B)
                A = pca.components_[0:m] / m ** .5
                C = A.T @ A
            elif a_mode == 'MX':
                kpca = KernelPCA(n_components=m, kernel='precomputed')
                kpca.fit(B @ B.T)
                A = kpca.eigenvectors_.T[0:m] / m ** .5
                C = B.T @ A.T @ A @ B
            elif a_mode == 'MXX' or a_mode == 'MG':
                kpca = KernelPCA(n_components=m, kernel='precomputed')
                kpca.fit(B)
                A = kpca.eigenvectors_.T[0:m] / m ** .5
                C = A.T @ A @ B

            A /= np.dot(C.T.ravel(), C.ravel()) ** .25
    elif m_mode == 'full':
        if i_mode == 'zero':
            if a_mode == 'WX':
                A = np.zeros((d, d))
            elif a_mode == 'MX' or a_mode == 'MXX' or a_mode == 'MG':
                A = np.zeros((n, n))
        elif i_mode == 'random':
            if a_mode == 'WX':
                A = rng.standard_normal(d * d).reshape(d, d) / d ** .5
                A = A.T @ A
                C = A
            elif a_mode == 'MX':
                A = rng.standard_normal(n * n).reshape(n, n) / n ** .5
                A = A.T @ A
                C = B.T @ A @ B
            elif a_mode == 'MXX' or a_mode == 'MG':
                A = rng.standard_normal(n * n).reshape(n, n) / n ** .5
                A = A.T @ A
                C = A @ B

            A /= np.dot(C.T.ravel(), C.ravel()) ** .5
        elif i_mode == 'identity':
            if a_mode == 'WX':
                A = np.diag(np.ones(d) / d ** .5)
            elif a_mode == 'MX':
                A = np.diag(np.ones(n) / np.dot((B.T @ B).T.ravel(), (B.T @ B).ravel()) ** .5)
            elif a_mode == 'MXX' or a_mode == 'MG':
                A = np.diag(np.ones(n) / np.dot(B.T.ravel(), B.ravel()) ** .5)
        elif i_mode == 'centered':
            U = np.identity(n) - 1 / n

            if a_mode == 'WX':
                A = B.T @ U @ B
                C = A
            elif a_mode == 'MX':
                A = U
                C = B.T @ A @ B
            elif a_mode == 'MXX' or a_mode == 'MG':
                A = U
                C = A @ B

            A /= np.dot(C.T.ravel(), C.ravel()) ** .5
    elif m_mode == 'diagonal':
        if i_mode == 'zero':
            if a_mode == 'WX':
                A = np.zeros(d)
            elif a_mode == 'MX' or a_mode == 'MXX' or a_mode == 'MG':
                A = np.zeros(n)
        elif i_mode == 'random':
            if a_mode == 'WX':
                A = rng.standard_normal(d) ** 2
                C = A
            elif a_mode == 'MX':
                A = rng.standard_normal(n) ** 2
                C = B.T @ (A.reshape(-1, 1) * B)
            elif a_mode == 'MXX' or a_mode == 'MG':
                A = rng.standard_normal(n) ** 2
                C = A.reshape(-1, 1) * B

            A /= np.dot(C.T.ravel(), C.ravel()) ** .5
        elif i_mode == 'identity':
            if a_mode == 'WX':
                A = np.ones(d) / d ** .5
            elif a_mode == 'MX':
                A = np.ones(n) / np.dot((B.T @ B).T.ravel(), (B.T @ B).ravel()) ** .5
            elif a_mode == 'MXX' or a_mode == 'MG':
                A = np.ones(n) / np.dot(B.T.ravel(), B.ravel()) ** .5

    if e_mode == 'single':
        if i_mode == 'zero':
            E = np.zeros(1)
        elif i_mode == 'random':
            E = rng.standard_normal(1) ** 2
        elif i_mode == 'centered' or i_mode == 'identity' or i_mode == 'pca':
            E = np.ones(1)
    elif e_mode == 'multiple':
        if i_mode == 'zero':
            E = np.zeros(n)
        elif i_mode == 'random':
            E = rng.standard_normal(n) ** 2
        elif i_mode == 'centered' or i_mode == 'identity' or i_mode == 'pca':
            E = np.ones(n)

    mlnn_params = {
        'r': r,
        's': s,
        'l': l,
        'inner': inner,
        'outer': outer,
        'a_mode': a_mode,
        'e_mode': e_mode,
        'm_mode': m_mode,
    }

    line_search_params = {
        'alpha_0': alpha_0,
        'armijo': armijo,
        'max_backtracks': max_backtracks,
    }

    optimize_params = {
        'min_delta_F': min_delta_F,
        'max_steps': max_steps,
        'max_time': max_time,
    }

    mlnn = MLNN(B, T, N, A, E, mlnn_params, line_search_params, optimize_params)
    mlnn.optimize(verbose=True)
    mlnn.minimize(verbose=True)

    #plt.show()


if __name__ == '__main__':
    main()

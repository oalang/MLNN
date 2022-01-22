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
    
#    kernel = 'linear'
    kernel = 'rbf'

#    k_mode = 'linear'
    k_mode = 'nonlinear'

    a_mode = 'full'
#    a_mode = 'diagonal'
#    a_mode = 'decomposed'

    e_mode = 'single'
#    e_mode = 'multiple'

    i_mode = 'zero'
#    i_mode = 'random'
#    i_mode = 'centered'
#    i_mode = 'identity'
#    i_mode = 'pca'

    d = 2

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
    N = np.sum(T == 1, axis=1, keepdims=True) - 1

    sigma2 = 10 ** 1.12
    P = X @ X.T
    D = P.diagonal().reshape(-1, 1) + P.diagonal().reshape(1, -1) - 2 * P
    G = np.exp(D / (-2 * sigma2))

    if k_mode == 'linear':
        B = X
    elif k_mode == 'nonlinear':
        if kernel == 'linear':
            B = P
        elif kernel == 'rbf':
            B = G

    if k_mode == 'linear':
        C = None
    elif k_mode == 'nonlinear':
        # C = B
        C = None

    n, m = B.shape

    if a_mode == 'full':
        if i_mode == 'zero':
            A = np.zeros((m, m))
        else:
            if i_mode == 'random':
                A = rng.standard_normal(m * m).reshape(m, m) / m ** .5
                A = A.T @ A
            elif i_mode == 'identity':
                A = np.diag(np.ones(m) / m ** .5)
            elif i_mode == 'centered':
                U = np.identity(n) - 1 / n
                A = B.T @ U @ B

            if k_mode == 'linear':
                K = A
            elif k_mode == 'nonlinear':
                K = A @ C
            A /= np.dot(K.T.ravel(), K.ravel()) ** .5
    elif a_mode == 'diagonal':
        if i_mode == 'zero':
            A = np.zeros(m).reshape(m, 1)
        else:
            if i_mode == 'random':
                A = rng.standard_normal(m).reshape(m, 1) ** 2
            elif i_mode == 'identity':
                A = np.ones(m).reshape(m, 1) / m ** .5

            if k_mode == 'linear':
                K = A
            elif k_mode == 'nonlinear':
                K = A * C
            A /= np.dot(K.T.ravel(), K.ravel()) ** .5
    elif a_mode == 'decomposed':
        if i_mode == 'random':
            A = rng.standard_normal(d * m).reshape(d, m) / d ** .5
        elif i_mode == 'pca':
            if k_mode == 'linear':
                pca = PCA(n_components=d)
                pca.fit(B)
                A = pca.components_ / d ** .5
            elif k_mode == 'nonlinear':
                kpca = KernelPCA(n_components=d, kernel='precomputed')
                kpca.fit(C)
                A = kpca.eigenvectors_.T / d ** .5

        if k_mode == 'linear':
            K = A.T @ A
        elif k_mode == 'nonlinear':
            K = A @ C @ A.T
        A /= np.dot(K.T.ravel(), K.ravel()) ** .25

    if e_mode == 'single':
        if i_mode == 'zero':
            E = np.zeros(1).item()
        elif i_mode == 'random':
            E = rng.standard_normal(1).item() ** 2
        elif i_mode == 'centered' or i_mode == 'identity' or i_mode == 'pca':
            E = np.ones(1).item()
    elif e_mode == 'multiple':
        if i_mode == 'zero':
            E = np.zeros(n).reshape(n, 1)
        elif i_mode == 'random':
            E = rng.standard_normal(n).reshape(n, 1) ** 2
        elif i_mode == 'centered' or i_mode == 'identity' or i_mode == 'pca':
            E = np.ones(n).reshape(n, 1)

    mlnn_params = {
        'r': r,
        's': s,
        'l': l,
        'inner': inner,
        'outer': outer,
        'k_mode': k_mode,
        'a_mode': a_mode,
        'e_mode': e_mode,
        'i_mode': i_mode,
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

    mlnn = MLNN(B, T, N, C, A, E, mlnn_params, line_search_params, optimize_params)
    mlnn.optimize(verbose=True)
    mlnn.minimize(verbose=True)

    #plt.show()


if __name__ == '__main__':
    main()

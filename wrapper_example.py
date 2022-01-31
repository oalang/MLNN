import numpy as np
import matplotlib.pyplot as plt

import loss
from mlnn import MLNN

from sklearn.datasets import load_wine
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def main():
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

    k_mode = 'linear'
    # k_mode = 'nonlinear'

    a_mode = 'full'
    # a_mode = 'diagonal'
    # a_mode = 'decomposed'

    e_mode = 'single'
    # e_mode = 'multiple'

    i_mode = 'zero'
    # i_mode = 'random'
    # i_mode = 'centered'
    # i_mode = 'identity'
    # i_mode = 'pca'

    keep_a_psd = False
    keep_a_centered = False
    keep_e_positive = False

    d = 2

    r = 1
    s = 0
    l = 1
    q = 1
    inner_loss = loss.SmoothReLU(.5)
    outer_loss = loss.SmoothReLU(.5)
    # outer_loss = None

    alpha_0 = 1e-3
    armijo = 1e-6
    max_backtracks = 50

    min_delta_F = 1e-6
    max_steps = 100
    max_time = 1

    mlnn_params = {
        'r': r,
        's': s,
        'l': l,
        'q': q,
        'inner_loss': inner_loss,
        'outer_loss': outer_loss,
        'k_mode': k_mode,
        'a_mode': a_mode,
        'e_mode': e_mode,
        'i_mode': i_mode,
        'keep_a_psd': keep_a_psd,
        'keep_a_centered': keep_a_centered,
        'keep_e_positive': keep_e_positive,
    }

    optimize_params = {
        'min_delta_F': min_delta_F,
        'max_steps': max_steps,
        'max_time': max_time,
    }

    line_search_params = {
        'alpha_0': alpha_0,
        'armijo': armijo,
        'max_backtracks': max_backtracks,
    }

    X = X_train_scaled
    Y = Y_train

    mlnn = MLNN(d, mlnn_params, optimize_params, line_search_params)
    mlnn.fit(X, Y)

    # plt.show()


if __name__ == '__main__':
    main()
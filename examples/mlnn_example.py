import numpy as np
import matplotlib.pyplot as plt
import loss
from mlnn.engine import MLNNEngine
from mlnn.callback import MLNNCallback
from mlnn.optimize import MLNNSteepestDescent, MLNNBFGS

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
    
    rbf = False

    kernel = 'linear'
#    kernel = 'rbf'

    a_mode = 'full'
#    a_mode = 'diagonal'
#    a_mode = 'decomposed'

    e_mode = 'single'
    e_mode = 'multiple'

    i_mode = 'zero'
#    i_mode = 'random'
#    i_mode = 'centered'
#    i_mode = 'identity'
#    i_mode = 'pca'

    keep_a_psd = False
    keep_a_centered = False
    keep_e_positive = False

    n_components = 2

    r = 1
    s = 0
    l = 1
    q = 1
    inner_loss = loss.SmoothReLU2(.5)
    outer_loss = loss.SmoothReLU2(.5)
#    outer_loss = None

    optimize_method = 'fixed'
    initialization = i_mode
    min_delta_F = 1e-6
    max_steps = 100
    max_time = 1000
    fixed_arguments = 'AE'
    max_arg_steps = 5

    line_search_method = 'strong_wolfe'
    use_prev_f = False
    alpha_0 = 1e-3
    armijo = 1e-6
    wolfe = .9
    max_ls_iterations = 50

    X = X_train_scaled
    Y = Y_train
    T = np.where(np.equal(Y.reshape(-1, 1), Y.reshape(1, -1)), 1, -1)
    N = np.sum(T == 1, axis=1, keepdims=True) - 1

    sigma2 = 10 ** 1.12
    P = X @ X.T
    D = P.diagonal().reshape(-1, 1) + P.diagonal().reshape(1, -1) - 2 * P
    G = np.exp(D / (-2 * sigma2))

    if kernel == 'linear':
        B = X
    elif kernel == 'nonlinear':
        if rbf:
            B = G
        else:
            B = G

    if kernel == 'linear':
        C = None
    elif kernel == 'nonlinear':
        #C = B
        C = None

    mlnn_params = {
        'r': r,
        's': s,
        'l': l,
        'q': q,
        'inner_loss': inner_loss,
        'outer_loss': outer_loss,
        'kernel': kernel,
        'a_mode': a_mode,
        'e_mode': e_mode,
        'keep_a_psd': keep_a_psd,
        'keep_a_centered': keep_a_centered,
        'keep_e_positive': keep_e_positive,
    }

    optimize_params = {
        'optimize_method': optimize_method,
        'initialization': initialization,
        'min_delta_F': min_delta_F,
        'max_steps': max_steps,
        'max_time': max_time,
        'fixed_arguments': fixed_arguments,
        'max_arg_steps': max_arg_steps,
    }

    line_search_params = {
        'line_search_method': line_search_method,
        'use_prev_f': use_prev_f,
        'alpha_0': alpha_0,
        'armijo': armijo,
        'wolfe': wolfe,
        'max_ls_iterations': max_ls_iterations,
    }

    mlnn = MLNNEngine(B, T, N, C, mlnn_params)
    callback = MLNNCallback(print_stats=True)
    optimizer = MLNNSteepestDescent(mlnn, callback=callback, n_components=n_components, optimize_params=optimize_params, line_search_params=line_search_params)
    optimizer.minimize(verbose=False)
    optimizer.report()

    line_search_params['line_search_method'] = 'strong_wolfe'
    mlnn = MLNNEngine(B, T, N, C, mlnn_params)
    callback = MLNNCallback(print_stats=True)
    optimizer = MLNNSteepestDescent(mlnn, callback=callback, n_components=n_components, optimize_params=optimize_params, line_search_params=line_search_params)
    optimizer.minimize(verbose=False)
    optimizer.report()

    mlnn = MLNNEngine(B, T, N, C, mlnn_params)
    callback = MLNNCallback(print_stats=True)
    optimizer = MLNNBFGS(mlnn, callback=callback, n_components=n_components, optimize_params=optimize_params, line_search_params=line_search_params)
    optimizer.minimize(verbose=False)
    optimizer.report()

    #plt.show()


if __name__ == '__main__':
    main()

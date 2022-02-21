from time import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NeighborhoodComponentsAnalysis, KNeighborsClassifier
from sklearn.metrics import accuracy_score

from mlnn import MLNN


def main():
    data = load_digits()

    X_original = np.array(data['data'])
    Y_original = np.array(data['target'], dtype=int)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=12345)
    for train_index, test_index in split.split(X_original, Y_original):
        X_train, Y_train = X_original[train_index, :], Y_original[train_index]
        X_test, Y_test = X_original[test_index, :], Y_original[test_index]

    pipeline = Pipeline([('std_scaler', StandardScaler())])
    X_train_scaled = pipeline.fit_transform(X_train)
    X_test_scaled = pipeline.transform(X_test)

    pca = PCA(n_components=2)
    start_time = time()
    X_train_pca = pca.fit_transform(X_train_scaled)
    pca_time = time() - start_time
    X_test_pca = pca.transform(X_test_scaled)
    
    lda = LinearDiscriminantAnalysis(n_components=2)
    start_time = time()
    X_train_lda = lda.fit_transform(X_train_scaled, Y_train)
    lda_time = time() - start_time
    X_test_lda = lda.transform(X_test_scaled)

    nca = NeighborhoodComponentsAnalysis(n_components=2)
    start_time = time()
    X_train_nca = nca.fit_transform(X_train_scaled, Y_train)
    nca_time = time() - start_time
    X_test_nca = nca.transform(X_test_scaled)

    mlnn = MLNN(n_components=2)
    start_time = time()
    X_train_mlnn = mlnn.fit_transform(X_train_scaled, Y_train)
    mlnn_time = time() - start_time
    X_test_mlnn = mlnn.transform(X_test_scaled)

    pca_knn = KNeighborsClassifier(3)
    pca_knn.fit(X_train_pca, Y_train)
    Y_test_pca = pca_knn.predict(X_test_pca)
    pca_accuracy = accuracy_score(Y_test, Y_test_pca)

    lda_knn = KNeighborsClassifier(3)
    lda_knn.fit(X_train_lda, Y_train)
    Y_test_lda = lda_knn.predict(X_test_lda)
    lda_accuracy = accuracy_score(Y_test, Y_test_lda)

    nca_knn = KNeighborsClassifier(3)
    nca_knn.fit(X_train_nca, Y_train)
    Y_test_nca = nca_knn.predict(X_test_nca)
    nca_accuracy = accuracy_score(Y_test, Y_test_nca)

    mlnn_knn = KNeighborsClassifier(3)
    mlnn_knn.fit(X_train_mlnn, Y_train)
    Y_test_mlnn = mlnn_knn.predict(X_test_mlnn)
    mlnn_accuracy = accuracy_score(Y_test, Y_test_mlnn)

    plt.figure(figsize=(10, 10))

    plt.subplot(221)
    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=Y_train)
    plt.title(f"PCA\ntime = {pca_time: .3f}s, accuracy = {pca_accuracy: .3f}")

    plt.subplot(222)
    plt.scatter(X_train_lda[:, 0], X_train_lda[:, 1], c=Y_train)
    plt.title(f"LDA\ntime = {lda_time: .3f}s, accuracy = {lda_accuracy: .3f}")

    plt.subplot(223)
    plt.scatter(X_train_nca[:, 0], X_train_nca[:, 1], c=Y_train)
    plt.title(f"NCA\ntime = {nca_time: .3f}s, accuracy = {nca_accuracy: .3f}")

    plt.subplot(224)
    plt.scatter(X_train_mlnn[:, 0], X_train_mlnn[:, 1], c=Y_train)
    plt.title(f"MLNN\ntime = {mlnn_time: .3f}s, accuracy = {mlnn_accuracy: .3f}")

    plt.show()


if __name__ == '__main__':
    main()

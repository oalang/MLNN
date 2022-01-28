import numpy as np
import matplotlib.pyplot as plt
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

    X = X_train_scaled
    Y = Y_train

    mlnn = MLNN()
    mlnn.fit(X, Y)

    # plt.show()


if __name__ == '__main__':
    main()

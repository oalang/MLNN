# METRIC LEARNING FOR NEAREST NEIGHBOR CLASSIFICATION (MLNN)

This is still very much a work in progress.

MLNN is a novel metric learning technique which minimizes an upper bound on the leave-one-out error (LOOE) of a
radius-based neighbors classifier. Like a support vector machine (SVM), MLNN aims to separate data points of different
classes by the largest possible margin. Its loss function is minimized by learning an optimal transformation of the
training data, as well as an optimal radius. Over-training is avoided by ignoring data points whose local neighborhood
is densely packed with points of the same class as itself. MLNN can learn both linear and nonlinear embeddings.

The code in this repository includes a class called MLNNEngine which computes the value of the MLNN loss function and
its gradient. I have implemented a steepest descent optimizer, MLNNSteepestDescent, which performs a line search using
backoff to find a step size which satisfies Armijo's condition for sufficient decrease. The steepest descent optimizer
can also be run using SciPy's line_search function, which finds a step size satisfying the strong Wolfe conditions. In
addition, I have implemented an optimizer, MLNNBFGS, which uses SciPy's L-BFGS-B minimization method.

The MLNN class is an API in the style of Scikit-learn's fit/transform paradigm.

    from mlnn import MLNN
    mlnn = MLNN()
    X_train_mlnn = mlnn.fit_transform(X_train, Y_train)
    X_test_mlnn = mlnn.transform(X_test)

To see how MLNN can be used for dimensionality reduction, run ./examples/digits_example.py. Here we compare MLNN with
Scikit-learn's PCA, LinearDiscriminantAnalysis, and NeighborhoodComponentsAnalysis.

MLNN is an extension of the work described in:

[Sriperumbudur, B K and Lang, O A and Lankriet, G R G. 2008. Metric Embedding for Kernel Classification Rules.
25th International Conference on Machine Learning.
](http://icml2008.cs.helsinki.fi/papers/582.pdf)

"""Utility module."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def fit_plot(X, Y, fit_classifier=None):
    """Fit and plot."""
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'inline')
    h = .02  # mesh step size
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    if fit_classifier:
        # we create an instance of Neighbe the number of bootstrap resamples
        # (n_boot) or set ci to None.ours Classifier and fit the data.
        fit_classifier.fit(X, Y)
        Z = fit_classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(1, figsize=(8, 6))
        plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y[:, 1], cmap=cm_bright)
    plt.xlabel('$x_0$')
    plt.ylabel('$x_1$')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    # plt.show()
    return plt


def get_PU_labels(Y, pct_missings=None, random_seed=None):
    """Get PU labels."""
    # To PU
    if pct_missings is None:
        pct_missings = np.arange(0., 1 + 1e-8, 0.1)
    Y_pu = {}
    np.random.seed(random_seed)
    n_samples = len(Y)
    for pct in pct_missings:
        y = np.argmax(Y, 1)
        flip = np.random.rand(n_samples)
        y[(y != 0) & (flip < pct)] = 0
        Y_pu[pct] = np.eye(Y.shape[1])[y]
        if random_seed is not None:
            print('Positive (pct_missing={}):'.format(pct),
                  np.sum(np.argmax(Y, 1)), ' vs.', np.sum(y))
    return Y_pu

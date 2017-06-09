"""Utility module."""
import numpy as np
import matplotlib.pyplot as plt

def fit_plot(X, Y, fit_classifier=None):
    """Fit and plot."""
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'inline')
    h = .02  # mesh step size
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    n_classes = Y.shape[1]
    f, ax = plt.subplots(n_classes-1, figsize=(10, 8*(n_classes-1)))
    if n_classes == 2:
        axs = [ax]

    if fit_classifier:
        # we create an instance of Neighbe the number of bootstrap resamples
        # (n_boot) or set ci to None.ours Classifier and fit the data.
        fit_classifier.fit(X, Y)
        Z = fit_classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])

        cmaps = [plt.cm.RdBu, plt.cm.RdYlGn]
        for k in range(1, n_classes):
            z = Z[:, k]
            # Put the result into a color plot
            zz = z.reshape(xx.shape)
            axs[k-1].contourf(xx, yy, zz, cmap=cmaps[k-1], alpha=.6)
            axs[k-1].set_title('Class {}'.format(k))

    # Plot also the training points
    ms = ['o', 's', '^']
    cs = ['r', 'b', 'g']
    n_samples = Y.shape[0]
    for k in range(1, n_classes):
        y = np.argmax(Y, axis=1)
        for idx in range(n_samples):
            axs[k-1].scatter(X[idx, 0], X[idx, 1], c=cs[y[idx]], marker=ms[y[idx]],
                    s=50)
        axs[k-1].set_xlabel('$x_0$')
        axs[k-1].set_ylabel('$x_1$')

        axs[k-1].set_xlim(xx.min(), xx.max())
        axs[k-1].set_ylim(yy.min(), yy.max())
    # plt.show()
    return plt


def get_PU_labels(Y, pct_missings=None, random_seed=None, verbose=False):
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
        if verbose is True:
            print('Positive (pct_missing={}):'.format(pct),
                  np.sum(np.argmax(Y, 1)), ' vs.', np.sum(y))
    return Y_pu

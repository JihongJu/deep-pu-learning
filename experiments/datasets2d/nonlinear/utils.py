"""Utility module."""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def fit_and_plot(X, Y, classifier=None, marker_size=None):
    """Fit and plot."""
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'inline')

    h = .02  # mesh step size
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    min_size, max_size = 5, 30  # min and max mark size
    default_size = 10
    S = np.ones(X.shape[0]) * default_size  # default maker size
    if marker_size is None:
        marker_size = ["default"]
    if not isinstance(marker_size, list):
        marker_size = [marker_size]

    n_classes = Y.shape[1]
    n_rows = n_classes - 1
    n_cols = len(marker_size)

    f, axs = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 8 * n_rows))
    if isinstance(axs, matplotlib.axes.SubplotBase):
        axs = np.array([axs])
    if n_rows == 1:
        axs = np.expand_dims(axs, 0)
    elif n_cols == 1:
        axs = np.expand_dims(axs, 1)

    if classifier:
        # we create an instance of Neighbe the number of bootstrap resamples
        # (n_boot) or set ci to None.ours Classifier and fit the data.
        classifier.fit(X, Y)
        Z = classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])

        def normalize_size(S, Y, min_size=5, max_size=30):
            """Normalize S to (min_size, max_size)."""
            N, K = Y.shape
            for k in range(K):
                is_k = Y[:, k] > 0.
                S_k = S[is_k]
                S_k = (S_k - np.min(S_k)) / (np.max(S_k) - np.min(S_k))
                S[is_k] = S_k * (max_size - min_size) + min_size
            return S

        for j in range(n_cols):
            if marker_size[j] == "gradient":
                G = classifier.calc_gradients(X, Y)[0]
                G = np.abs(G)
                S = np.sum(G, axis=1)
                S = normalize_size(S, Y, min_size, max_size)
            elif marker_size[j] == "loss":
                L = classifier.calc_loss(X, Y)
                S = np.sum(L, axis=1)
                S = normalize_size(S, Y, min_size, max_size)

            cmaps = [plt.cm.RdBu, plt.cm.RdYlGn]
            for k in range(1, n_classes):
                z = Z[:, k]
                # Put the result into a color plot
                zz = z.reshape(xx.shape)
                axs[k - 1, j].contourf(xx, yy, zz, cmap=cmaps[k - 1], alpha=.6)
                axs[k - 1, j].set_title(
                    'Class {} {}'.format(k, marker_size[j]))

    # Plot also the training points
    ms = ['o', 's', '^']
    cs = ['r', 'b', 'g']
    n_samples = Y.shape[0]
    y = np.argmax(Y, axis=1)
    for k in range(1, n_classes):
        for j in range(n_cols):
            for idx in range(n_samples):
                axs[k - 1, j].scatter(X[idx, 0], X[idx, 1], c=cs[y[idx]],
                                      marker=ms[y[idx]], s=10 * S[idx])
            axs[k - 1, j].set_xlabel('$x_0$')
            axs[k - 1, j].set_ylabel('$x_1$')

            axs[k - 1, j].set_xlim(xx.min(), xx.max())
            axs[k - 1, j].set_ylim(yy.min(), yy.max())
            axs[k - 1, j].set_xticks(())
            axs[k - 1, j].set_yticks(())
    # plt.show()
    return plt


def get_PU_labels(Y, pct_missings=None, random_state=None, verbose=False):
    """Get PU labels."""
    # To PU
    if pct_missings is None:
        pct_missings = np.arange(0., 1 + 1e-8, 0.1)
    Y_pu = {}
    np.random.seed(random_state)
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

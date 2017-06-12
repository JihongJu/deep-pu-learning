"""Util methods."""
import numpy as np
import matplotlib.pyplot as plt


def show_label_partition(Y):
    """Show partition."""
    print("There are totally {} samples.".format(len(Y)))
    print("+:", np.sum(Y == 1))
    print("-:", np.sum(Y == 0))


def fit_plot(X, Y, classifier=None):
    """Fit and plot."""
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'inline')
    h = .02  # mesh step size
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    if fit_classifier:
        # we create an instance of Neighbe the number of bootstrap resamples
        # (n_boot) or set ci to None.ours Classifier and fit the data.
        fit_classifier.fit(X, Y)
        Z = fit_classifier.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(1, figsize=(4, 3))
        plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Pastel2)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Set1)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.show()
    return plt


def get_gaussians(n_samples, pct_positive=0.2, distance=5, random_seed=None):
    """Get Gaussians."""
    np.random.seed(random_seed)
    n_positives = int(n_samples * pct_positive)
    n_negatives = n_samples - n_positives
    C = np.array([[.5, 0.], [0., .5]])
    X = np.r_[np.random.randn(n_negatives, 2) +
              np.array([-distance, distance]),
              np.dot(np.random.randn(n_positives, 2), C)]
    Y = np.r_[np.zeros(n_negatives), np.ones(n_positives)].astype('int')
    return X, Y


def get_3gaussians(n_samples, pct_positive=0.2, distance=5, random_seed=None):
    """Get three Gaussians."""
    np.random.seed(random_seed)
    n_positives = int(n_samples * pct_positive)
    n_positives_1 = int(n_positives / 2)
    n_positives_2 = n_positives - n_positives_1
    n_negatives = n_samples - n_positives
    C = np.array([[.5, 0.], [0., .5]])
    X = np.r_[np.random.randn(n_negatives, 2) + np.array([distance, distance]),
              np.dot(np.random.randn(n_positives_1, 2), C),
              np.dot(np.random.randn(n_positives_1, 2), C)
              + np.array([+distance, -distance])]
    Y = np.r_[np.zeros(n_negatives), np.ones(n_positives_1),
              2 * np.ones(n_positives_2)].astype('int')
    return X, Y


def get_PU_labels(Y, betas, random_seed=None):
    """Get PU labels."""
    # To PU
    Y_pus = []
    np.random.seed(random_seed)
    n_samples = len(Y)
    for beta in betas:
        Y_pu = Y.copy()
        flip = np.random.rand(n_samples)
        Y_pu[(Y != 0) & (flip < beta)] = 0
        Y_pus.append((beta, Y_pu))
        if random_seed is not None:
            print('Positive (beta={}):'.format(beta),
                  np.sum(Y), ' vs.', np.sum(Y_pu))
    return Y_pus

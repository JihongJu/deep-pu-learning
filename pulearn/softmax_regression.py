"""Softmax regression with SGD."""
# Sebastian Raschka 2016
# Implementation of the mulitnomial logistic regression algorithm for
# classification.

# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np


class SoftmaxRegression(object):
    """Linear Softmax regression classifier.

    Parameters
    ------------
    n_classes : int
        A positive integer to declare the number of class labels
        if not all class labels are present in a partial training set.
        Gets the number of class labels automatically if None.
    learning_rate : float
        Learning rate (between 0.0 and 1.0). Defaults to 0.01.
    epochs : int
        The number of passes over the training data. Defaults to 100.
    alpha : float
        Constant that multiplies the L2 regularization term. Defaults to 0.0.
    batch_size: int
        Size of minibatches for stochastic optimizers. Defaults to 200.
    class_weight: dict, {class_label: weight} or "balanced" or None, optional
        Preset for the class_weight fit parameter.

        Weights associated with classes. If not given, all classes are
        supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as 1 / (np.bincount(y) + 1)
    random_state : int seed, RandomState instance, or None (default)
        The seed of the pseudo random number generator to use when shuffling
        the data.

    Attributes
    -----------
    w_ : 2d-array, shape={n_features, 1}
      Model weights after fitting.
    b_ : 1d-array, shape={1,}
      Bias unit after fitting.
    cost_ : list
        List of floats, the average cross_entropy for each epoch.

    """

    def __init__(self, n_classes,
                 learning_rate=0.01,
                 epochs=100,
                 alpha=0.0,
                 batch_size=1,
                 class_weight=None,
                 random_state=None):
        """Init."""
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.alpha = alpha
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.class_weight = class_weight
        self.random_state = random_state

    def _fit(self, X, y, init_params=True):
        if init_params:
            if self.n_classes is None:
                self.n_classes = np.max(y) + 1
            self._n_features = X.shape[1]

            self.b_, self.w_ = self._init_params(
                weights_shape=(self._n_features, self.n_classes),
                bias_shape=(self.n_classes,),
                random_state=self.random_state)
            self.cost_ = []

        y_enc = self._one_hot(y=y, n_labels=self.n_classes, dtype=np.float)

        for i in range(self.epochs):
            self.i_ = i
            for idx in self._yield_minibatches_idx(
                    batch_size=self.batch_size,
                    data_ary=y,
                    shuffle=True):
                # givens:
                # w_ -> n_feat x n_classes
                # b_  -> n_classes

                # net_input, softmax and diff -> n_samples x n_classes:
                net = self._net_input(X[idx], self.w_, self.b_)
                prob = self._softmax(net)
                diff = self._diff(prob, y_enc[idx])

                # gradient -> n_features x n_classes
                grad = np.dot(X[idx].T, diff)

                # update in opp. direction of the cost gradient
                self.w_ -= (self.learning_rate * grad +
                            self.learning_rate * self.alpha * self.w_)
                self.b_ -= (self.learning_rate * np.sum(diff, axis=0))

            # compute cost of the whole epoch
            net = self._net_input(X, self.w_, self.b_)
            prob = self._softmax(net)
            cost = self._cost(prob, y_enc)
            self.cost_.append(cost)
        return self

    def fit(self, X, y, init_params=True):
        """Learn model from training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples], or [n_samples, n_classes]
            Target values.
        init_params : bool (default: True)
            Re-initializes model parametersprior to fitting.
            Set False to continue training with weights from
            a previous model fitting.

        Returns
        -------
        self : object

        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self._fit(X=X, y=y, init_params=init_params)
        self._is_fitted = True
        return self

    def _predict(self, X):
        probas = self.predict_proba(X)
        return self._to_classlabels(probas)

    def predict(self, X):
        """Predict targets from X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        target_values : array-like, shape = [n_samples]
          Predicted target values.

        """
        if not self._is_fitted:
            raise AttributeError('Model is not fitted, yet.')
        return self._predict(X)

    def predict_proba(self, X):
        """Predict class probabilities of X from the net input.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        Class probabilties : array-like, shape= [n_samples, n_classes]

        """
        net = self._net_input(X, self.w_, self.b_)
        softm = self._softmax(net)
        return softm

    def calc_loss(self, X, y):
        """Return loss."""
        prob = self.predict_proba(X)
        y_enc = self._one_hot(y, self.n_classes)
        loss = self._loss(prob, y_enc)
        return loss

    def calc_gradient(self, X, y):
        """Return gradient w.r.t output (before activation)."""
        prob = self.predict_proba(X)
        y_enc = self._one_hot(y, self.n_classes)
        return self._diff(prob, y_enc)

    def _net_input(self, X, W, b):
        return (X.dot(W) + b)

    def _softmax(self, z):
        z -= np.max(z, axis=1, keepdims=True)  # stable softmax
        return (np.exp(z.T) / np.sum(np.exp(z), axis=1)).T

    def _cross_entropy(self, prob, y_enc):
        prob = prob.clip(min=1e-16)
        return - (y_enc) * np.log(prob)

    def _cost(self, prob, y_enc):
        loss = self._loss(prob, y_enc)
        loss = loss + self.alpha * np.sum(self.w_ ** 2)
        return 0.5 * np.mean(loss)

    def _loss(self, prob, y_enc):
        """Return the cross entropy loss.

        Parameters
        ----------
        prob: 2D array of shape (N, K).
        y_enc: 2D array of shape (N, K)

        Return
        ------
        loss: 2D array of shape (N, K).

        """
        loss = self._cross_entropy(prob, y_enc)
        loss = np.sum(loss, axis=1)
        sample_weight = self._to_sample_weight(y_enc, tiled=False)
        loss = loss * sample_weight
        return loss

    def _diff(self, prob, y_enc):
        """Return the cross entropy derivatives.

        Parameters
        ----------
        prob: 2D array of shape (N, K).
        y_enc: 2D array of shape (N, K)

        Return
        ------
        diff: 2D array of shape (N, K).

        """
        diff = prob - y_enc
        sample_weight = self._to_sample_weight(y_enc, tiled=True)
        diff = diff * sample_weight
        return diff

    def _to_classlabels(self, z):
        return z.argmax(axis=1)

    def _init_params(self, weights_shape, bias_shape=(1,), dtype='float64',
                     scale=0.01, random_state=None):
        """Initialize weight coefficients."""
        if random_state:
            np.random.seed(random_state)
        w = np.random.normal(loc=0.0, scale=scale, size=weights_shape)
        b = np.zeros(shape=bias_shape)
        return b.astype(dtype), w.astype(dtype)

    def _one_hot(self, y, n_labels, dtype="float"):
        """One hot encoding."""
        y = np.asarray(y, dtype="int")
        if len(y.shape) == 1:
            return np.eye(n_labels, dtype=dtype)[y]
        else:
            return y

    def _yield_minibatches_idx(self, batch_size, data_ary, shuffle=True):
        n_samples = data_ary.shape[0]
        indices = np.arange(n_samples)

        remainder = data_ary.shape[0] % batch_size
        n_batches = int(n_samples / batch_size)

        if shuffle:
            indices = np.random.permutation(indices)
        if n_batches > 1:
            if remainder:
                minis = np.array_split(indices[:-remainder], n_batches)
                minis[-1] = np.concatenate((minis[-1],
                                            indices[-remainder:]),
                                           axis=0)
            else:
                minis = np.array_split(indices, n_batches)

        else:
            minis = (indices,)

        for idx_batch in minis:
            yield idx_batch

    def _to_class_weight(self, y_enc):
        n_samples, n_classes = y_enc.shape
        if self.class_weight is None:
            class_weight = np.ones(n_classes)
        elif self.class_weight is "balanced":
            class_weight = np.ones(n_classes) / (np.sum(y_enc, axis=0) + 1)
        else:
            classes = sorted(self.class_weight.keys())
            if len(classes) != n_classes:
                raise ValueError("The length of class_weight has to "
                                 "match to n_classes.")
            class_weight = np.asarray([self.class_weight[k] for k in classes])
        class_weight = class_weight.astype('float')
        class_weight = class_weight / np.max(class_weight)
        return class_weight

    def _to_sample_weight(self, y_enc, tiled=True):
        n_samples, n_classes = y_enc.shape
        class_weight = self._to_class_weight(y_enc)
        sample_weight = np.ones(y_enc.shape)
        for cl in range(n_classes):
            idx = np.where(y_enc[:, cl])[0]
            sample_weight[idx, :] = class_weight[cl]
        if tiled is True:
            return sample_weight
        return sample_weight[:, 0]

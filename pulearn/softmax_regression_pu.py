"""Positives and unlabelled learning."""
import numpy as np
from .softmax_regression import SoftmaxRegression


class WeightedUnlabelledSoftmaxRegression(SoftmaxRegression):
    """Weight unlabelled samples."""

    def __init__(self, n_classes,
                 learning_rate=0.01,
                 epochs=100,
                 alpha=0.0,
                 batch_size=1,
                 class_weight=None,
                 unlabelled_weight=None,
                 random_state=None):
        """Init."""
        self.unlabelled_weight = unlabelled_weight
        super(WeightedUnlabelledSoftmaxRegression, self).__init__(
            n_classes,
            learning_rate,
            epochs,
            alpha,
            batch_size,
            class_weight,
            random_state)

    def _to_class_weight(self, y_enc):
        class_weight = super(WeightedUnlabelledSoftmaxRegression,
                             self)._to_class_weight(y_enc)
        if self.unlabelled_weight is not None:
            n_samples, n_classes = y_enc.shape
            classes = sorted(self.unlabelled_weight.keys())
            if len(classes) != n_classes:
                raise ValueError("The length of class_weight has to "
                                 "match to n_classes.")
            unlabelled_weight = np.asarray(
                [self.unlabelled_weight[k] for k in classes], dtype="float")
            class_weight = class_weight * unlabelled_weight
            class_weight = class_weight / np.max(class_weight)
            return class_weight
        else:
            return class_weight


class HardBootstrappingSoftmaxRegression(SoftmaxRegression):
    """Hard Bootstrapping Softmax Regression."""

    def __init__(self, n_classes,
                 unlabelled_weight=None,
                 learning_rate=0.01,
                 epochs=100,
                 alpha=0.0,
                 batch_size=1,
                 class_weight=None,
                 random_state=None):
        """Init."""
        self.unlabelled_weight = unlabelled_weight
        super(HardBootstrappingSoftmaxRegression, self).__init__(
            n_classes,
            learning_rate,
            epochs,
            alpha,
            batch_size,
            class_weight,
            random_state)

    def _loss(self, prob, y_enc):
        loss = self._cross_entropy(prob, y_enc)

        pred = np.argmax(prob, axis=1)
        pred_enc = self._one_hot(
            y=pred, n_labels=self.n_classes, dtype=np.int)
        cross_ent_pred = self._cross_entropy(prob, pred_enc)
        betas = self._to_unlabelled_sample_weight(y_enc)
        loss = loss * betas + (1 - betas) * cross_ent_pred

        loss = np.sum(loss, axis=1)
        sample_weight = self._to_sample_weight(y_enc, tiled=False)
        loss = loss * sample_weight
        return loss

    def _diff(self, prob, y_enc):
        diff = prob - y_enc

        pred = np.argmax(prob, axis=1)
        pred_enc = self._one_hot(
            y=pred, n_labels=self.n_classes, dtype=np.int)
        betas = self._to_unlabelled_sample_weight(y_enc)
        diff = betas * diff + \
            (1 - betas) * (prob - pred_enc)

        sample_weight = self._to_sample_weight(y_enc, tiled=True)
        diff = diff * sample_weight
        return diff

    def _to_unlabelled_weight(self, y_enc, tiled=True):
        n_samples, n_classes = y_enc.shape
        if self.unlabelled_weight is None:
            unlabelled_weight = np.ones(n_classes)
        else:
            classes = sorted(self.unlabelled_weight.keys())
            if len(classes) != n_classes:
                raise ValueError("The length of unlabelled_weight has to "
                                 "match to n_classes.")
            unlabelled_weight = np.asarray(
                [self.unlabelled_weight[k] for k in classes])
        unlabelled_weight = unlabelled_weight.astype('float')
        unlabelled_weight /= np.max(unlabelled_weight)
        return unlabelled_weight

    def _to_unlabelled_sample_weight(self, y_enc):
        n_samples, n_classes = y_enc.shape
        unlabelled_weight = self._to_unlabelled_weight(y_enc)
        unlabelled_sample_weight = np.ones(y_enc.shape)
        for cl in range(n_classes):
            idx = np.where(y_enc[:, cl])[0]
            unlabelled_sample_weight[idx, :] = unlabelled_weight[cl]

        return unlabelled_sample_weight


class UnlabelledExponentialLossSoftmaxRegression(
        WeightedUnlabelledSoftmaxRegression):
    """Use exponential loss for unlabelled samples."""

    def _loss(self, prob, y_enc):
        loss = self._cross_entropy(prob, y_enc)

        negidx = np.where(y_enc[:, 0])[0]
        loss[negidx, :] = 0
        loss[negidx, 0] = (1 - prob[negidx, 0])

        loss = np.sum(loss, axis=1)
        sample_weight = self._to_sample_weight(y_enc, tiled=False)
        loss = loss * sample_weight

        return loss

    def _diff(self, prob, y_enc):
        diff = prob - y_enc
        negidx = np.where(y_enc[:, 0])[0]
        for cl in range(self.n_classes):
            if cl == 0:
                diff[negidx, cl] = prob[negidx, 0] * (prob[negidx, cl] - 1)
            else:
                diff[negidx, cl] = prob[negidx, 0] * prob[negidx, cl]

        sample_weight = self._to_sample_weight(y_enc, tiled=True)
        diff = diff * sample_weight

        return diff


# class WeightedUSoftmaxRegression(SoftmaxRegression):
#     """Weighted unlabeled samples learning."""
#
#     def __init__(self, r_unlabeled=3, a_unlabeled=0., b_unlabeled=1.,
#                  eta=0.01, epochs=100,
#                  l2=0.0,
#                  minibatches=1,
#                  n_classes=None,
#                  class_weight=None,
#                  random_seed=None):
#         """Init."""
#         super(WeightedUSoftmaxRegression, self).__init__(eta, epochs,
#                                                          l2, minibatches,
#                                                          n_classes,
#                                                          class_weight,
#                                                          random_seed)
#         self.r_unlabeled = float(r_unlabeled)
#         self.a_unlabeled = float(a_unlabeled)
#         self.b_unlabeled = float(b_unlabeled)
#
#     def _loss(self, prob, y_enc):
#         loss = self._cross_entropy(prob, y_enc)
#         loss = np.sum(loss, axis=1)
#         if self.a_unlabeled is not None:
#             sample_weights = self._sample_weights(prob, y_enc, tiled=False)
#             loss = loss * sample_weights
#         return loss
#
#     def _diff(self, prob, y_enc):
#         diff = prob - y_enc
#         if self.a_unlabeled is not None:
#             sample_weights = self._sample_weights(prob, y_enc, tiled=True)
#             diff = diff * sample_weights
#         return diff
#
#     def _sample_weights(self, prob, y_enc, tiled=True):
#         sample_weights = np.ones(y_enc.shape)
#         unl_idx = np.where(y_enc[:, 0])[0]
#         sample_weights[unl_idx, :] = (self.b_unlabeled - self.a_unlabeled)\
#             * np.power(prob[unl_idx, :], self.r_unlabeled) \
#             + self.a_unlabeled
#         if tiled is False:
#             return sample_weights[:, 0]
#         return sample_weights

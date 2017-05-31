"""Positives and unlabelled learning."""
import numpy as np
from softmax_regression import SoftmaxRegression


class HardBootstrappingSoftmaxRegression(SoftmaxRegression):
    """Hard Bootstrapping Softmax Regression."""

    def __init__(self, betas, eta=0.01, epochs=100,
                 l2=0.0,
                 minibatches=1,
                 n_classes=None,
                 class_weight=None,
                 unbalanced=None,
                 random_seed=None):
        """Init."""
        super(HardBootstrappingSoftmaxRegression, self).__init__(eta, epochs,
                                                                 l2,
                                                                 minibatches,
                                                                 n_classes,
                                                                 class_weight,
                                                                 unbalanced,
                                                                 random_seed)
        self.betas = betas

    def _loss(self, prob, y_enc):
        loss = self._cross_entropy(prob, y_enc)
        if self.betas is not None:
            pred = np.argmax(prob, axis=1)
            pred_enc = self._one_hot(
                y=pred, n_labels=self.n_classes, dtype=np.int)
            cross_ent_pred = self._cross_entropy(prob, pred_enc)
            betas = self._get_betas(y_enc, tiled=True)
            loss = loss * betas + (1 - betas) * cross_ent_pred
        loss = np.sum(loss, axis=1)
        if self.class_weight is not None:
            sample_weights = self._get_sample_weights(y_enc, tiled=False)
            loss = loss * sample_weights
        return loss

    def _diff(self, prob, y_enc):
        diff = prob - y_enc
        if self.betas is not None:
            pred = np.argmax(prob, axis=1)
            pred_enc = self._one_hot(
                y=pred, n_labels=self.n_classes, dtype=np.int)
            betas = self._get_betas(y_enc, tiled=True)
            diff = betas * (prob - y_enc) + \
                (1 - betas) * (prob - pred_enc)
        if self.class_weight is not None:
            sample_weights = self._get_sample_weights(y_enc, tiled=True)
            diff = diff * sample_weights
        return diff

    def _get_betas(self, y_enc, tiled=True):
        betas = np.array(self.betas).astype('float')
        betas /= float(np.max(betas))
        sample_betas = np.ones(y_enc.shape)
        for cl in range(self.n_classes):
            idx = np.where(y_enc[:, cl])[0]
            sample_betas[idx, :] = betas[cl]
        if tiled is True:
            return sample_betas
        return sample_betas[:, 0]


class ClassDepLossSoftmaxRegression(SoftmaxRegression):
    """Class dependent loss SoftmaxRegression."""

    def __init__(self, eta=0.01, epochs=100,
                 l2=0.0,
                 minibatches=1,
                 n_classes=None,
                 class_weight=None,
                 unbalanced=None,
                 random_seed=None):
        """Init."""
        super(ClassDepLossSoftmaxRegression, self).__init__(eta, epochs,
                                                            l2,
                                                            minibatches,
                                                            n_classes,
                                                            class_weight,
                                                            unbalanced,
                                                            random_seed)

    def _loss(self, prob, y_enc):
        loss = self._cross_entropy(prob, y_enc)
        negidx = np.where(y_enc[:, 0])[0]
        loss[negidx, :] = 0   # redundant
        loss[negidx, 0] = (1 - prob[negidx, 0])
        loss = np.sum(loss, axis=1)
        if self.class_weight is not None:
            sample_weights = self._get_sample_weights(y_enc, tiled=False)
            loss = loss * sample_weights
        return loss

    def _diff(self, prob, y_enc):
        diff = prob - y_enc
        negidx = np.where(y_enc[:, 0])[0]
        for cl in range(self.n_classes):
            if cl == 0:
                diff[negidx, cl] = prob[negidx, 0] * (prob[negidx, cl] - 1)
            else:
                diff[negidx, cl] = prob[negidx, 0] * prob[negidx, cl]
        if self.class_weight is not None:
            sample_weights = self._get_sample_weights(y_enc, tiled=True)
            diff = diff * sample_weights
        return diff


class WeightedUSoftmaxRegression(SoftmaxRegression):
    """Weighted unlabeled samples learning."""

    def __init__(self, r_unlabeled=3, a_unlabeled=0., b_unlabeled=1.,
                 eta=0.01, epochs=100,
                 l2=0.0,
                 minibatches=1,
                 n_classes=None,
                 class_weight=None,
                 random_seed=None):
        """Init."""
        super(WeightedUSoftmaxRegression, self).__init__(eta, epochs,
                                                         l2, minibatches,
                                                         n_classes,
                                                         class_weight,
                                                         random_seed)
        self.r_unlabeled = float(r_unlabeled)
        self.a_unlabeled = float(a_unlabeled)
        self.b_unlabeled = float(b_unlabeled)

    def _loss(self, prob, y_enc):
        loss = self._cross_entropy(prob, y_enc)
        loss = np.sum(loss, axis=1)
        if self.a_unlabeled is not None:
            sample_weights = self._sample_weights(prob, y_enc, tiled=False)
            loss = loss * sample_weights
        return loss

    def _diff(self, prob, y_enc):
        diff = prob - y_enc
        if self.a_unlabeled is not None:
            sample_weights = self._sample_weights(prob, y_enc, tiled=True)
            diff = diff * sample_weights
        return diff

    def _sample_weights(self, prob, y_enc, tiled=True):
        sample_weights = np.ones(y_enc.shape)
        unl_idx = np.where(y_enc[:, 0])[0]
        sample_weights[unl_idx, :] = (self.b_unlabeled - self.a_unlabeled)\
            * np.power(prob[unl_idx, :], self.r_unlabeled) \
            + self.a_unlabeled
        if tiled is False:
            return sample_weights[:, 0]
        return sample_weights

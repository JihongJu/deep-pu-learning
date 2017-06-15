import numpy as np
import tensorflow as tf
import pulearn.losses.tensorflow_losses as L
from .multilayer_perceptron import MultilayerPerceptron


class WeightedUnlabelledMultilayerPerceptron(MultilayerPerceptron):
    """Weighted unlabelled samples."""

    def __init__(self, n_input, n_classes, n_hiddens=[8, 8],
                 learning_rate=0.01,
                 batch_size=100,
                 epochs=30, alpha=1e-3,
                 display_step=10,
                 class_weight=None,
                 unlabelled_weight=None,
                 verbose=None):
        """Init."""
        self.unlabelled_weight = unlabelled_weight
        super(WeightedUnlabelledMultilayerPerceptron, self).__init__(
            n_input,
            n_classes,
            n_hiddens,
            learning_rate,
            batch_size,
            epochs,
            alpha,
            display_step,
            class_weight,
            verbose)

    def _to_class_weight(self, Y):
        class_weight = super(WeightedUnlabelledMultilayerPerceptron,
                             self)._to_class_weight(Y)
        if self.unlabelled_weight is not None:
            n_samples, n_classes = Y.shape
            classes = sorted(self.unlabelled_weight.keys())
            if len(classes) != n_classes:
                raise ValueError("The length of class_weight has to "
                                 "match to n_classes.")
            unlabelled_weight = np.asarray(
                [self.unlabelled_weight[k] for k in classes], dtype="float")
            class_weight = class_weight * unlabelled_weight
            class_weight = class_weight / np.min(class_weight)
            if self.verbose:
                print("Re-weighing to {}".format(class_weight))
            return class_weight
        else:
            return class_weight


class UnlabelledExponentialLossMultilayerPerceptron(
        WeightedUnlabelledMultilayerPerceptron):
    """Use exponential loss for unlabelled samples."""

    def _loss(self, out, y, class_weight):
        loss = L.cross_entropy_and_exponential_loss_with_logits(
            logits=out,
            labels=y)
        loss = self._balance(loss, y, class_weight)
        return loss


class HardBootstrappingMultilayerPerceptron(MultilayerPerceptron):
    """'Hard Boostrapping' method from ."""

    def __init__(self, n_input, n_classes, n_hiddens=[8, 8],
                 learning_rate=0.01,
                 batch_size=100,
                 epochs=30, alpha=1e-3,
                 display_step=10,
                 class_weight=None,
                 unlabelled_weight=None,
                 verbose=None):
        """Init."""
        self.unlabelled_weight = unlabelled_weight
        super(HardBootstrappingMultilayerPerceptron, self).__init__(
            n_input,
            n_classes,
            n_hiddens,
            learning_rate,
            batch_size,
            epochs,
            alpha,
            display_step,
            class_weight,
            verbose)

    def _loss(self, out, y, class_weight):
        loss = tf.nn.softmax_cross_entropy_with_logits(
            logits=out, labels=y)
        y_cat = tf.argmax(y, axis=1)

        prob = tf.nn.softmax(out, dim=-1)
        y_pred = tf.argmax(prob, axis=1)
        y_pred_enc = tf.one_hot(y_pred, self.n_classes, dtype=tf.float32)
        loss_pred = tf.nn.softmax_cross_entropy_with_logits(
            logits=out, labels=y_pred_enc)

        n_samples, n_classes = y.get_shape()
        unlabelled_weight = self._to_unlabelled_weight()

        for k in range(n_classes):
            b_k = unlabelled_weight[k]
            if b_k < 1.:
                ks = tf.scalar_mul(k, tf.ones_like(y_cat))
                mask_k = tf.equal(y_cat, ks)
                loss_k = tf.add(
                    tf.scalar_mul(b_k, loss),
                    tf.scalar_mul(1 - b_k, loss_pred))
                loss = tf.where(mask_k, loss_k, loss)

        loss = self._balance(loss, y, class_weight)
        return loss

    def _to_unlabelled_weight(self):
        n_classes = self.n_classes
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
        unlabelled_weight = unlabelled_weight / np.max(unlabelled_weight)
        if self.verbose:
            print("Using {} for betas.".format(unlabelled_weight))
        return unlabelled_weight

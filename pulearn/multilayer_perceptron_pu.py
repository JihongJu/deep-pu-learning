import numpy as np
import tensorflow as tf
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
        loss = self._cross_entropy_and_exponential_loss_with_logits(
            logits=out,
            labels=y,
            class_weight=class_weight
        )
        loss = tf.reduce_sum(loss, axis=1)
        return loss

    def _cross_entropy_and_exponential_loss_with_logits(self, logits, labels,
                                                        class_weight):
        """Class dependent loss.

        Using cross entropy for P and absolute difference for U.
        """
        loss_pos = tf.nn.weighted_cross_entropy_with_logits(
            logits=logits,
            targets=labels,
            pos_weight=class_weight)
        prob = tf.nn.softmax(logits)
        loss_neg = tf.multiply(
            labels,
            tf.subtract(labels, prob))

        is_negative = labels[:, 0] > 0.5
        n_features = loss_pos.get_shape()[1]
        negative_mask = tf.stack([is_negative] * int(n_features), axis=1)
        loss = tf.where(negative_mask, loss_neg, loss_pos)

        return loss

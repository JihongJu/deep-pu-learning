"""PU learning modules."""
import numpy as np
import tensorflow as tf
from multilayer_perceptron import MultilayerPerceptron


def cross_entropy_and_difference_with_logits(logits, labels, weights):
    """Class dependent loss.

    Using cross entropy for P and absolute difference for U.
    """
    loss_pos = tf.nn.weighted_cross_entropy_with_logits(
        logits=logits,
        targets=labels,
        pos_weight=weights)
    loss_neg = tf.multiply(
        labels,
        tf.subtract(labels, logits))

    is_negative = labels[:, 0] > 0.5
    n_features = loss_pos.get_shape()[1]
    negative_mask = tf.stack([is_negative] * int(n_features), axis=1)
    loss = tf.where(negative_mask, loss_neg, loss_pos)

    return loss


class ClassDepLossMultilayerPerceptron(MultilayerPerceptron):
    """Class dependent loss MLP."""

    def _loss(self, prob, y, class_weight):
        loss = tf.reduce_mean(
            cross_entropy_and_difference_with_logits(
                logits=prob,
                labels=y,
                weights=class_weight
            ))
        return loss

    def _cost(self, prob, y, class_weight):
        self.loss = self._loss(prob, y, class_weight)
        regularizer = tf.constant(0.)
        for w in self.weights.values():
            regularizer = tf.add(regularizer, tf.nn.l2_loss(w))
        cost = tf.reduce_mean(self.loss + self.regularization * regularizer)
        return cost


class HardBoostrappingMultilayerPerceptron(MultilayerPerceptron):
    """Hard Boostrapping MultilayerPerceptron."""

    def __init__(self, n_input, n_classes, n_hiddens=[8, 8],
                 learning_rate=0.01, batch_size=100,
                 training_epochs=30, regularization=1e-3,
                 display_step=10,
                 betas=None,
                 class_weight=None,
                 imbalanced=None, verbose=None):
        """Init."""
        if betas is None:
            self._betas = np.ones(n_classes, dtype="float")
        else:
            self._betas = np.array(betas, dtype="float")
        self._betas /= np.max(self._betas)
        super(HardBoostrappingMultilayerPerceptron, self).__init__(n_input,
                                                                   n_classes,
                                                                   n_hiddens,
                                                                   learning_rate,
                                                                   batch_size,
                                                                   training_epochs,
                                                                   regularization,
                                                                   display_step,
                                                                   class_weight,
                                                                   imbalanced,
                                                                   verbose)

    def _cost(self, prob, y, class_weight):
        class_weight = tf.cast(class_weight, dtype=tf.float32)
        y_pred = tf.argmax(prob, 1)
        y_pred_enc = tf.one_hot(y_pred, self.n_classes, dtype=tf.float32)

        betas = []
        for k in range(self.n_classes):
            beta_k = prob[:, 0] > -1
            beta_k = tf.cast(beta_k, dtype=tf.float32)
            beta_k = tf.multiply(tf.constant(self._betas[k], dtype=tf.float32),
                                 beta_k)
            betas.append(beta_k)
        betas = tf.stack(betas, axis=1)
        labl_loss = tf.multiply(
            betas,
            tf.nn.weighted_cross_entropy_with_logits(targets=y,
                                                     logits=prob,
                                                     pos_weight=class_weight)
        )
        pred_loss = tf.multiply(
            tf.subtract(tf.constant(1., dtype=tf.float32), betas),
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=y_pred_enc,
                logits=prob)
        )
        loss = tf.add(labl_loss, pred_loss)
        sample_weights = self._get_sample_weights(y, class_weight)
        loss = tf.multiply(
            sample_weights,
            loss
        )
        regularizer = tf.constant(0.)
        for w in self.weights.values():
            regularizer = tf.add(regularizer, tf.nn.l2_loss(w))
            cost = tf.reduce_mean(loss + self.regularization * regularizer)
        return cost

    def _get_sample_weights(self, labels, class_weight):
        sample_weights = tf.ones_like(labels)
        for k in range(self.n_classes):
            weights_k = tf.multiply(
                class_weight[k],
                sample_weights
            )
            is_k = labels[:, k] > 0.
            n_features = labels.get_shape()[1]
            mask_k = tf.stack([is_k] * int(n_features), axis=1)
            sample_weights = tf.where(mask_k, weights_k, sample_weights)
        return sample_weights

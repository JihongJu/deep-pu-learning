"""PU learning modules."""
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

    is_negative = labels[:, 0] > 0.
    n_features = loss_pos.get_shape()[1]
    negative_mask = tf.stack([is_negative] * int(n_features), axis=1)
    loss = tf.where(negative_mask, loss_neg, loss_pos)

    return loss


class ClassDepLossMultilayerPerceptron(MultilayerPerceptron):
    """Class dependent loss."""

    def _cost(self, prob, y, class_weight):
        loss = tf.reduce_mean(
            cross_entropy_and_difference_with_logits(
                logits=prob,
                labels=y,
                weights=class_weight
            ))
        regularizer = tf.constant(0.)
        for w in self.weights.values():
            regularizer = tf.add(regularizer, tf.nn.l2_loss(w))
        cost = tf.reduce_mean(loss + self.beta * regularizer)
        return cost

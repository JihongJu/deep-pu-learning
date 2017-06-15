import tensorflow as tf


_FLOATX = 'float32'
_EPSILON = 10e-8


def cross_entropy_and_exponential_loss_with_logits(logits, labels):
    """Class dependent loss.

    Using cross entropy for P and absolute difference for U.
    """
    loss_pos = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels)
    probs = tf.nn.softmax(logits)
    loss_neg = exponential_loss_with_probs(probs[:, 0], labels[:, 0])
    y_cat = tf.argmax(labels, axis=-1)
    negative_mask = tf.equal(y_cat, tf.zeros_like(y_cat))
    loss = tf.where(negative_mask, loss_neg, loss_pos)

    return loss


def cross_entropy_and_exponential_loss(probs, labels):
    """Class dependent loss.

    Using cross entropy for P and absolute difference for U.
    """
    loss_pos = softmax_cross_entropy_with_probs(
        probs=probs,
        labels=labels)
    loss_neg = exponential_loss_with_probs(probs[:, 0], labels[:, 0])
    y_cat = tf.argmax(labels, axis=-1)
    negative_mask = tf.equal(y_cat, tf.zeros_like(y_cat))
    loss = tf.where(negative_mask, loss_neg, loss_pos)

    return loss


def softmax_cross_entropy_with_probs(probs, labels):
    """Softmax cross entropy loss with probabilities."""
    probs /= tf.reduce_sum(probs, axis=-1, keep_dims=True)
    # manual computation of crossentropy
    epsilon = tf.convert_to_tensor(_EPSILON, dtype=probs.dtype.base_dtype)
    probs = tf.clip_by_value(probs, epsilon, 1. - epsilon)
    return - tf.reduce_sum(labels * tf.log(probs), axis=-1)


def exponential_loss_with_probs(probs, labels):
    """Expoential loss with probabilities."""
    return tf.multiply(labels,
                       tf.subtract(labels, probs))

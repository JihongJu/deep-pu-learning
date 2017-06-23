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


def hard_bootstrapping_with_probs(probs, labels, betas):
    """Hard Bootstrapping Loss."""
    loss = softmax_cross_entropy_with_probs(
        probs=probs, labels=labels)
    y_cat = tf.argmax(labels, axis=1)

    n_samples, n_classes = labels.get_shape()

    y_pred = tf.argmax(probs, axis=1)
    y_pred_enc = tf.one_hot(y_pred, n_classes, dtype=tf.float32)
    loss_pred = softmax_cross_entropy_with_probs(
        probs=probs, labels=y_pred_enc)

    for k in range(n_classes):
        b_k = betas[k]
        if b_k < 1.:
            ks = tf.scalar_mul(k, tf.ones_like(y_cat))
            mask_k = tf.equal(y_cat, ks)
            loss_k = tf.add(
                tf.scalar_mul(b_k, loss),
                tf.scalar_mul(1 - b_k, loss_pred))
            loss = tf.where(mask_k, loss_k, loss)
    return loss

import keras.backend as K
from keras.losses import categorical_crossentropy
from .tensorflow_losses import (
    cross_entropy_and_exponential_loss,
    hard_bootstrapping_with_probs
)


def unlabeled_exponential_loss(y_true, y_pred):
    """Class dependent loss.

    Using cross entropy for P and absolute difference for U.
    """
    return cross_entropy_and_exponential_loss(probs=y_pred,
                                              labels=y_true)


def hard_bootstrapping_loss(betas):
    """Hard Bootstrapping loss."""
    def f(y_true, y_pred):
        return hard_bootstrapping_with_probs(y_true, y_pred, betas)
    return f

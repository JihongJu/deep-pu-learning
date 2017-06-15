import keras.backend as K
from .tensorflow_losses import cross_entropy_and_exponential_loss


def unlabelled_exponential_loss(y_true, y_pred):
    """Class dependent loss.

    Using cross entropy for P and absolute difference for U.
    """
    return cross_entropy_and_exponential_loss(probs=y_pred,
                                              labels=y_true)

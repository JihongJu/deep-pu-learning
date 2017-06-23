import keras.backend as K
from keras.losses import categorical_crossentropy
from .tensorflow_losses import cross_entropy_and_exponential_loss


def unlabeled_exponential_loss(y_true, y_pred):
    """Class dependent loss.

    Using cross entropy for P and absolute difference for U.
    """
    return cross_entropy_and_exponential_loss(probs=y_pred,
                                              labels=y_true)



def fade_in_unlabeled_exponential_loss(y_true, y_pred):
    """Controll unlabeled exponential loss fade in with alpha."""
    return alpha * categorical_crossentropy(y_true, y_pred) \
            + (1 - alpha) * unlabeled_exponential_loss(y_true, y_pred)

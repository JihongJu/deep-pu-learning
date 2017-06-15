import numpy as np
import keras.backend as K
import pulearn.losses.keras_losses as L


def test_cross_entropy_and_exponential_loss():
    y_pred = K.variable([[.5, .5], [.8, .2]], dtype=K.floatx())
    y_true = K.variable([[1., 0.], [.0, 1.]], dtype=K.floatx())

    loss = L.cross_entropy_and_exponential_loss(y_pred, y_true)
    assert np.allclose(K.eval(loss), [0.5, 1.60943794])

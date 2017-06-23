import numpy as np
import tensorflow as tf
import pulearn.losses.tensorflow_losses as L


def test_hard_bootstrapping_with_probs():
    y_pred = tf.Variable([[.4, .6], [.8, .2]], dtype=tf.float32)
    y_true = tf.Variable([[1., 0.], [.0, 1.]], dtype=tf.float32)
    betas = np.array([0.5, 1.])

    loss = L.hard_bootstrapping_with_probs(y_pred, y_true, betas)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loss = loss.eval()
        assert np.allclose(loss, [0.71355814, 1.60943794])

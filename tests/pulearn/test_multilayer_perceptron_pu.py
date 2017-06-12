import numpy as np
import tensorflow as tf
from pulearn import (
    WeightedUnlabelledMultilayerPerceptron,
    HardBootstrappingMultilayerPerceptron,
    UnlabelledExponentialLossMultilayerPerceptron
)


def test_hardbootrapping():
    hbmlp = HardBootstrappingMultilayerPerceptron(n_input=2, n_classes=2)

    # test_to_unlabelled_weight
    hbmlp.unlabelled_weight = {0: 0.5, 1: 1}
    assert np.allclose(hbmlp._to_unlabelled_weight(),
                       [0.5, 1.])

    # test_loss
    out = tf.Variable([[-5., 5.], [8, -2.]], dtype=tf.float32)
    y_enc = tf.Variable([[1., 0.], [1., 0.]], dtype=tf.float32)
    class_weight = tf.Variable([1., 1.], dtype=tf.float32)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loss = sess.run(hbmlp._loss(out, y_enc, class_weight))
        print(loss)
        assert np.allclose(loss, [5.00004578, 4.54177061e-05])

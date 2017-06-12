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


def test_weighted_unlabelled():
    wumlp = WeightedUnlabelledMultilayerPerceptron(n_input=2, n_classes=2,
                                                   class_weight="balanced")
    # test_to_class_weight
    Y = np.asarray([[1, 0], [1, 0], [0, 1]])
    wumlp.unlabelled_weight = {0: 1, 1: 2}
    assert np.allclose(wumlp._to_class_weight(Y),
                       [1, 3])


def test_unlabelled_exponential_loss():
    uelmlp = UnlabelledExponentialLossMultilayerPerceptron(n_input=2,
                                                           n_classes=2)

    out = tf.Variable([[-5., 5.], [8, -2.]], dtype=tf.float32)
    y_enc = tf.Variable([[1., 0.], [1., 0.]], dtype=tf.float32)
    class_weight = tf.Variable([1., 1.], dtype=tf.float32)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loss = sess.run(uelmlp._loss(out, y_enc, class_weight))
        print(loss)
        assert np.allclose(loss, [9.99954581e-01, 4.54177061e-05])

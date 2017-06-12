import unittest
import numpy as np
import tensorflow as tf

from pulearn import MultilayerPerceptron


class TestMultilayerPerceptron(unittest.TestCase):
    def setUp(self):
        self.classifier = MultilayerPerceptron(n_input=2, n_classes=2)

    def tearDown(self):
        self.classifier.close_session()

    def test_to_class_weight(self):
        y_enc = np.asarray(
            [[0, 1],
             [1, 0],
             [0, 1]]
        )
        self.classifier.class_weight = None
        class_weight = self.classifier._to_class_weight(y_enc)
        assert np.allclose(class_weight, np.ones(2))

        self.classifier.class_weight = "balanced"
        class_weight = self.classifier._to_class_weight(y_enc)
        assert np.allclose(class_weight, [1.5, 1.])

        self.classifier.class_weight = {0: 0.5, 1: 1}
        class_weight = self.classifier._to_class_weight(y_enc)
        assert np.allclose(class_weight, [1., 2.])

        # To avoid iterative multiplication
        class_weight = self.classifier._to_class_weight(y_enc)
        assert np.allclose(class_weight, [1, 2])

    def test_loss(self):
        out = tf.Variable([[-5., 5.], [8, -2.]], dtype=tf.float32)
        y_enc = tf.Variable([[1., 0.], [1., 0.]], dtype=tf.float32)
        class_weight = tf.Variable([1., 1.], dtype=tf.float32)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            loss = sess.run(self.classifier._loss(out, y_enc, class_weight))
            print(loss)
            assert np.allclose(loss, [10., 4.54177061e-05])

    def test_balance(self):
        loss = tf.Variable(np.ones(2), dtype=tf.float32)
        y_enc = tf.Variable([[1., 0.], [0., 1.]], dtype=tf.float32)
        class_weight = tf.Variable([0.5, 1.], dtype=tf.float32)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            loss = sess.run(self.classifier._balance(
                loss, y_enc, class_weight))
            print(loss)
            assert np.allclose(loss, [0.5, 1.])

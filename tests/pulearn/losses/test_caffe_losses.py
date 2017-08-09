import unittest
import numpy as np
from mock import MagicMock, patch
from pulearn.losses.caffe_losses import (
    SoftmaxWithLossLayer,
    WeightedSoftmaxWithLossLayer,
    ExponentialWithLossLayer
)
import tensorflow as tf
import pulearn.losses.tensorflow_losses as L


class TestSoftmaxWithLossLayer(unittest.TestCase):

    @patch('pulearn.losses.caffe_losses.SoftmaxWithLossLayer.__init__',
           lambda *_: None)
    @patch('pulearn.losses.caffe_losses.WeightedSoftmaxWithLossLayer.__init__',
           lambda *_: None)
    @patch('pulearn.losses.caffe_losses.ExponentialWithLossLayer.__init__',
           lambda *_: None)
    def setUp(self):
        self.swll = SoftmaxWithLossLayer()
        self.wswll = WeightedSoftmaxWithLossLayer()
        self.ewll = ExponentialWithLossLayer()
        class_weight = {0: 0.5}
        for k in range(1, 3):
            class_weight[k] = 1.
        pyloss_params = dict(axis=1, normalization=1,
                             ignore_label=255, loss_weight=1,
                             class_weight=class_weight)
        scores = np.ones([1, 3, 2, 2]) * 0.5
        labels = np.reshape([0, 1, 2, 0], [1, 1, 2, 2])
        # labels = np.eye(3)[labels]
        # labels = np.rollaxis(labels, -1, 1).astype('int')
        self.bottom = [scores, labels]
        self.top = [np.zeros(1)]
        self.swll.param_str = str(pyloss_params)
        self.swll.setup(self.bottom, self.top)
        self.wswll.param_str = str(pyloss_params)
        self.wswll.setup(self.bottom, self.top)
        self.ewll.param_str = str(pyloss_params)
        self.ewll.setup(self.bottom, self.top)

    def test_to_sample_weight(self):
        # test _to_class_weight
        assert np.all(self.wswll.class_weight == [0.5, 1, 1])
        # test _to_sample_weight
        labels = self.bottom[1]
        assert np.all(
            self.wswll._to_sample_weight(labels) ==
            np.reshape([0.5, 1, 1, 0.5] * 3, [1, 3, 2, 2]))

    def test_compute_loss(self):
        prob = np.reshape([[0.1, 0.4, 0.4, 0.4],
                           [0.8, 0.3, 0.3, 0.3],
                           [0.1, 0.3, 0.3, 0.3]], [1, 3, 2, 2])
        label = np.reshape([0, 1, 1, 255], [1, 1, 2, 2])

        swll_expected = np.reshape([[-np.log(0.1), 0, 0, 0],
                                    [0, -np.log(0.3), -np.log(0.3), 0],
                                    [0, 0, 0, 0]], [1, 3, 2, 2])
        loss = self.swll.compute_loss(prob, label.copy())
        assert np.allclose(loss, swll_expected)

        wswll_expected = np.reshape([[-np.log(0.1) * 0.5, 0, 0, 0],
                                     [0, -np.log(0.3), -np.log(0.3), 0],
                                     [0, 0, 0, 0]], [1, 3, 2, 2])
        loss = self.wswll.compute_loss(prob, label.copy())
        assert np.allclose(loss, wswll_expected)

        ewll_expected = np.reshape([[0.9, 0, 0, 0],
                                    [0, -np.log(0.3), -np.log(0.3), 0],
                                    [0, 0, 0, 0]], [1, 3, 2, 2])
        loss = self.ewll.compute_loss(prob, label.copy())
        assert np.allclose(loss, ewll_expected)

        # test with tf loss
        tf_prob = np.reshape(prob, [3, 4])
        tf_prob = np.transpose(tf_prob, [1, 0])
        y_pred = tf.Variable(tf_prob, dtype=tf.float32)
        tf_label = np.reshape([[1, 0, 0],
                               [0, 1, 0],
                               [0, 1, 0],
                               [1, 0, 0]], [4, 3])
        y_true = tf.Variable(tf_label, dtype=tf.float32)
        tf_loss_expected = [loss[0, 0, 0, 0],
                            loss[0, 1, 0, 1], loss[0, 1, 1, 0]]
        tf_loss = L.cross_entropy_and_exponential_loss(y_pred, y_true)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            tf_loss = tf_loss.eval()
            assert np.allclose(tf_loss[:3], tf_loss_expected)

    def test_compute_diff(self):
        prob = np.reshape([[0.1, 0.4, 0.4, 0.4],
                           [0.8, 0.3, 0.3, 0.3],
                           [0.1, 0.3, 0.3, 0.3]], [1, 3, 2, 2])
        label = np.reshape([0, 1, 1, 255], [1, 1, 2, 2])

        swll_expected = np.reshape([[-0.9, 0.4, 0.4, 0.],
                                    [0.8, -0.7, -0.7, 0.],
                                    [0.1, 0.3, 0.3, 0.]], [1, 3, 2, 2])

        diff = self.swll.compute_diff(prob, label.copy())
        assert np.allclose(diff, swll_expected)

        wswll_expected = np.reshape([[-0.9 * 0.5, 0.4, 0.4, 0.],
                                     [0.8 * 0.5, -0.7, -0.7, 0.],
                                     [0.1 * 0.5, 0.3, 0.3, 0.]], [1, 3, 2, 2])
        diff = self.wswll.compute_diff(prob, label.copy())
        assert np.allclose(diff, wswll_expected)

        ewll_expected = np.reshape([[-0.09, 0.4, 0.4, 0.],
                                    [0.08, -0.7, -0.7, 0.],
                                    [0.01, 0.3, 0.3, 0.]], [1, 3, 2, 2])
        diff = self.ewll.compute_diff(prob, label.copy())
        assert np.allclose(diff, ewll_expected)

        # test with tf diff
        logit = np.reshape([-2., 5, 0], [1, 3, 1, 1])
        logit -= np.max(logit, axis=1, keepdims=True)
        score_exp = np.exp(logit)
        prob = score_exp / np.sum(score_exp, axis=1, keepdims=True)
        print(prob)
        label = np.reshape([0], [1, 1, 1, 1])
        diff = self.ewll.compute_diff(prob, label.copy())

        # test with tf diff
        tf_logit = np.reshape(logit, [3, 1])
        tf_logit = np.transpose(tf_logit, [1, 0])
        y_out = tf.Variable(tf_logit, dtype=tf.float32)
        y_pred = tf.nn.softmax(y_out)
        tf_label = np.reshape([1, 0, 0], [1, 3])
        y_true = tf.Variable(tf_label, dtype=tf.float32)
        tf_diff_expected = np.reshape(diff, [1, 3])
        tf_loss = L.cross_entropy_and_exponential_loss(y_pred, y_true)
        tf_diff = tf.gradients(tf_loss, [y_out])[0]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            tf_diff = tf_diff.eval()
            print(tf_diff, diff)
            assert np.allclose(tf_diff, tf_diff_expected)

    def test_negative_mask(self):
        label = np.reshape([0, 0, 1, 1], [1, 1, 2, 2])
        mask = self.ewll.negative_mask(label)
        expected_mask = np.reshape(
            [True, True, False, False] * 3, [1, 3, 2, 2])
        label_enc = self.ewll.one_hot_encode(label, squeeze=True)
        label_enc[mask] = 255

        assert np.allclose(expected_mask, mask)

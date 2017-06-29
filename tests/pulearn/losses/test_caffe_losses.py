import unittest
import numpy as np
from mock import MagicMock, patch
from pulearn.losses.caffe_losses import (
    SoftmaxWithLossLayer,
    WeightedSoftmaxWithLossLayer
)


class TestSoftmaxWithLossLayer(unittest.TestCase):

    @patch('pulearn.losses.caffe_losses.SoftmaxWithLossLayer.__init__',
           lambda *_: None)
    @patch('pulearn.losses.caffe_losses.WeightedSoftmaxWithLossLayer.__init__',
           lambda *_: None)
    def setUp(self):
        self.swll = SoftmaxWithLossLayer()
        self.wswll = WeightedSoftmaxWithLossLayer()
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

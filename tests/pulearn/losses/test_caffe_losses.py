import unittest
import numpy as np
from mock import MagicMock, patch
from pulearn.losses.caffe_losses import WeightedSoftmaxWithLossLayer


class TestSoftmaxWithLossLayer(unittest.TestCase):

    @patch('pulearn.losses.caffe_losses.SoftmaxWithLossLayer.__init__',
           lambda *_: None)
    @patch('pulearn.losses.caffe_losses.WeightedSoftmaxWithLossLayer.__init__',
           lambda *_: None)
    def setUp(self):
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
        self.wswll.param_str = str(pyloss_params)
        self.wswll.setup(self.bottom, self.top)

    def test_to_sample_weight(self):
        # test _to_class_weight
        assert np.all(self.wswll.class_weight == [1, 2, 2])
        # test _to_sample_weight
        labels = self.bottom[1]
        assert np.all(
            self.wswll._to_sample_weight(labels) ==
            np.reshape([1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1], [1, 3, 2, 2]))

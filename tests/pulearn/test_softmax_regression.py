import unittest
import numpy as np
from pulearn import SoftmaxRegression


class TestSoftmaxRegression(unittest.TestCase):

    def setUp(self):
        self.classifier = SoftmaxRegression(n_classes=2)
        np.random.seed(42)
        self.X = np.random.random((200, 2))
        self.y = np.random.randint(2, size=200)
        np.random.seed(None)

    def tearDown(self):
        self.classifier = SoftmaxRegression(n_classes=2)
        np.random.seed(None)

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
        assert np.allclose(class_weight, [1., 2 / 3.])

        self.classifier.class_weight = {0: 0.5, 1: 1}
        class_weight = self.classifier._to_class_weight(y_enc)
        assert np.allclose(class_weight, [0.5, 1.])

        # To avoid iterative multiplication
        class_weight = self.classifier._to_class_weight(y_enc)
        assert np.allclose(class_weight, [0.5, 1.])

    def test_sample_weight(self):
        y_enc = np.asarray(
            [[0, 1],
             [1, 0],
             [0, 1]]
        )
        self.classifier.class_weight = "balanced"
        sample_weight = self.classifier._to_sample_weight(y_enc, tiled=True)
        assert np.allclose(sample_weight,
                           np.asarray([
                               [2 / 3., 2 / 3.],
                               [1., 1.],
                               [2 / 3., 2 / 3.]]))

        sample_weight = self.classifier._to_sample_weight(y_enc, tiled=False)
        assert np.allclose(sample_weight, [2 / 3., 1., 2 / 3.])

    def test_yield_minibatches_idx(self):
        for batch in self.classifier._yield_minibatches_idx(5,
                                                            np.arange(13)):
            assert len(batch) == 5 or len(batch == 3)

    def test_one_hot(self):
        y = [1, 0, 1, 2]
        y_enc = self.classifier._one_hot(y, 3, dtype="float")
        assert np.allclose(y_enc,
                           np.asarray([
                               [0., 1., 0.],
                               [1., 0., 0.],
                               [0., 1., 0.],
                               [0., 0., 1.]]))

    def test_loss(self):
        prob = np.asarray([[0.6, 0.4], [0.1, 0.9]])
        y_enc = np.asarray([[1, 0], [1, 0]])
        loss = self.classifier._loss(prob, y_enc)
        print(loss)
        assert np.allclose(loss, [0.51082562, 2.30258509])

    def test_diff(self):
        prob = np.asarray([[0.6, 0.2, 0.2], [0.1, 0.1, 0.8]])
        y_enc = np.asarray([[1, 0, 0], [1, 0, 0]])
        diff = self.classifier._diff(prob, y_enc)
        print(diff)
        assert np.all(diff[:, 0] < 0)

    def test_fit(self):
        self.classifier.fit(self.X, self.y)

    def test_predict(self):
        self.classifier.fit(self.X, self.y)
        y_pred = self.classifier.predict(self.X)
        assert y_pred.shape == (200,)

    def test_predict_proba(self):
        self.classifier.fit(self.X, self.y)
        prob = self.classifier.predict_proba(self.X)
        assert prob.shape == (200, 2)

    def test_calc_loss(self):
        self.classifier.fit(self.X, self.y)
        loss = self.classifier.calc_loss(self.X, self.y)
        assert loss.shape == (200,)

    def test_calc_gradient(self):
        self.classifier.fit(self.X, self.y)
        grad = self.classifier.calc_gradient(self.X, self.y)
        assert grad.shape == (200, 2)

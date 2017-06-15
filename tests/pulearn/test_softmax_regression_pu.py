import numpy as np
from pulearn import (
    WeightedUnlabelledSoftmaxRegression,
    HardBootstrappingSoftmaxRegression
)


def test_weighted_unlabelled():
    y_enc = np.random.randint(2, size=(10, 2))

    wusr = WeightedUnlabelledSoftmaxRegression(2)
    class_weight = wusr._to_class_weight(y_enc)
    assert np.allclose(class_weight, np.ones(2))

    wusr = WeightedUnlabelledSoftmaxRegression(
        2, unlabelled_weight={0: 0.5, 1: 1})
    class_weight = wusr._to_class_weight(y_enc)
    assert np.allclose(class_weight, np.asarray([0.5, 1]))

    class_weight = wusr._to_class_weight(y_enc)
    assert np.allclose(class_weight, np.asarray([0.5, 1]))


def test_hard_bootstrapping():
    hbsr = HardBootstrappingSoftmaxRegression(
        2, unlabelled_weight={0: 0.5, 1: 1})

    y_enc = np.asarray([[1, 0], [0, 1]])
    assert np.allclose(hbsr._to_unlabelled_sample_weight(y_enc),
                       [[0.5,  0.5], [1.,   1.]])

    prob = np.asarray([[0.6, 0.4], [0.1, 0.9]])
    y_enc = np.asarray([[1, 0], [1, 0]])
    loss = hbsr._loss(prob, y_enc)
    print(loss)
    assert np.allclose(loss, [0.51082562,  1.2039728])

    diff = hbsr._diff(prob, y_enc)
    print(diff)
    assert np.all(diff[:, 0] < 0)

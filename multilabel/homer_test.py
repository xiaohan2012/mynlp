import numpy as np

from scipy.sparse import csr_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.datasets import make_multilabel_classification as make_ml_clf

from nose.tools import assert_equal

from .homer import (_balanced_kmeans, _construct_hierarchy_recursively,
                    _rec_flatten, HOMER)


def test_balanced_kmeans_simple():
    Y = np.asarray([[1, 1, 0, 0],
                    [1, 1, 0, 0],
                    [0, 0, 1, 1],
                    [0, 0, 1, 1]],
                   dtype=np.float32)

    actual = _balanced_kmeans(Y, k=2,
                              centroids=[0, 2])
    expected = [set([0, 1]), set([2, 3])]

    assert_equal(actual, expected)

    # random centroid initialization
    # the result is random
    # not sure if the test will
    # pass on a different computer
    actual = _balanced_kmeans(Y, k=2, random_state=123456)
    expected = [set([2, 3]), set([0, 1])]

    assert_equal(actual, expected)

    # sparse matrix test
    Y = csr_matrix(Y)
    actual = _balanced_kmeans(Y, k=2, random_state=123456)
    assert_equal(actual, expected)


def test_balanced_kmeans_sparse():
    Y = np.asarray([[1, 1, 0, 0],
                    [1, 1, 0, 0],
                    [0, 0, 1, 1],
                    [0, 0, 1, 1]],
                   dtype=np.float32)
    Y = csr_matrix(Y)
    actual = _balanced_kmeans(Y, k=2,
                              centroids=[0, 2])
    expected = [set([0, 1]), set([2, 3])]

    assert_equal(actual, expected)

    actual = _balanced_kmeans(Y, k=2, random_state=123456)
    expected = [set([2, 3]), set([0, 1])]

    assert_equal(actual, expected)


def test_balanced_kmeans_with_overflow():
    '''
    there is one 'extra' label
    '''
    Y = np.asarray([[1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1, 1]],
                   dtype=np.float32)

    actual = _balanced_kmeans(Y, k=2,
                              centroids=[0, 4])
    expected = [set([0, 1, 2]), set([4, 5, 3])]

    assert_equal(actual, expected)


def test_balanced_kmeans_with_empty_cluster():
    '''
    emepty cluster(cluster with one element) will be created
    '''
    Y = np.asarray([[1, 0, 100],
                    [1, 0, 100],
                    [0, 1, 100],
                    [0, 1, 100]],
                   dtype=np.float32)

    actual = _balanced_kmeans(Y, k=2,
                              centroids=[0, 2])
    expected = [set([0, 1]), set([2])]

    assert_equal(actual, expected)


def test_construct_hierarchy_recursively():
    Y = np.asarray([[1, 1.1, 0.0, 0.0, 1.0],
                    [1, 1.1, 0.0, 0.0, 1.0],
                    [0, 0.0, 1.0, 1.1, 0.5],
                    [0, 0.0, 1.0, 1.1, 0.5]],
                   dtype=np.float32)

    actual = _construct_hierarchy_recursively(
        Y,
        labels=['a', 'A', 'b', 'B', 'c'],
        k=2,
        random_state=12345)

    expected = [[['c'], ['a', 'A']], ['b', 'B']]

    assert_equal(actual, expected)

    # sparse test
    actual = _construct_hierarchy_recursively(
        Y,
        labels=['a', 'A', 'b', 'B', 'c'],
        k=2,
        random_state=12345)

    assert_equal(actual, expected)


def test_rec_flatten():
    actual = _rec_flatten([[['c'], ['a', 'A']], ['b', 'B']])
    expected = ['c', 'a', 'A', 'b', 'B']

    assert_equal(actual, expected)


def test_homer_shallowly():
    '''a very shallow test for HOMER'''
    n_classes = 4
    X, y = make_ml_clf(n_samples=50, n_features=20,
                       n_classes=n_classes,
                       n_labels=2,
                       length=10, allow_unlabeled=False,
                       return_indicator=True,
                       random_state=123456)

    model = HOMER(base_clf=OneVsRestClassifier(LinearSVC(random_state=0)),
                  k=3,
                  max_iter=20,
                  random_state=123456)

    model.fit(X, y)

    assert_equal(model._label_n, n_classes)
    assert_equal(type(model._estimator_hierarchy[0]),
                 OneVsRestClassifier)
    for clf in model._estimator_hierarchy[1]:
        assert_equal(type(clf[0]), OneVsRestClassifier)
        assert_equal(len(clf[1]), 0)  # no more children clf

    model.predict(X)
    assert_equal(model._meta_y_hier[0].shape, (50, 2))
    for y_hier in model._meta_y_hier[1]:
        assert_equal(y_hier.shape, (50, 2))

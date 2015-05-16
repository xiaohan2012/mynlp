import numpy as np
from nose.tools import assert_equal
from .homer import (balanced_kmeans, construct_hierarchy_recursively)

def test_balanced_kmeans_simple():
    Y = np.asarray([[1, 1, 0, 0],
                    [1, 1, 0, 0],
                    [0, 0, 1, 1],
                    [0, 0, 1, 1]],
                   dtype = np.float32)
    
    actual = balanced_kmeans(Y, k=2,
                             centroids=[0, 2])
    expected = [set([0, 1]), set([2, 3])]

    assert_equal(actual, expected)

    # random centroid initialization
    # the result is random
    # not sure if the test will
    # pass on a different computer
    actual = balanced_kmeans(Y, k=2, random_state=123456)
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
                   dtype = np.float32)
    
    actual = balanced_kmeans(Y, k=2,
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
                   dtype = np.float32)
    
    actual = balanced_kmeans(Y, k=2,
                             centroids=[0, 2])
    expected = [set([0, 1]), set([2])]

    assert_equal(actual, expected)


def test_construct_hierarchy_recursively():
    Y = np.asarray([[1, 1.1, 0.0, 0.0, 1.0],
                    [1, 1.1, 0.0, 0.0, 1.0],
                    [0, 0.0, 1.0, 1.1, 0.5],
                    [0, 0.0, 1.0, 1.1, 0.5]],
                   dtype=np.float32)
    
    actual = construct_hierarchy_recursively(
        Y,
        labels=['a', 'A', 'b', 'B', 'c'],
        k=2, 
        random_state=12345)

    expected = [[['c'], ['a', 'A']], ['b', 'B']]
    
    assert_equal(actual, expected)

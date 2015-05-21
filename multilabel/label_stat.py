# Label space statistics
# Reference:
# - Min-Ling Zhang and Zhi-Hua Zhou, A Review on Multi-Label Learning Algorithms

import numpy as np
from scipy import sparse


def _to_dense_array(y):
    if sparse.issparse(y):
        y = y.todense()

    y = np.array(y)
    return y


def label_cardinality(y):
    """
    measure the degree of multi-labeledness

    >>> y = np.array([[0,0,1], [1,0,0], [1,1,0]])
    >>> label_cardinality(y) # doctest: +ELLIPSIS
    1.333...
    >>> label_cardinality(sparse.csr_matrix(y)) # doctest: +ELLIPSIS
    1.333...
    """
    return np.mean(np.sum(_to_dense_array(y), axis=1))


def label_density(y):
    """
    Label density normalizes label cardinality by the
    number of possible labels in the label space

    >>> y = np.array([[0,0,1], [1,0,0], [1,1,0]])
    >>> label_density(y) # doctest: +ELLIPSIS
    0.4444...
    >>> label_density(sparse.csr_matrix(y)) # doctest: +ELLIPSIS
    0.4444...
    """
    return label_cardinality(y) / y.shape[1]


def label_diversity(y):
    """
    The number of unique label assignment

    >>> y = np.array([[0,0,1], [1,0,0], [1,1,0], [0,0,1]])
    >>> label_diversity(y)
    3
    """
    values = set()
    y = _to_dense_array(y)

    for r in y:
        values.add(tuple(r.tolist()))

    return len(values)

def normalized_label_diversity(y):
    """
    The number of unique label assignment divided by example number

    >>> y = np.array([[0,0,1], [1,0,0], [1,1,0], [0,0,1]])
    >>> normalized_label_diversity(y)
    0.75
    """
    return label_diversity(y) / float(y.shape[0])

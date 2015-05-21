import codecs
import cPickle as pkl
import numpy as np

from scipy.sparse import csr_matrix
from arff import loads


def load_sparse_arff(path, label_n):
    rows = []
    labels = []
    for i, r in enumerate(loads(codecs.open(path, 'r', 'utf8').read())):
        print i
        m = len(r._values)
        rows.append(r._values[:m-label_n])
        labels.append(r._values[m-label_n:])

    # convert to sparse matrix
    row_n = len(rows)
    # X = csr_matrix((len(rows), m-label_n), dtype=np.bool_)
    X = np.zeros((len(rows), m-label_n), dtype=np.bool_)
    for i, r in enumerate(rows):
        print "%d / %d" % (i, row_n)
        for j, v in enumerate(r):
            if v != None:
                X[i, j] = int(v)

    # y = csr_matrix((len(rows), label_n), dtype=np.bool_)
    y = np.zeros((len(rows), label_n), dtype=np.bool_)

    for i, r in enumerate(labels):
        print "%d / %d" % (i, row_n)
        for j, v in enumerate(r):
            if v != None:
                y[i, j] = int(v)

    return csr_matrix(X), csr_matrix(y)

dataset_type = "test"
X, y = load_sparse_arff('data/delicious-%s.arff' % dataset_type, label_n=983)

pkl.dump(X, open('data/del_X_%s.pkl' % dataset_type, 'w'))
pkl.dump(y, open('data/del_y_%s.pkl' % dataset_type, 'w'))

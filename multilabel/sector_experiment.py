import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from homer import HOMER

X_path = '/cs/puls/Experiments/hxiao-test/feature-data.mat'
# X_path = '/cs/puls/Experiments/hxiao-test/original-esmerk-sectors-PC50_PROJ.mat'
Y_path = '/cs/puls/Experiments/hxiao-test/label-data.mat'
label_path = '/cs/puls/Experiments/hxiao-test/sector-labels'

RANDOM_PROJECTION_FLAG = True

X = loadmat(X_path)['featureData']
# X = loadmat(X_path)['projection']

if RANDOM_PROJECTION_FLAG:
    from sklearn.random_projnection import SparseRandomProjection

    print "Applying random projection to reduce dimension"
    print "Shape before: %r" % (X.shape, )

    transformer = SparseRandomProjection(random_state=0)
    X = transformer.fit_transform(X)
    print "Shape after: %r" % (X.shape, )
else:
    print "Random projection: OFF"

y = loadmat(Y_path)['labelData']
with open(label_path, 'r') as f:
    label_names = map(lambda l: l.strip(), f.readlines())

# import pdb
# pdb.set_trace()

model = HOMER(base_clf=OneVsRestClassifier(LinearSVC(random_state=0),
                                           n_jobs=3),
              k=3,
              max_iter=20,
              random_state=123456,
              verbose=True)

# sample subset of all the data
rng = np.random.RandomState(0)
# SAMPLE_N = None
SAMPLE_N = 10000
if SAMPLE_N:
    print "Sample size: %d" % SAMPLE_N
    rows = rng.permutation(X.shape[0])[:SAMPLE_N]
    X = X[rows, :]
    y = y[rows, :]
else:
    print "Sample size: all data"
    SAMPLE_N = X.shape[0]

# sample train and test
train_ratio = 0.9
train_n = int(SAMPLE_N*train_ratio)

rows = rng.permutation(SAMPLE_N)
train_rows = rows[: train_n]
test_rows = rows[train_n:]

train_X = X[train_rows, :]
train_y = y[train_rows, :]
test_X = X[test_rows, :]
test_y = y[test_rows, :]

from exp_util import run_experiment

run_experiment(model, train_X, train_y, test_X, test_y, label_names)

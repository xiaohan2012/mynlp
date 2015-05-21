import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

# from sklearn.datasets import make_multilabel_classification
# from sklearn.preprocessing import LabelBinarizer

X_path = '/cs/puls/Experiments/hxiao-test/feature-data.mat'
Y_path = '/cs/puls/Experiments/hxiao-test/label-data.mat'

X = loadmat(X_path)['featureData']
y = loadmat(Y_path)['labelData']

RANDOM_PROJECTION_FLAG = True

if RANDOM_PROJECTION_FLAG:
    from sklearn.random_projection import SparseRandomProjection

    print "Applying random projection to reduce dimension"
    print "Shape before: %r" % (X.shape, )

    transformer = SparseRandomProjection()
    X = transformer.fit_transform(X)
    print "Shape after: %r" % (X.shape, )


# sample subset of all the data
rng = np.random.RandomState(0)
sample_n = 10000
rows = rng.permutation(X.shape[0])[:sample_n]
X = X[rows, :]
y = y[rows, :]

# sample train and test
train_ratio = 0.8
train_n = int(sample_n*train_ratio)

rows = rng.permutation(sample_n)
train_rows = rows[: train_n]
test_rows = rows[train_n:]

train_X = X[train_rows, :]
train_y = y[train_rows, :]
test_X = X[test_rows, :]
test_y = y[test_rows, :]


model = OneVsRestClassifier(LinearSVC(random_state=0), n_jobs=-1)

from exp_util import run_experiment

run_experiment(model, train_X, train_y, test_X, test_y)

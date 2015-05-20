import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, jaccard_similarity_score,
                             hamming_loss, precision_recall_fscore_support)

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

import timeit
start = timeit.default_timer()

model.fit(train_X, train_y)

end = timeit.default_timer()

print "Training takes %.2f secs" % (end - start)

pred_y = model.predict(test_X)

print "Subset accuracy: %.2f\n" % (accuracy_score(test_y, pred_y)*100)

print "Hamming loss: %.2f\n" % (hamming_loss(test_y, pred_y))

print "Accuracy(Jaccard): %.2f\n" % (jaccard_similarity_score(test_y, pred_y))

p_ex, r_ex, f_ex, _ = precision_recall_fscore_support(test_y, pred_y,
                                                      average="samples")

print "Precision/Recall/F1(example) : %.2f  %.2f  %.2f\n" \
    % (p_ex, r_ex, f_ex)

p_mic, r_mic, f_mic, _ = precision_recall_fscore_support(test_y, pred_y,
                                                         average="micro")

print "Precision/Recall/F1(micro) : %.2f  %.2f  %.2f\n" \
    % (p_mic, r_mic, f_mic)

p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(test_y, pred_y,
                                                         average="macro")

print "Precision/Recall/F1(macro) : %.2f  %.2f  %.2f\n" \
    % (p_mac, r_mac, f_mac)

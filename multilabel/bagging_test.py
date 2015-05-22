import numpy as np

from cPickle import load
from scipy.io import loadmat
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_multilabel_classification
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.metrics import (accuracy_score, jaccard_similarity_score,
                             hamming_loss, precision_recall_fscore_support)

from multilabel_bagging import MultilabelBaggingClassifier

# X_path = '/cs/puls/Experiments/hxiao-test/feature-data.mat'
# Y_path = '/cs/puls/Experiments/hxiao-test/label-data.mat'

# X = loadmat(X_path)['featureData']
# # X = loadmat(X_path)['projection']

# RANDOM_PROJECTION_FLAG = True
# if RANDOM_PROJECTION_FLAG:
#     from sklearn.random_projection import SparseRandomProjection

#     print "Applying random projection to reduce dimension"
#     print "Shape before: %r" % (X.shape, )

#     transformer = SparseRandomProjection(random_state=0)
#     X = transformer.fit_transform(X)
#     print "Shape after: %r" % (X.shape, )
# else:
#     print "Random projection: OFF"

# y = loadmat(Y_path)['labelData']

# rng = np.random.RandomState(0)
# # SAMPLE_N = None
# SAMPLE_N = 10000
# if SAMPLE_N:
#     print "Sample size: %d" % SAMPLE_N
#     rows = rng.permutation(X.shape[0])[:SAMPLE_N]
#     X = X[rows, :]
#     y = y[rows, :]
# else:
#     print "Sample size: all data"
#     SAMPLE_N = X.shape[0]

# X, y = make_multilabel_classification(n_samples=10000,
#                                       n_features=500,
#                                       n_classes=900,
#                                       n_labels=20,
#                                       return_indicator=True,
#                                       allow_unlabeled=False,
#                                       random_state=123)

# train_X, test_X, train_y, test_y = train_test_split(X, y,
#                                                     test_size=0.2,
#                                                     random_state=123)


train_X = load(open('data/del_X_train.pkl')).todense()
train_y = load(open('data/del_y_train.pkl')).todense()
test_X = load(open('data/del_X_test.pkl')).todense()
test_y = load(open('data/del_y_test.pkl')).todense()


model = MultilabelBaggingClassifier(base_estimator=OneVsRestClassifier(GaussianNB(),
                                                                       n_jobs=-1),
                                    n_estimators=20,
                                    n_jobs=-1,
                                    verbose=2)

model.fit(train_X, train_y)

print "Making predictions..."

pred_y = model.predict(test_X)


def print_summary(test_y, pred_y):
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


print_summary(test_y, pred_y)

# print "single model case..."

# model2 = OneVsRestClassifier(GaussianNB(),
#                              n_jobs=-1)

# model2.fit(train_X, train_y)

# print_summary(test_y, model2.predict(test_X))



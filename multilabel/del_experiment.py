import numpy as np

from cPickle import load

from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import LinearSVC

from homer import HOMER
from label_stat import label_summary

train_X = load(open('data/del_X_train.pkl')).todense()
train_y = load(open('data/del_y_train.pkl')).todense()
test_X = load(open('data/del_X_test.pkl')).todense()
test_y = load(open('data/del_y_test.pkl')).todense()

train_X, train_y, test_X, test_y = map(np.array, (train_X, train_y,
                                                  test_X, test_y))

train_X, train_y, test_X, test_y = map(lambda d: np.asarray(d, dtype=np.int8),
                                       (train_X, train_y, test_X, test_y))

# binary_model = LinearSVC(random_state=0)
# binary_model = GaussianNB()
binary_model = BernoulliNB()

# model = HOMER(base_clf=OneVsRestClassifier(binary_model, n_jobs=3),
#               k=3,
#               max_iter=20,
#               # random_state=123456,
#               # verbose=True,
#               verbose=False)

model = OneVsRestClassifier(binary_model, n_jobs=-1)

print "Using model: ", model

print "%d samples, %d features of training set:" % train_X.shape
print label_summary(train_y)

from exp_util import run_experiment

run_experiment(model, train_X, train_y, test_X, test_y)

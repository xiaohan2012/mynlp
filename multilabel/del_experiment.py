import numpy as np

from cPickle import load

from sklearn.multiclass import OneVsRestClassifier
from cc import ClassifierChain
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import LinearSVC

from homer import HOMER
from label_stat import label_summary
from util import (load_toy_data, load_delicious_data, load_sector_data)
from exp_util import run_experiment

# train_X, test_X, train_y, test_y = load_toy_data()
# train_X, test_X, train_y, test_y = load_delicious_data()
train_X, test_X, train_y, test_y = load_sector_data()

# binary_model = LinearSVC(random_state=0)
binary_model = GaussianNB()
# binary_model = BernoulliNB()

# model = HOMER(base_clf=OneVsRestClassifier(binary_model, n_jobs=3),
#               k=3,
#               max_iter=20,
#               # random_state=123456,
#               # verbose=True,
#               verbose=False)

# model = OneVsRestClassifier(binary_model, n_jobs=-1)

models = [ClassifierChain(binary_model, n_jobs=-1, verbose=2),
          OneVsRestClassifier(binary_model, n_jobs=-1)
]

print "%d samples, %d features of training set:" % train_X.shape
print label_summary(train_y)

for model in models:
    print "Using model: ", model
    print "#" * 20
    run_experiment(model, train_X, train_y, test_X, test_y)
    print 


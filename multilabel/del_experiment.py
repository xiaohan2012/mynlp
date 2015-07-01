import numpy as np

from cPickle import load

from sklearn.multiclass import OneVsRestClassifier
from cc import ClassifierChain
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from homer import HOMER
from label_stat import label_summary
from util import (load_toy_data, load_delicious_data, load_sector_data)
from exp_util import run_experiment

# train_X, test_X, train_y, test_y = load_toy_data()
# train_X, test_X, train_y, test_y = load_delicious_data()


train_X, test_X, train_y, test_y = load_sector_data(random_state=0)

LDA_FLAG = True
# LDA_FLAG = False

if LDA_FLAG:
    print "LDA: ON"
    train_X_lda = load(open('data/train_X_lda_ntopic_500_X_rng_0.pkl'))
    test_X_lda = load(open('data/test_X_lda_ntopic_500_X_rng_0.pkl'))

    train_X = np.concatenate([train_X, train_X_lda], axis=1)
    test_X = np.concatenate([test_X, test_X_lda], axis=1)
    
    import pdb
    pdb.set_trace()
else:
    print "LDA: OFF"

# binary_model = LinearSVC(random_state=0)
# binary_model = GaussianNB()
# binary_model = BernoulliNB()
binary_model = LogisticRegression()

# model = HOMER(base_clf=OneVsRestClassifier(binary_model, n_jobs=3),
#               k=3,
#               max_iter=20,
#               # random_state=123456,
#               # verbose=True,
#               verbose=False)

# model = OneVsRestClassifier(binary_model, n_jobs=-1)

models = [# ClassifierChain(binary_model, n_jobs=5, verbose=2),
          OneVsRestClassifier(binary_model, n_jobs=-1)
]



print "%d samples, %d features of training set:" % train_X.shape
print label_summary(train_y)

for model in models:
    print "Using model: ", model
    print "#" * 20
    run_experiment(model, train_X, train_y, test_X, test_y)
    print 


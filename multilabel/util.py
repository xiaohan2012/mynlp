import numpy as np
from cPickle import load, dump
from scipy.io import loadmat

from sklearn.datasets import make_multilabel_classification
from sklearn.random_projection import SparseRandomProjection

from sklearn.cross_validation import train_test_split


def load_delicious_data():
    train_X = load(open('data/del_X_train.pkl')).todense()
    train_y = load(open('data/del_y_train.pkl')).todense()
    test_X = load(open('data/del_X_test.pkl')).todense()
    test_y = load(open('data/del_y_test.pkl')).todense()

    train_X, train_y, test_X, test_y = map(lambda d: np.asarray(d,
                                                                dtype=np.int8),
                                           (train_X, train_y, test_X, test_y))

    return train_X, test_X, train_y, test_y


def load_toy_data(random_state=0):
    X, y = make_multilabel_classification(return_indicator=True,
                                          allow_unlabeled=False,
                                          random_state=random_state)

    return train_test_split(X, y,
                            test_size=0.2,
                            random_state=random_state)


def load_sector_data(random_state=0):
    X_path = 'data/sector_X.pkl'
    Y_path = 'data/sector_Y.pkl'
    X = load(open(X_path))
    Y = load(open(Y_path))
    return map(lambda d: np.array(d.todense()),  # convert to np.array
               train_test_split(X, Y, test_size=0.1,
                                random_state=random_state))

def create_sector_subset(sample_n, X_output_path, Y_output_path):
    X_path ='/cs/puls/Experiments/hxiao-test/feature-data.mat'
    Y_path = '/cs/puls/Experiments/hxiao-test/label-data.mat'

    X = loadmat(X_path)['featureData']
    Y = loadmat(Y_path)['labelData']

    print "Applying random projection to reduce dimension"
    print "Shape before: %r" % (X.shape, )

    transformer = SparseRandomProjection(random_state=0)
    X = transformer.fit_transform(X)
    print "Shape after: %r" % (X.shape, )
    print "Random projection: OFF"

    rng = np.random.RandomState(0)
    print "Sample size: %d" % sample_n
    rows = rng.permutation(X.shape[0])[:sample_n]
    X = X[rows, :] 
    Y = Y[rows, :]

    dump(X, open(X_output_path, 'w'))
    dump(Y, open(Y_output_path, 'w'))


def convert_to_gensim_corpus(X):
    """convert doc2term matrix to corpus in Gensim format """
    print "Preparing corpus..."
    corpus = []
    print_every = X.shape[0] / 10
    for i in xrange(X.shape[0]):
        if (i+1)% print_every == 0:
            print("{} / {}".format(i+1, X.shape[0]))
        word_ids = X[i,:].nonzero()[1]
        word_freqs = X[i,:].data
        if len(word_ids) == 0:
            raise ValueError("no words")
        corpus.append(zip(word_ids, word_freqs))
    return corpus

if __name__ == "__main__":
    create_sector_subset(10000, 'data/sector_X.pkl', 'data/sector_Y.pkl')

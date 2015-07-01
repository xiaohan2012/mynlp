
import numpy as np
from gensim.models.ldamulticore import LdaMulticore
from sklearn.cross_validation import train_test_split

from util import convert_to_gensim_corpus

import logging
logging.basicConfig(level=logging.WARNING)

def get_lda_model(X, n_topics):
    """
    do LDA on the doc2term matrix and return the model and corpus in Gensim format 
    """
    corpus = convert_to_gensim_corpus(X)
    print "Performing LDA..."
    model = LdaMulticore(corpus, num_topics=n_topics)
    return model, corpus

def infer_lda_topics(model, corpus):
    """Infer topics on a set of documents"""
    res = np.zeros((len(corpus), model.num_topics))
    print_every = len(corpus) / 10
    for i, doc in enumerate(corpus):
        if (i+1)%print_every == 0:
            print("{} / {}".format(i+1, len(corpus)))
        for topic_id, proba in model[doc]:
            res[i,topic_id] = proba
    return res


if __name__ == "__main__":
    N_TOPICS = 100
    SAMPLE_N = 10000
    RANDOM_STATE = 0
    
    from scipy.io import loadmat
    import cPickle as pkl
    X_path = '/cs/puls/Experiments/hxiao-test/feature-data.mat'
    Y_path = '/cs/puls/Experiments/hxiao-test/label-data.mat'
        
    X = loadmat(X_path)['featureData']
    Y = loadmat(Y_path)['labelData']

    rng = np.random.RandomState(RANDOM_STATE)
    rows = rng.permutation(X.shape[0])[:SAMPLE_N]

    X = X[rows, :]
    Y = Y[rows, :]
    
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.1,
                                                        random_state=RANDOM_STATE)
    model, train_corpus = get_lda_model(train_X, N_TOPICS)

    train_lda_repr = infer_lda_topics(model, train_corpus)
    test_lda_repr = infer_lda_topics(model, convert_to_gensim_corpus(test_X))
    
    pkl.dump(train_lda_repr, open('data/train_X_lda_rng_0.pkl', 'w'))
    pkl.dump(test_lda_repr, open('data/test_X_lda_rng_0.pkl', 'w'))

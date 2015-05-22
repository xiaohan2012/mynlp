import numpy as np
import itertools
from sklearn.base import clone
from sklearn.externals.joblib import Parallel, delayed
from sklearn.ensemble.base import _partition_estimators


def _parallel_build_estimator(n_estimators, estimator, X, Y, start, verbose):
    clfs = []
    for i in range(n_estimators):
        if verbose > 1:
            print("building estimator %d of %d" % (i + 1, n_estimators))

        clf = clone(estimator._estimator)

        Xi = np.hstack([X, Y[:, :(start+i)]])
        Yi = Y[:, start+i]
        assert Xi.shape[1] > 0
        assert Yi.shape[0] > 0

        clfs.append(clf.fit(Xi, Yi))

    return clfs


class ClassifierChain(object):
    def __init__(self, base_estimator, n_jobs=1, verbose=0):
        self._estimator = base_estimator
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, Y):
        X, Y = map(np.atleast_2d, (X, Y))
        assert X.shape[0] == Y.shape[0]
        Ny = Y.shape[1]

        self.estimators_ = []
        n_jobs, n_estimators, starts = _partition_estimators(Ny, self.n_jobs)

        results = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_build_estimator)(
                n_estimators[i],
                self,
                X,
                Y,
                starts[i],
                verbose=self.verbose)
            for i in range(n_jobs))

        self.estimators_ += list(itertools.chain.from_iterable(results))

        return self

    def predict(self, X):
        Y = np.empty([X.shape[0], len(self.estimators_)])
        for i, clf in enumerate(self.estimators_):
            Y[:, i] = clf.predict(np.hstack([X, Y[:, :i]]))
        return Y

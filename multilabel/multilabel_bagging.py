import numbers
import numpy as np
import itertools
from sklearn.ensemble.bagging import (_parallel_build_estimators,
                                      _partition_estimators)
from sklearn.ensemble import BaggingClassifier
from sklearn.utils import check_X_y, check_random_state

from sklearn.externals.joblib import Parallel, delayed

# from pyspark import SparkContext

# sc = SparkContext("spark://ukko027:50511", "Mutlilabel bagging")


MAX_INT = np.iinfo(np.int32).max

class MultilabelBaggingClassifier(BaggingClassifier):

    def _validate_y(self, y):
        self.classes_ = np.arange(y.shape[1])
        self.n_classes_ = len(self.classes_)
        return y

    def fit(self, X, y, sample_weight=None):
        """Build a Bagging ensemble of estimators from the training
           set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        y : array-like, shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if the base estimator supports
            sample weighting.

        Returns
        -------
        self : object
            Returns self.
        """
        ## TODO:
        # check if the classifier in OneVSRestClassiifer has `predict_proba``
        # attribute
        random_state = check_random_state(self.random_state)

        # Convert data
        X, y = check_X_y(X, y, ['csr', 'csc', 'coo'], multi_output=True)

        # Remap output
        n_samples, self.n_features_ = X.shape
        y = self._validate_y(y)

        # Check parameters
        self._validate_estimator()

        if isinstance(self.max_samples, (numbers.Integral, np.integer)):
            max_samples = self.max_samples
        else:  # float
            max_samples = int(self.max_samples * X.shape[0])

        if not (0 < max_samples <= X.shape[0]):
            raise ValueError("max_samples must be in (0, n_samples]")

        if isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            max_features = int(self.max_features * self.n_features_)

        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features must be in (0, n_features]")

        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available"
                            " if bootstrap=True")

        # Free allocated memory, if any
        self.estimators_ = None

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(self.n_estimators,
                                                             self.n_jobs)
        seeds = random_state.randint(MAX_INT, size=self.n_estimators)
        
        all_results = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_build_estimators)(
                n_estimators[i],
                self,
                X,
                y,
                sample_weight,
                seeds[starts[i]:starts[i + 1]],
                verbose=self.verbose)
            for i in range(n_jobs))

        # params = [(n_estimators[i],
        #            self,
        #            X,
        #            y,
        #            sample_weight,
        #            seeds[starts[i]:starts[i + 1]],
        #            self.verbose)
        #           for i in range(n_jobs)]
        
        # all_results = sc.parallelize(params).map(_parallel_build_estimators)

        # Reduce
        self.estimators_ = list(itertools.chain.from_iterable(
            t[0] for t in all_results))
        self.estimators_samples_ = list(itertools.chain.from_iterable(
            t[1] for t in all_results))
        self.estimators_features_ = list(itertools.chain.from_iterable(
            t[2] for t in all_results))

        if self.oob_score:
            self._set_oob_score(X, y)

        return self

    def predict(self, X):
        """Predict class for X.
        The predicted class of an input sample is computed as the class with
        the highest mean predicted probability. If base estimators do not
        implement a ``predict_proba`` method, then it resorts to voting.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        Returns
        -------
        y : array of shape = [n_samples, n_classes]
            The predicted multilabel classes.
        """
        predicted_probabilitiy = self.predict_proba(X)
        return np.asarray(predicted_probabilitiy > 0.5,
                          dtype=np.int64)
        # print predicted_probabilitiy
        # return self.classes_.take((np.argmax(predicted_probabilitiy, axis=1)),
        # axis=0)

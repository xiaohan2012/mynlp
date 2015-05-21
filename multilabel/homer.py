import numpy as np
import math
from collections import defaultdict
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.base import clone
from scipy import sparse


def _construct_hierarchy_recursively(Y, labels, k,
                                     metric='cosine',
                                     max_iter=20,
                                     random_state=None,
                                     verbose=False):
    if sparse.issparse(Y):
        Y = Y.todense()

    Y = np.array(Y)  # enforce it to np.array

    def aux(Y, labels, k):
        if Y.shape[1] <= k:
            return labels
        else:
            hierarchy = []

            label_partitions = _balanced_kmeans(Y, k, max_iter=max_iter,
                                                metric=metric,
                                                random_state=random_state)
            if verbose:
                print "Partitioned %r into:" % sorted(labels)
                for parti in label_partitions:
                    print "- %r" % (sorted([labels[l] for l in parti]))
                print '\n\n'
            for partition in label_partitions:
                # convert it to slice-compatible type
                partition = list(partition)
                subset_labels = [labels[ind] for ind in partition]
                
                # TODO: should we try to use the entire instance space?
                rows = np.nonzero(np.any(Y[:, partition], axis=1))[0]
                cols = np.array(partition)
                
                # subset_Y = Y[rows[:, np.newaxis], cols]
                subset_Y = Y[:, cols]  # use all the rows
                ans = aux(subset_Y,
                          subset_labels,
                          k)
                hierarchy.append(ans)
            return hierarchy

    return aux(Y, labels, k)


def _balanced_kmeans(Y, k,
                     metric='cosine',
                     centroids=None,
                     max_iter=20,
                     random_state=None):
    '''
    Y: columns of label features, each column corresponds to one label
    k: number of clusters to have
    
    Return:
    a partition of the label set
    
    '''
    if sparse.issparse(Y):
        Y = Y.todense()

    Y_t = np.transpose(Y)

    label_n = Y.shape[1]
    max_cluster_size = math.ceil(label_n / float(k))

    if not centroids:
        # sample k random centroids
        rng = np.random.RandomState(random_state)
        centroids = rng.permutation(label_n)[:k]

    centroid_vecs = Y_t[centroids, :]

    converged = False
    labels = np.arange(label_n)

    clusters = defaultdict(set)
    previous_clusters = None

    iter_i = 0
    while not converged:
        iter_i += 1
        if iter_i > max_iter:
            break

        # we previous_clusters is not None,
        # we assume we have converged
        converged = (previous_clusters != None)

        for l in labels:
            sorted_centroids = np.argsort(
                pairwise_distances(centroid_vecs, Y_t[l, :], metric=metric).flatten())

            for i in sorted_centroids.flatten():
                c = centroids[i]
                if len(clusters[c]) < max_cluster_size:
                    # find the closest centroid that still has capacity
                    clusters[c].add(l)
                        
                    if previous_clusters and \
                       l not in previous_clusters[c]:
                        converged = False
                    break

        # update centroid
        for i, cluster_labels in enumerate(clusters.values()):
            if len(cluster_labels) > 1:
                centroid_vecs[i, :] = np.mean(Y_t[list(cluster_labels), :], axis=0)
        
        previous_clusters = clusters
        clusters = defaultdict(set)
        
    return previous_clusters.values()


def _rec_flatten(rec_lst):
    '''Recursive flatten a nested list, rec_lst
    '''
    if not isinstance(rec_lst, list):
        return [rec_lst]
    else:
        ans = []
        for i in rec_lst:
            ans += _rec_flatten(i)
        return ans


class HOMER(object):
    def __init__(self, base_clf, k,
                 metric='cosine',
                 max_iter=20, random_state=None, verbose=False):
        '''
        base_clf: the underlying multilabel classifier
        '''
        self.base_clf = base_clf
        self.k = k
        self.random_state = random_state
        self.metric = metric
        self.max_iter = max_iter
        self.verbose = verbose
        
    def fit(self, X, y, label_names=None):
        self._feature_n = X.shape[1]
        
        # Dangerous operation but some justification:
        # - we are likely to operate on the dimension reduced matrix
        # - dense matrix support richer&fancier indexing

        if sparse.issparse(X):
            X = np.array(X.todense())  # to np.array

        if sparse.issparse(y):
            y = np.array(y.todense())

        # partition the label set
        # using contrained k-means
        label_n = y.shape[1]
        self._labels_names = (label_names if label_names else np.arange(label_n))
        self._labels_indices = np.arange(label_n)

        if self.verbose:
            print "Constructing label hierarchy.."

        label_hierarchy = _construct_hierarchy_recursively(
            y,
            labels=self._labels_indices,
            metric=self.metric,
            k=self.k,
            random_state=self.random_state,
            max_iter=self.max_iter)

        if self.verbose:
            print "Label Hierarchy"
            print _construct_hierarchy_recursively(
                y,
                labels= self._labels_names,
                metric=self.metric,
                k=self.k,
                random_state=self.random_state,
                max_iter=self.max_iter,
                verbose=True)

        self.fitted_clf_n = 0

        def aux(partitions):
            
            '''build the classifier recursivley according to the hierarchy'''
            if isinstance(partitions, list) and\
               len(partitions) > 1:  # internal node
                flattened_partitions = [_rec_flatten(p) for p in partitions]
                # import pdb
                # pdb.set_trace()
                y_bin = np.vstack([np.any(y[:, p], axis=1)  # dim 1
                                   for p in flattened_partitions])

                y_bin = np.transpose(y_bin)  # why transpose?see the above line
                row_inds = np.any(y_bin, axis=1)

                if not row_inds.any():
                    raise ValueError("Zero rows are selected for partition %r" % flattened_partitions)

                sub_X = X[row_inds, :]
                sub_y = y_bin[row_inds, :]

                clf = clone(self.base_clf).fit(sub_X, sub_y)
                
                if self.verbose:
                    self.fitted_clf_n += 1
                    print 'Finished fitting %d classifiers with %d examples' % \
                        (self.fitted_clf_n, sub_X.shape[0])
                    
                children_clfs = []
                for partition in partitions:
                    # we can dive deeper
                    if _rec_flatten(partition) != partition:
                        children_clfs.append(aux(partition))

                return (clf, children_clfs)
        
        self._label_n = label_n
        self._label_hierarchy = label_hierarchy
        self._estimator_hierarchy = aux(label_hierarchy)

    def predict(self, X):
        assert X.shape[1] == self._feature_n, "feature number does not equal " + \
            "to the trained one"

        def pred_meta_y(eh):
            '''
            eh: acronym for estimator_hierarchy
            
            Return:
            prediction for (meta)labels'''
            pred_y = eh[0].predict(X)

            child_ys = []
            for child_eg in eh[1]:
                child_ys.append(pred_meta_y(child_eg))

            return (pred_y, child_ys)

        ## convert back the meta-label encoding
        meta_y_hier = pred_meta_y(self._estimator_hierarchy)
        self._meta_y_hier = meta_y_hier  # for testing purpose

        pred_y = np.ones((X.shape[0], self._label_n))

        def decode_meta_y(pred_y, meta_y_hier, label_hier):
            '''make certain columns and rows zero'''
            meta_y = meta_y_hier[0]
            for part, col in \
                zip(label_hier, 
                    np.arange(meta_y.shape[1])):

                label_values = np.array(_rec_flatten(part))
                row_inds = np.nonzero(meta_y[:, col] == 0)[0]
                if row_inds.size:
                    pred_y[row_inds[:, np.newaxis],  # some broadcasting trick
                           label_values] = 0

            for child_label_hier, child_y_hier \
                in zip(label_hier, meta_y_hier[1]):

                decode_meta_y(pred_y, child_y_hier, child_label_hier)

            return pred_y

        pred_y = decode_meta_y(pred_y, meta_y_hier, self._label_hierarchy)

        return pred_y

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)

import numpy as np
import math
from collections import defaultdict
from sklearn.metrics.pairwise import pairwise_distances

def construct_hierarchy_recursively(Y, labels, k, 
                                    max_iter=20,
                                    rng=np.random.RandomState(123456)):
    def aux(Y, labels, k):

        if Y.shape[1] <= k:
            return labels
        else:
            hierarchy = []

            label_partitions = balanced_kmeans(Y, k, max_iter=max_iter,
                                               rng=rng)

            for partition in label_partitions:
                # convert it to slice-compatible type
                partition = list(partition)
                print partition
                subset_labels = [labels[ind] for ind in partition]
                subset_Y = Y[np.any(Y[:, partition], axis=1), :][:, partition]
                ans = aux(subset_Y,
                          subset_labels,
                          k)
                hierarchy.append(ans)
            return hierarchy
    
    return aux(Y, labels, k)
    
def balanced_kmeans(Y, k,
                    centroids=None,
                    max_iter=20,
                    rng=np.random.RandomState(123456)):
    '''
    Y: columns of label features, each column corresponds to one label
    k: number of clusters to have
    
    Return:
    a partition of the label set
    
    '''
    Y_t = np.transpose(Y)    

    label_n = Y.shape[1]
    max_cluster_size = math.ceil(label_n / float(k))

    if not centroids:
        # sample k random centroids
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
                pairwise_distances(centroid_vecs, Y_t[l, :]).flatten())

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


class HOMER(object):
    def __init__(self, br_clf, k):
        '''
        br_clf: the name of the binary relevance classifier
        '''
        self.br_clf = br_clf
        self.k = k

    def fit(self, X, Y):
        # partition the label set
        # using contrained k-means
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X, Y):
        self.fit(X, Y)
        return self.transform(X)

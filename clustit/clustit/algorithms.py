""" module containing the implementations of clustering algorithms used in clustit """

import utils
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

import sklearn.cluster

def hierarchical_clustering(edgelist=None, distance_matrix=None,
                            names=None, method='complete', threshold=None):
    """ create a flat clustering based on hierarchical clustering methods and a threshold """
    if edgelist is not None:
        distance_matrix, names = utils.edgelist_to_distance_matrix(edgelist)

    linkage = sch.linkage(distance_matrix, method=method)

    threshold = threshold or 0.7*linkage[:,2].max()
    labels = sch.fcluster(linkage, threshold, criterion='distance')

    return labels

def dbscan(edgelist=None, distance_matrix=None, threshold=None):
    """ cluster using DBSCAN algorithm """
    if edgelist is not None:
        distance_matrix, names = utils.edgelist_to_distance_matrix(edgelist)

    threshold = threshold or 1

    core_samples, labels = sklearn.cluster.dbscan(distance_matrix, metric='precomputed',
                                algorithm='brute', eps=threshold, min_samples=2)
    return labels




""" module containing the implementations of clustering algorithms used in clustit """

from __future__ import print_function
import sys

import clustit.utils as utils
import scipy.cluster.hierarchy as sch
from sklearn.cluster.hierarchical import _TREE_BUILDERS
import hdbscan
#from clustit.plot import plot_dendrogram
import scipy

import sklearn.cluster
from builtins import input


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

    threshold = threshold or 2.8

    core_samples, labels = sklearn.cluster.dbscan(distance_matrix, metric='precomputed',
                                algorithm='brute', eps=threshold, min_samples=3)
    return labels

def hierarchical_dbscan(edgelist=None, distance_matrix=None):
    """ cluster using the Hierarchical DBSCAN algorithm """
    if edgelist is not None:
        distance_matrix = utils.edgelist_to_distance_matrix(edgelist)
    hdbscan_clusterer = hdbscan.HDBSCAN(metric="precomputed", min_samples=2)
    labels = hdbscan_clusterer.fit_predict(distance_matrix)
    return labels

def spectral(edgelist=None, distance_matrix=None,n_clusters=10):
    """ cluster using spectral clustering """

    if edgelist is not None:
        distance_matrix, names = utils.edgelist_to_distance_matrix(edgelist)


    sc = sklearn.cluster.SpectralClustering(n_clusters, affinity='precomputed')
    labels = sc.fit_predict(distance_matrix)

    return labels

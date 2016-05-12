""" module containing the implementations of clustering algorithms used in clustit """

import utils
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import _cpy_linkage_methods
import numpy

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

    threshold = threshold or 2.8

    core_samples, labels = sklearn.cluster.dbscan(distance_matrix, metric='precomputed',
                                algorithm='brute', eps=threshold, min_samples=2)
    return labels

def agglomarative_clustering(edgelist=None, distance_matrix=None, num_clusters=8, method='complete', metric='precomputed'):
    """ computes an agglomerative clustering as one of the hierarchical clustering methods """
    if edgelist is not None:
        distance_matrix, names = utils.edgelist_to_distance_matrix(edgelist)

    #all_methods =  _cpy_linkage_methods
    #all_metrics = ['precomputed', 'cosine', 'euclidean', 'cityblock', 'manhattan']

    all_methods = ['average']
    all_metrics = ['precomputed']

    for method in all_methods:
        for metric in all_metrics:
             if method == 'ward' and metric != 'euclidean': continue
             model = sklearn.cluster.AgglomerativeClustering(linkage=method, affinity=metric,
                                                         n_clusters=num_clusters, connectivity=distance_matrix, compute_full_tree='auto')
             model = model.fit(distance_matrix)
             labels = model.labels_

        print method, metric
    return labels


def spectral(edgelist=None, distance_matrix=None):
    """ cluster using spectral clustering """

    if edgelist is not None:
        distance_matrix, names = utils.edgelist_to_distance_matrix(edgelist)


    sc = sklearn.cluster.SpectralClustering(n_clusters=10, affinity='precomputed')
    labels = sc.fit_predict(distance_matrix)

    return labels

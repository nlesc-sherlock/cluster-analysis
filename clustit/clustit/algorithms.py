""" module containing the implementations of clustering algorithms used in clustit """

from __future__ import print_function
import sys

import utils
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import _cpy_linkage_methods
import numpy

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
                                algorithm='brute', eps=threshold, min_samples=2)
    return labels

def agglomerative_clustering(edgelist=None, distance_matrix=None, num_clusters=4, method='complete', metric='precomputed'):
    """ computes an agglomerative clustering as one of the hierarchical clustering methods """
    if edgelist is not None:
        distance_matrix, names = utils.edgelist_to_distance_matrix(edgelist)

    num_clusters=int(raw_input("Enter the number of clusters: "))
    assert isinstance(num_clusters, int)


    method_options = _cpy_linkage_methods
    method_options.remove('single')

    metric_options = ['precomputed', 'cosine', 'euclidean', 'cityblock']
    print('The list of available metrics:', metric_options , file=sys.stdout)
    metric_options = set([])
    in_metric = input('Input the metric name:')
    assert isinstance(in_metric, str)    # native str on Py2 and Py3
    metric_options.add(in_metric)

    #tree_cutoff_options = [True, False, 'auto']
    tree_cutoff_options = []



    print('The list of available methods:', method_options, file=sys.stdout)
    method_options = set([])
    in_method = input('Input the method name:')
    assert isinstance(in_method, str)    # native str on Py2 and Py3
    method_options.add(in_method)




    for method in method_options:
        for metric in metric_options:
            #for tree_cutoff in tree_cutoff_options:
            if method == 'ward' and metric != 'euclidean': continue
            model = sklearn.cluster.AgglomerativeClustering(linkage=method, affinity=metric,
                                                             n_clusters=num_clusters, connectivity=distance_matrix, compute_full_tree='auto')
            model = model.fit(distance_matrix)
            labels = model.labels_

        print(method, metric)
        return labels


def spectral(edgelist=None, distance_matrix=None):
    """ cluster using spectral clustering """

    if edgelist is not None:
        distance_matrix, names = utils.edgelist_to_distance_matrix(edgelist)


    sc = sklearn.cluster.SpectralClustering(n_clusters=10, affinity='precomputed')
    labels = sc.fit_predict(distance_matrix)

    return labels

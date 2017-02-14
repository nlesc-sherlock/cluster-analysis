""" module containing the plot functions used in clustit """

import pylab
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from matplotlib import pyplot
import numpy




def compute_linkage(matrix, method='complete',metric='euclidean'):
    Y = sch.linkage(matrix, method=method,metric=metric)
    return Y


def compute_dendrogram(linkage):
    dendrogram = sch.dendrogram(linkage, orientation='right')
    return dendrogram

def get_clusters_from_dendogram(dendrogram, label='leaves'):
    #this function is adapted from from:
    #http://nxn.se/post/90198924975/extract-cluster-elements-by-color-in-python
    colors = set(dendrogram['color_list'])
    cluster_idxs = {c: [] for c in colors}
    for c, pi in zip(dendrogram['color_list'], dendrogram['icoord']):
        for leg in pi[1:3]:
            i = (leg - 5.0) / 10.0
            if abs(i - int(i)) < 1e-5:
                cluster_idxs[c].append(int(i))

    cluster_classes = {}
    for c, l in cluster_idxs.items():
        i_l = [dendrogram[label][i] for i in l]
        cluster_classes[c] = i_l

    return cluster_classes



def plot_dendrogram(clusters, **kwargs):
    from scipy.cluster.hierarchy import dendrogram
    leaves = clusters.children_

    # Distances between each pair of leaves
    # Using uniform one for plotting
    distance = numpy.arange(leaves.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = numpy.arange(2, leaves.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = numpy.column_stack([leaves, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

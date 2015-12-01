import pylab
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from matplotlib import pyplot
import numpy


def plot_distance_matrices(matrix1, matrix2, matrix3):
    f, (ax1, ax2, ax3) = pyplot.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
    ax1.set_adjustable('box-forced')
    ax2.set_adjustable('box-forced')
    ax3.set_adjustable('box-forced')

    ax1.imshow(matrix_pce, vmin=0.0, vmax=20.0)
    ax1.set_title("PCE distance")
    ax2.imshow(matrix_ncc, vmin=0.0, vmax=20.0)
    ax2.set_title("NCC distance")
    ax3.imshow(matrix, vmin=0.0, vmax=20.0)
    ax3.set_title("Combined distance")
    pyplot.show()
    raw_input()



def compute_linkage(matrix, method='complete'):
    Y = sch.linkage(matrix, method=method)
    return Y


def compute_dendrogram(linkage):
    dendrogram = sch.dendrogram(linkage, orientation='right')
    return dendrogram


def plot_dendrogram_and_matrix(linkage, matrix):
    # Compute and plot dendrogram.
    fig = pylab.figure(figsize=(20,20))
    axdendro = fig.add_axes([0.09,0.1,0.2,0.8])
    dendrogram = sch.dendrogram(linkage, orientation='right')
    axdendro.set_xticks([])
    axdendro.set_yticks([])

    # Plot distance matrix.
    axmatrix = fig.add_axes([0.3,0.1,0.6,0.8])
    index = dendrogram['leaves']
    D = matrix[:]
    D = D[index,:]
    D = D[:,index]
    im = axmatrix.matshow(D, aspect='auto', origin='lower')
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])

    # Plot colorbar.
    axcolor = fig.add_axes([0.91,0.1,0.02,0.8])
    pylab.colorbar(im, cax=axcolor)

    # Display and save figure.
    fig.show()
    raw_input()

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










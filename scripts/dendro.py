#!/usr/bin/env python

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



def compute_linkage(matrix):
    Y = sch.linkage(matrix, method='complete')
    #Y = sch.linkage(matrix, method='average')
    #Y = sch.linkage(matrix, method='single')
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


def get_cluster_classes(dendrogram, label='leaves'):
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











if __name__ == "__main__":


    matrix_pce = numpy.fromfile("../data/set_2/matrix_304_pce.dat", dtype=numpy.float)
    matrix_ncc = numpy.fromfile("../data/set_2/matrix_304_ncc.dat", dtype=numpy.float)

    numfiles = int(numpy.sqrt(matrix_ncc.size))

    #map the ncc values into the range of pce values and shift to align medians
    matrix_ncc = matrix_ncc * 10000.0
    diff = numpy.median(matrix_pce) - numpy.median(matrix_ncc)
    matrix_ncc += diff
    matrix_ncc[matrix_ncc < 0.0] = 0.0

    matrix_pce[matrix_pce > 200.0] = 200.0
    matrix_ncc[matrix_ncc > 200.0] = 200.0

    #prevent div by zero
    matrix_pce += 0.0000001
    matrix_ncc += 0.0000001

    #convert similarity score to distance
    matrix_pce = 200.0 / matrix_pce
    matrix_ncc = 200.0 / matrix_ncc

    #set maximum distance at 200.0
    matrix_pce[matrix_pce > 200.0] = 200.0
    matrix_ncc[matrix_ncc > 200.0] = 200.0

    #pylab.hist(matrix_pce, 200)
    #pylab.figure()
    #pylab.hist(matrix_ncc, 200)
    #pylab.show()


    #reshape to square matrix form
    matrix_pce = matrix_pce.reshape(numfiles,numfiles)
    matrix_ncc = matrix_ncc.reshape(numfiles,numfiles)

    #zero diagonal
    index = range(numfiles)
    matrix_pce[index, index] = 0.0
    matrix_ncc[index, index] = 0.0


    #experiment with methods for combining the distance matrices into one
    matrix = numpy.minimum(matrix_pce, matrix_ncc)  #minimum distance
    #matrix = numpy.sqrt(matrix_pce * matrix_ncc)   #geometric mean
    #matrix = (matrix_pce + matrix_ncc) / 2.0       #arithmetic mean

    #pylab.hist(matrix.ravel(), 200)
    #pylab.show()

    #plot_distance_matrices(matrix_pce, matrix_ncc, matrix)

    linkage = compute_linkage(matrix)

    dendrogram = plot_dendrogram_and_matrix(linkage, matrix)

    clusters = get_cluster_classes(dendrogram)

    i=0
    print "\nfound clusters"
    for c in clusters.values():
        if len(c) > 0:
            print "cluster ", i
            i+=1
            print sorted(c)


    #get the actual clustering
    filelist = numpy.loadtxt("../data/set_2/filelist.txt", dtype=numpy.string_)
    true_clustering = numpy.array([s.split("_")[-2] for s in filelist])
    index = numpy.array(range(len(filelist)))
    colors = set(true_clustering)
    true_clusters = [index[true_clustering == c] for c in colors]

    i=0
    print "\nactual clusters"
    for c in true_clusters:
        print "cluster ", i
        i+=1
        print sorted(c)





    threshold = 0.7*max(linkage[:,2]) # default threshold used in sch.dendogram



    #go interactive
    #import readline
    #import rlcompleter
    #readline.parse_and_bind("tab: complete")
    #import code
    #code.interact(local=dict(globals(), **locals()))







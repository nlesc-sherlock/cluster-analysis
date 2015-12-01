#!/usr/bin/env python

from matplotlib import pyplot
import numpy

import dendro
import exif

import scipy.cluster.hierarchy as sch


def map_ncc_scores_to_pce_domain(matrix_pce, matrix_ncc):
    #map the ncc values into the range of pce values and shift to align medians
    matrix_ncc = matrix_ncc * 10000.0
    diff = numpy.median(matrix_pce) - numpy.median(matrix_ncc)
    matrix_ncc += diff
    matrix_ncc[matrix_ncc < 0.0] = 0.0

    return matrix_pce, matrix_ncc

def convert_similarity_to_distance(matrix):
    #cut off too high values
    matrix[matrix > 200.0] = 200.0

    #prevent div by zero
    matrix += 0.0000001

    #convert similarity score to distance
    matrix = 200.0 / matrix

    #set maximum distance at 200.0
    matrix[matrix > 200.0] = 200.0

    #reshape to square matrix form
    numfiles = int(numpy.sqrt(matrix.size))
    matrix = matrix.reshape(numfiles, numfiles)

    #zero diagonal
    index = range(numfiles)
    matrix[index, index] = 0.0

    return matrix







if __name__ == "__main__":


    matrix_pce = numpy.fromfile("../data/set_2/matrix_304_pce.dat", dtype=numpy.float)
    matrix_ncc = numpy.fromfile("../data/set_2/matrix_304_ncc.dat", dtype=numpy.float)

    matrix_pce, matrix_ncc = map_ncc_scores_to_pce_domain(matrix_pce, matrix_ncc)
    matrix_ncc = convert_similarity_to_distance(matrix_ncc)
    matrix_pce = convert_similarity_to_distance(matrix_pce)

    #experiment with methods for combining the distance matrices into one
    matrix = numpy.minimum(matrix_pce, matrix_ncc)  #minimum distance
    #matrix = numpy.sqrt(matrix_pce * matrix_ncc)   #geometric mean
    #matrix = (matrix_pce + matrix_ncc) / 2.0       #arithmetic mean

    #pylab.hist(matrix.ravel(), 200)
    #pylab.show()
    #plot_distance_matrices(matrix_pce, matrix_ncc, matrix)

    #hierarchical clustering part starts here
    linkage = dendro.compute_linkage(matrix)

    dendrogram = dendro.plot_dendrogram_and_matrix(linkage, matrix)

    #clusters = dendro.get_clusters_from_dendogram(dendrogram)

    #compute flat clustering in the exact same way as sch.dendogram colors the clusters
    threshold = 0.7*linkage[:,2].max() # default threshold used in sch.dendogram
    cluster = sch.fcluster(linkage, threshold, criterion='distance')
    print "flat clustering:\n", numpy.array(cluster) - 1

    #get the actual clustering
    filelist = numpy.loadtxt("../data/set_2/filelist.txt", dtype=numpy.string_)
    true_clustering = numpy.array([s.split("_")[-2] for s in filelist])
    print "true clustering:\n", numpy.array(true_clustering, dtype=numpy.int)

    #index = numpy.array(range(len(filelist)))
    #colors = set(true_clustering)
    #true_clusters = [index[true_clustering == c] for c in colors]
    #i=0
    #print "\nactual clusters"
    #for c in true_clusters:
    #    print "cluster ", i
    #    i+=1
    #    print sorted(c)




    #go interactive
    #import readline
    #import rlcompleter
    #readline.parse_and_bind("tab: complete")
    #import code
    #code.interact(local=dict(globals(), **locals()))









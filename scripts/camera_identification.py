#!/usr/bin/env python

from matplotlib import pyplot
import numpy

import dendro
import exif

import scipy.cluster.hierarchy as sch


def plot_distance_matrices(matrix1, matrix2, matrix3):
    f, (ax1, ax2, ax3) = pyplot.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
    ax1.set_adjustable('box-forced')
    ax2.set_adjustable('box-forced')
    ax3.set_adjustable('box-forced')

    vmax = matrix1.max()
    ax1.imshow(matrix1, vmin=0.0, vmax=vmax)
    ax1.set_title("PCE distance")
    ax2.imshow(matrix2, vmin=0.0, vmax=vmax)
    ax2.set_title("NCC distance")
    ax3.imshow(matrix3, vmin=0.0, vmax=vmax)
    ax3.set_title("Combined distance")

    size = matrix3.shape[0]
    def hover_func(x, y):
        x = int(x+1)
        y = int(y+1)
        if y >= size:
            y = size-1
        if x >= size:
            x = size-1
        return str(x) + "," + str(y) + " " + str(matrix1[y,x]) + " " + str(matrix2[y,x]) + " "+ str(matrix3[y,x])

    ax1.format_coord = hover_func
    ax2.format_coord = hover_func
    ax3.format_coord = hover_func
    pyplot.show()
    raw_input()

def map_ncc_scores_to_pce_domain(matrix_pce, matrix_ncc):
    #map the ncc values into the range of pce values and shift to align medians
    matrix_ncc = matrix_ncc * 10000.0
    diff = numpy.median(matrix_pce) - numpy.median(matrix_ncc)      #median works a bit better
    #diff = numpy.average(matrix_pce) - numpy.average(matrix_ncc)
    matrix_ncc += diff
    matrix_ncc[matrix_ncc < 0.0] = 0.0

    return matrix_pce, matrix_ncc

def convert_similarity_to_distance(matrix):
    #cut off too high values
    matrix[matrix > 200.0] = 200.0
    matrix[matrix < 0.0] = 0.0

    #prevent div by zero
    matrix += 0.0000001

    #convert similarity score to distance, tried various options
    matrix = 200.0 / matrix                 #best for the moment
    #matrix = 200 - matrix
    #matrix = numpy.sqrt(200.0 - matrix)
    #matrix = - numpy.log(matrix/200.0)     #sort of okay, also had FP

    #set maximum distance at 200.0
    matrix[matrix > 200.0] = 200.0

    #reshape to square matrix form
    numfiles = int(numpy.sqrt(matrix.size))
    matrix = matrix.reshape(numfiles, numfiles)

    #zero diagonal
    index = range(numfiles)
    matrix[index, index] = 0.0

    return matrix

def combine_pce_and_ncc_distances(matrix_pce, matrix_ncc):
    #experiment with methods for combining the distance matrices into one
    matrix = numpy.minimum(matrix_pce, matrix_ncc)  #minimum distance
    #matrix = numpy.sqrt(matrix_pce * matrix_ncc)   #geometric mean
    #matrix = (matrix_pce + matrix_ncc) / 2.0       #arithmetic mean

    #print "pce median max", numpy.median(matrix_pce), matrix_pce.max()
    #print "ncc median max", numpy.median(matrix_ncc), matrix_ncc.max()
    #maxval = matrix_pce.max()
    #matrix[matrix > maxval] = maxval
#    numfiles = int(numpy.sqrt(len(matrix_ncc)))

#    for i in range(numfiles):
#        for j in range(numfiles):
#            if matrix_pce[i,j]

    return matrix






if __name__ == "__main__":

    #load the distance matrixes from files
    matrix_pce = numpy.fromfile("../data/set_2/matrix_304_pce.dat", dtype=numpy.float)
    matrix_ncc = numpy.fromfile("../data/set_2/matrix_304_ncc.dat", dtype=numpy.float)

    matrix_pce, matrix_ncc = map_ncc_scores_to_pce_domain(matrix_pce, matrix_ncc)
    matrix_ncc = convert_similarity_to_distance(matrix_ncc)
    matrix_pce = convert_similarity_to_distance(matrix_pce)

    matrix = combine_pce_and_ncc_distances(matrix_pce, matrix_ncc)

    #import pylab
    #pylab.hist(matrix_pce.ravel(), 200, label='PCE')
    #pylab.hist(matrix_ncc.ravel(), 200, label='NCC')
    #pylab.hist(matrix.ravel(), 200, label='Combined')
    #pylab.legend()
    #pylab.show()

    #plot_distance_matrices(matrix_pce, matrix_ncc, matrix)

    #hierarchical clustering part starts here
    linkage = dendro.compute_linkage(matrix)

    #dendrogram = dendro.compute_dendrogram(linkage)
    #dendrogram = dendro.plot_dendrogram_and_matrix(linkage, matrix)

    #clusters = dendro.get_clusters_from_dendogram(dendrogram)

    #compute flat clustering in the exact same way as sch.dendogram colors the clusters
    threshold = 0.7*linkage[:,2].max() # default threshold used in sch.dendogram
    cluster = numpy.array(sch.fcluster(linkage, threshold, criterion='distance'), dtype=numpy.int)
    print "flat clustering:\n", cluster - 1

    #get the actual clustering
    filelist = numpy.loadtxt("../data/set_2/filelist.txt", dtype=numpy.string_)
    true_clustering = numpy.array([s.split("_")[-2] for s in filelist], dtype=numpy.int)
    print "true clustering:\n", true_clustering

    #try some metrics from sklearn
    from sklearn import metrics
    print "\n"
    print "adjusted rand score [-1.0 (bad) to 1.0 (good)]", metrics.adjusted_rand_score(true_clustering, cluster)
    print ""
    print "mutual information based score [0.0 (bad) to 1.0 (good)]", metrics.adjusted_mutual_info_score(true_clustering, cluster)
    print ""
    print "homogeneity, completeness, v measure [0.0 (bad) to 1.0 (good)]", metrics.homogeneity_completeness_v_measure(true_clustering, cluster)
    print ""



    #function to rename the labels from the found clustering to the labels in the true clustering
    #not very useful but may come in handy at some point
    def rename_clusters(clustering, true_clustering):
        clustering = numpy.array(clustering)
        #initialize new labels as -1
        labels = numpy.zeros(clustering.shape, dtype=numpy.int) -1
        true_clustering = numpy.array(true_clustering)
        clusters = set(clustering)
        #for each cluster that we found
        for c in clusters:
            #get the ids of the clusters its values really belong to
            true_ids = true_clustering[clustering == c]
            if len(true_ids) > 0:
                id = int(numpy.median(true_ids))
                #using all 80% match here
                #could use most common value or something
                matches = list(true_ids).count(id)
                if matches > 0.8*len(true_ids):
                    labels[clustering == c] = id
                #else:
                #    print "problem at c=", c, "id=", id, "true_ids=", true_ids


        #do something with clusters that we could not assign a new name
        for i in range(len(clustering)):
            if labels[i] == -1:
                #labels[i] = clustering[i] #reuse old name, problem: could merge clusters unindentedly
                labels[i] = 9 #assign arbitrary cluster id, lumping everything we could not match into a 'rest' cluster

        return labels

#    print "renamed clustering:\n", rename_clusters(cluster, true_clustering)



    #go interactive
    #import readline
    #import rlcompleter
    #readline.parse_and_bind("tab: complete")
    #import code
    #code.interact(local=dict(globals(), **locals()))









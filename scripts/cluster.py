#!/usr/bin/env python

import numpy
import pandas
import itertools
import scipy.cluster.hierarchy as sch

from matplotlib import pyplot

#S = numpy.fromfile("/var/scratch/bwn200/results/matrix-prnu-testcase.dat", dtype='>d')
#S = numpy.fromfile("/var/scratch/bwn200/results/matrix-pentax-pce.dat", dtype='>d')
S = numpy.fromfile("/home/bwn200/cluster-analysis/prnuextract/matrix-pentax-pce.dat", dtype='>d')
#S = numpy.fromfile("/home/bwn200/cluster-analysis/prnuextract/matrix-clusteringtest.dat", dtype='>d')



N = int(numpy.sqrt(S.size))
matrix = S.reshape(N,N)


def read_linkage_file(file):
    df = pandas.read_csv(file, header=None,
        names=['a', 'n1', 'n2', 'dist', 'csize'], delimiter=r'[\[\],\s]+',
        engine='python',index_col=False, skipinitialspace=True)

    n1 = numpy.array(df.n1)
    n2 = numpy.array(df.n2)
    dist = numpy.array(df.dist)
    csize = numpy.array(df.csize)
    return list(zip(n1,n2,dist,csize))


def complete(matrix):
    matrix_copy = numpy.copy(matrix)

    N = matrix.shape[0]
    next_cluster_id = N

    cluster_ids = list(range(N))
    cluster_sizes = {n : 1 for n in range(N)}
    linkage = []

    for i in range(N-1):
        #find the most similar pair of clusters
        index_max = matrix_copy.argmax()
        n1, n2 = numpy.unravel_index(index_max, (N,N))

        #rename n1 to a new cluster id
        cluster1 = cluster_ids[n1]
        cluster2 = cluster_ids[n2]
        cluster_sizes[next_cluster_id] = cluster_sizes[cluster1] + cluster_sizes[cluster2]
        cluster_ids[n1] = next_cluster_id

        #add to linkage
        linkage.append([cluster1, cluster2, matrix_copy[n1,n2], cluster_sizes[next_cluster_id]])

        #increment cluster id
        next_cluster_id += 1

        #update distances
        matrix_copy[n1, :] = numpy.fmin(matrix_copy[n1,:], matrix_copy[n2, :])
        matrix_copy[:, n1] = matrix_copy[n1, :]
        #effectively "remove" n2
        matrix_copy[n2, :] = 0
        matrix_copy[:, n2] = 0

    return linkage



def flat_clustering(linkage, threshold):
    N = len(linkage)+1
    next_cluster_id = N
    cluster_members = {n : [n] for n in range(N)}
    for i in range(N-1):
        if linkage[i,2] < threshold:
            break
        cluster_members[next_cluster_id] = cluster_members.pop(linkage[i,0]) + cluster_members.pop(linkage[i,1])
        next_cluster_id += 1
    labeling = numpy.zeros((N), dtype=numpy.int)
    label = 1
    for k,v in cluster_members.items():
        for i in v:
            labeling[i] = label
        label += 1
    return labeling, cluster_members

def false_positives(clusters, gt, verbose=False):
    """ return absolute number of false positives """
    fp = 0
    for k,v in clusters.items():
        pairs = list(itertools.product(v,v))
        for p in pairs:
            if gt[p[0]] != gt[p[1]]:
                fp += 1
                if verbose:
                    print("FP at:", p[0], "and", p[1])
    return fp

def false_negatives(cluster_labels, gt):
    """ return absolute number of false negatives """
    fn = 0
    gt_clusters = {k:[] for k in list(set(gt))}
    for i in range(len(gt)):
        gt_clusters[gt[i]].append(i)
    for k,v in gt_clusters.items():
        pairs = list(itertools.product(v,v))
        for p in pairs:
            if cluster_labels[p[0]]!=cluster_labels[p[1]]:
                fn += 1
    return fn

def average(matrix):
    matrix_copy = numpy.copy(matrix)

    N = matrix.shape[0]
    next_cluster_id = N

    cluster_ids = list(range(N))
    cluster_members = {n : [n] for n in range(N)}
    linkage = []

    for i in range(N-1):
        #find the most similar pair of clusters
        index_max = matrix_copy.argmax()
        n1, n2 = numpy.unravel_index(index_max, (N,N))

        #rename n1 to a new cluster id
        cluster1 = cluster_ids[n1]
        cluster2 = cluster_ids[n2]
        cluster_members[next_cluster_id] = cluster_members.pop(cluster1) + cluster_members.pop(cluster2)
        cluster_ids[n1] = next_cluster_id

        #add to linkage
        linkage.append([cluster1, cluster2, matrix_copy[n1,n2], len(cluster_members[next_cluster_id])])

        #update similarities
        for i in range(N):
            if i in cluster_members:
                sum = 0.0
                for a in cluster_members[next_cluster_id]:
                    for b in cluster_members[i]:
                        sum += matrix[a,b]
                matrix_copy[n1, i] = sum / (len(cluster_members[next_cluster_id])*len(cluster_members[i]))

        matrix_copy[:, n1] = matrix_copy[n1, :]
        #effectively "remove" n2
        matrix_copy[n2, :] = 0
        matrix_copy[:, n2] = 0

        #increment cluster id
        next_cluster_id += 1

    return linkage



#matrix = numpy.sqrt(matrix)
#max_v = numpy.amax(matrix)
#matrix = 1.0 - (matrix/(1.0-max_v))


#linkage = sch.linkage(matrix, method='average')


#matrix = numpy.log2(matrix)


#linkage = complete(matrix)
#linkage = average(matrix)

#file = '/home/bwn200/cluster-analysis/prnuextract/linkage-clusteringtest.txt'


file = '/home/bwn200/cluster-analysis/prnuextract/linkage-pentax-pce.txt'
linkage = read_linkage_file(file)


#print("linkage:")
#for l in linkage:
#    print(l)


linkage = numpy.array(linkage)


with open('/home/bwn200/cluster-analysis/data/pentax/filelist.txt', 'r') as f:
    filelist = f.read().split('\n')[:-1]
camlist = ["_".join(f.split("_")[:-1]) for f in filelist]
ground_truth_labels = numpy.array([int(i.split("_")[-1][0]) for i in camlist])


labels, clusters = flat_clustering(linkage, 60)
fp = false_positives(clusters, ground_truth_labels, verbose=True)

print(filelist[256], filelist[568])


num_clusters = []
thresholds = []
fps = []
fns = []


for threshold in numpy.linspace(0,70,100):
    labels, clusters = flat_clustering(linkage, threshold)
    fp = false_positives(clusters, ground_truth_labels)
    fp_rate = (fp/(N*(N-1)))*100.0
    fn = false_negatives(labels, ground_truth_labels)
    fn_rate = (fn/(N*(N-1)))*100.0
    thresholds.append(threshold)
    fps.append(fp_rate)
    fns.append(fn_rate)
    num_clusters.append(len(clusters))



print(numpy.array(list(zip(thresholds, fns, fps, num_clusters))))




def make_dual_plot(title, x_series, x_label, y1_series, y1_label, y2_series, y2_label):
    f, ax1 = pyplot.subplots()
    ax1.set_title(title)
    ax1.plot(x_series, y1_series, 'b-')
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y1_label, color='b')
    ax1.tick_params('y', colors='b')
    ax1.set_ylim([0,80])

    ax2 = ax1.twinx()
    ax2.plot(x_series, y2_series, 'g-')
    ax2.set_ylabel(y2_label, color='g')
    ax2.tick_params('y', colors='g')
    ax2.set_ylim([0,80])

    f.set_size_inches(10, 6, forward=True)
    f.tight_layout()

    f.savefig("plot.png", dpi=300)
    f.savefig("plot.eps", format="eps")


make_dual_plot("Varying threshold for pentax dataset", thresholds, "threshold", fps, "FPR (%)", fns, "FNR (%)")

pyplot.show()

exit()

threshold = 60.0

#linkage[:,2][linkage[:,2] > 200.0] = 200.0

linkage[:,2] = numpy.log10(linkage[:,2])
max_s = max(linkage[:,2])
linkage[:,2] = 1.0 - (linkage[:,2] / (max_s+1.0))

t_orig = numpy.log10(threshold)
threshold = 1.0 - (t_orig / (max_s+1.0))
print("max_s", max_s, "threshold", threshold)



#this one works so far
#linkage[:,2] = 200 / linkage[:,2]
#print(linkage)
#fcluster = sch.fcluster(linkage, 200/30, criterion='distance')
#print(fcluster)


#from matplotlib import pyplot
#sch.dendrogram(linkage, color_threshold=threshold)

#pyplot.show()


from dendro import plot_dendrogram_and_matrix

matrix[matrix>200] = 200

plot_dendrogram_and_matrix(linkage, matrix, color_threshold=threshold, title="Dendrogram and similarity matrix for Pentax dataset")






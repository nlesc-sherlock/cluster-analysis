#!/usr/bin/env python

import numpy

#S = numpy.fromfile("/var/scratch/bwn200/results/matrix-prnu-testcase.dat", dtype='>d')
S = numpy.fromfile("/var/scratch/bwn200/results/matrix-pentax-pce.dat", dtype='>d')

N = 638
matrix = S.reshape(N,N)


def complete(matrix):
    N = matrix.shape[0]
    next_cluster_id = N

    cluster_ids = list(range(N))
    cluster_sizes = {n : 1 for n in range(N)}
    linkage = []

    for i in range(N-1):
        #find the most similar pair of clusters
        index_max = matrix.argmax()
        n1, n2 = numpy.unravel_index(index_max, (N,N))

        #rename n1 to a new cluster id
        cluster1 = cluster_ids[n1]
        cluster2 = cluster_ids[n2]
        cluster_sizes[next_cluster_id] = cluster_sizes[cluster1] + cluster_sizes[cluster2]
        cluster_ids[n1] = next_cluster_id

        #add to linkage
        linkage.append([cluster1, cluster2, matrix[n1,n2], cluster_sizes[next_cluster_id]])

        #increment cluster id
        next_cluster_id += 1

        #update distances
        matrix[n1, :] = numpy.fmin(matrix[n1,:], matrix[n2, :])
        matrix[:, n1] = matrix[n1, :]
        #effectively "remove" n2
        matrix[n2, :] = 0
        matrix[:, n2] = 0

    return linkage




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





#linkage = complete(matrix)
linkage = average(matrix)

print("linkage:")
for l in linkage:
    print(l)


linkage = numpy.array(linkage)



import scipy.cluster.hierarchy as sch



#linkage[:,2][linkage[:,2] > 200.0] = 200.0

linkage[:,2] = numpy.log2(linkage[:,2])
max_s = max(linkage[:,2])
linkage[:,2] = 1.0 - (linkage[:,2] / (max_s+1.0))
t_orig = numpy.log2(60)
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

plot_dendrogram_and_matrix(linkage, matrix, color_threshold=threshold)

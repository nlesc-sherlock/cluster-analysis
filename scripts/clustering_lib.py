
import numpy
##from sklearn import cluster, datasets


def performFlatClustering(numClusters, distance, inputData):

    ## two algorithms with initial number of clusters
    from sklearn import cluster
    two_means = cluster.MiniBatchKMeans(n_clusters=numClusters)
    birch = cluster.Birch(n_clusters=numClusters)

    ## DBSSCAN with predefined similarity distance
    dbscan = cluster.DBSCAN(eps=.2)

    ## clustering with MeanShift and automatically estimate the region sizes
    regionSize = cluster.estimate_bandwidth(inputData, quantile=0.2, n_samples=len(inputData))
    ms = cluster.MeanShift(bandwidth=regionSize, bin_seeding=True)

    affinity_propagation = cluster.AffinityPropagation(affinity="precomputed")  # , damping=.9, preference=-200)
    affinity_propagation.fit_predict(inputData)

    algorithms = [two_means, birch, ms]
    for algo in algorithms:
        algo.fit(inputData)

    return


def perform_hierarcy_clustering(num_clusters, connect_matrix):

    #import hierarchy as cluster
    from sklearn import cluster
    from matplotlib import pyplot as plt

    linkageMethods = ['ward', "average"]


    #cluster.hierarchical.return_distance = True
    #cluster.hierarchical.ward_tree(connect_matrix,return_distance=True)
    #cluster.hierarchical.linkage_tree(connect_matrix,return_distance=True)

    #TODO: jaccard index with ward model


    agg_cluster = cluster.AgglomerativeClustering(linkage=linkageMethods[1], affinity="precomputed",
                                                n_clusters=num_clusters, connectivity=connect_matrix, compute_full_tree=True)
    #print agg_cluster.fit(connect_matrix)

    #feature_cluster = cluster.FeatureAgglomeration(linkage=linkageMethods[0], affinity="precomputed",
    #                                            n_clusters=num_clusters, connectivity=connect_matrix)

    models = [agg_cluster]#, feature_cluster]
    for algo in models:
        algo.fit(connect_matrix)

    #return agg_cluster.fit(connect_matrix).connectivity

    model = agg_cluster.fit(connect_matrix)
    plot_dendrogram(model, labels=model.labels_)
    plt.show()




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

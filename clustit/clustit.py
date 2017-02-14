#!/usr/bin/env python
""" Clustit - A simple tool for clustering data

Clustit is a command-line tool for clustering data based on an edgelist or distance matrix.
Clustit should serve as an easy way of accessing the different clustering algorithms already implemented in Python.

Dependencies
------------
 * Python 2.7 or Python 3.5
 * Scipy
 * scikit-learn
 * hdbscan (https://github.com/scikit-learn-contrib/hdbscan)
 * Pandas
 * LargeVis (https://github.com/lferry007/LargeVis)

Example usage
-------------
```
usage: clustit.py [-h] (-e edgelist | -m matrix) [-n names] [-c convert]
                  clustering_algorithm

specify either:
  -e edgelist, --edgelist edgelist
                        name of the edgelist file
  -m matrix, --matrix matrix
                        name of distance matrix file

positional arguments:
  clustering_algorithm  name of the clustering algorithm to use
                        choose from: hierarchical, dbscan, hdbscan, agglomerative, spectral

optional arguments:
  -h, --help            show this help message and exit
  -n names, --names names
                        filename storing a list of names for the items to be
                        clustered, in case distance matrix is used
  -c convert, --convert convert
                        convert similarity to distance with specified a cut-
                        off value

Examples:
./clustit.py -m ../data/pentax/matrix-pentax-pce.dat --convert=200 hierarchical
./clustit.py -e ../data/pentax/edgelist-pentax-pce.txt --convert=200 dbscan
./clustit.py -e ../data/pentax/edgelist-pentax-pce.txt --convert=200 agglomerative
```

Copyright and License
---------------------
* Copyright 2016 Netherlands eScience Center

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import print_function
import numpy
import argparse
import clustit.utils as utils
from clustit.algorithms import *

def parse_arguments():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("-e", "--edgelist", help="name of the edgelist file", metavar='edgelist')
    mode.add_argument("-m", "--matrix", help="name of distance matrix file", metavar='matrix')
    parser.add_argument("-n", "--names", help="filename storing a list of names for the items to be clustered, in case distance matrix is used", metavar='names')
    parser.add_argument("-c", "--convert", help="convert similarity to distance with specified a cut-off value", metavar='convert')
    parser.add_argument("clustering_algorithm", help="name of the clustering algorithm to use", choices=["hierarchical", "dbscan", "hdbscan", "spectral", "agglomerative"], metavar='clustering_algorithm')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    edgelist = None
    if args.edgelist:
        print("edgelist filename=" + args.edgelist)
        edgelist = utils.read_edgelist_file(args.edgelist)

    matrix = None
    if args.matrix:
        print("matrix filename=" + args.matrix)
        matrix = utils.read_distance_matrix_file(args.matrix)

    if args.convert:
        print("convert=" + args.convert)
        if args.edgelist:
            edgelist['d'] = utils.similarity_to_distance(edgelist['d'], float(args.convert))
        if args.matrix:
            matrix = utils.similarity_to_distance(matrix, float(args.convert))

    if args.names:
        print("names filenname=" + args.names)
    print("clustering_algorithm=" + args.clustering_algorithm)


    if args.clustering_algorithm == 'hierarchical':
        clustering = hierarchical_clustering(edgelist=edgelist, distance_matrix=matrix)
    elif args.clustering_algorithm == 'dbscan':
        clustering = dbscan(edgelist=edgelist, distance_matrix=matrix)
    elif args.clustering_algorithm == 'hdbscan':
        clustering = hierarchical_dbscan(edgelist=edgelist, distance_matrix=matrix)
    elif args.clustering_algorithm == 'spectral':
        clustering = spectral(edgelist=edgelist, distance_matrix=matrix)
    elif args.clustering_algorithm == 'agglomerative':
        clustering = agglomerative_clustering(edgelist=edgelist, distance_matrix=matrix)

    numpy.set_printoptions(threshold=numpy.nan)
    print(clustering)

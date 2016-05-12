#!/usr/bin/env python
""" Clustit - A simple tool for clustering data

Clustit is a command-line tool for clustering data based on an edgelist or distance matrix.
Clustit should serve as an easy way of accessing the different clustering algorithms already implemented in Python.

Dependencies
------------
 * Python 2.7 or Python 3.5
 * Scipy
 * scikit-learn

Example usage
-------------
```
./clustit.py (-e EDGELIST | -m MATRIX) <clustering_algorithm> [optional arguments]

specify either:
  -e EDGELIST, --edgelist EDGELIST
                        name of the edgelist file
  -m MATRIX, --matrix MATRIX
                        name of distance matrix file

for clustering_algorithm, choose from:
    hierarchical or dbscan

optional arguments:
  -h, --help            show help message and exit
  -n NAMES, --names NAMES
                        filename storing a list of names for the items to be
                        clustered, in case distance matrix is used
  -c CONVERT, --convert CONVERT
                        convert similarity to distance with specified a cut-
                        off value

Example:
./clustit.py -m ../data/pentax/matrix-pentax-pce.dat --convert=60 hierarchical
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
    parser.add_argument("clustering_algorithm", help="name of the clustering algorithm to use", choices=["hierarchical", "dbscan", "agglomarative"], metavar='clustering_algorithm')
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
        edgelist, matrix = utils.convert_similarity_to_distance(edgelist, matrix, float(args.convert))

    if args.names:
        print("names filenname=" + args.names)
    print("clustering_algorithm=" + args.clustering_algorithm)


    if args.clustering_algorithm == 'hierarchical':
        clustering = hierarchical_clustering(edgelist=edgelist, distance_matrix=matrix)
    elif args.clustering_algorithm == 'dbscan':
        clustering = dbscan(edgelist=edgelist, distance_matrix=matrix)
    elif args.clustering_algorithm == 'agglomarative':
        clustering = agglomarative_clustering(edgelist=edgelist, distance_matrix=matrix)


    print(clustering)

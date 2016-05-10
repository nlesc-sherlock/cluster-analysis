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
`./clustit <name of data file> <name of clustering algorithm> [optional parameters]`

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

def parse_arguments():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("-e", "--edgelist", help="name of the edgelist file")
    mode.add_argument("-m", "--matrix", help="name of distance matrix file")
    parser.add_argument("-n", "--names", help="filename storing a list of names for the items to be clustered, in case distance matrix is used")
    parser.add_argument("clustering_algorithm", help="name of the clustering algorithm to use", choices=["hierarchical", "dbscan"])
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    if args.edgelist:
        print("edgelist filename=" + args.edgelist)
    if args.matrix:
        print("matrix filename=" + args.matrix)
    if args.names:
        print("names filenname=" + args.names)
    print("clustering_algorithm=" + args.clustering_algorithm)


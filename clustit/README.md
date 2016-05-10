Clustit - A simple tool for clustering data
===========================================

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


Contributing to clustit
-----------------------
The <i>clustit</i> tool follows the Google Python style guide, with Sphinxdoc docstrings for module public functions. If you want to
contribute to the project please fork it, create a branch including your addition, and create a pull request.

The tests use relative imports and can be run directly after making
changes to the code. To run all tests use `nosetests` in the main directory.

Before creating a pull request or committing changes, please ensure the following:
* You have written unit tests to test your additions
* All unit tests pass
* The code is compatible with both Python 2.7 and Python 3.5

Contributing authors so far:
* Arnold Kuzniar
* Rena Bakhshi
* Ben van Werkhoven


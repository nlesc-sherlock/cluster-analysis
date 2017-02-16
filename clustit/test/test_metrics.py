from __future__ import print_function

import numpy

from clustit import metrics


def test_fpr():
    ground_truth = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    cluster_labels = [1, 1, 0, 2, 1, 3, 4, 1, 5]
    fpr = metrics.fpr(ground_truth, cluster_labels)
    print(ground_truth)
    print(cluster_labels)
    print(fpr)
    expected = 10/81
    print(expected)
    assert fpr == expected


def test_tpr():
    ground_truth = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    cluster_labels = [1, 1, 0, 2, 1, 3, 4, 1, 5]
    tpr = metrics.tpr(ground_truth, cluster_labels)
    print(ground_truth)
    print(cluster_labels)
    print(tpr)
    expected = 2/81
    print(expected)
    assert tpr == expected

def test_fnr():
    ground_truth = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    cluster_labels = [1, 1, 0, 2, 1, 3, 4, 1, 5]
    fnr = metrics.fnr(ground_truth, cluster_labels)
    print(ground_truth)
    print(cluster_labels)
    print(fnr)
    expected = 16/81  #four in the first cluster, six in the other two
    print(expected)
    assert fnr == expected


def test_tnr():
    ground_truth = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    cluster_labels = [1, 1, 0, 2, 1, 3, 4, 1, 5]
    tnr = metrics.tnr(ground_truth, cluster_labels)
    print(ground_truth)
    print(cluster_labels)
    print(tnr)
    expected = sum([4, 4, 6, 3, 6, 6, 6, 3, 6]) / 81
    print(expected)
    assert tnr == expected


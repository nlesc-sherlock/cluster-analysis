from __future__ import print_function

import numpy

from clustit import metrics


def test_fp():
    ground_truth = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    cluster_labels = [1, 1, 0, 2, 1, 3, 4, 1, 5]
    fp = metrics.fp(ground_truth, cluster_labels)
    print(ground_truth)
    print(cluster_labels)
    print(fp)
    expected = 10
    print(expected)
    assert fp == expected


def test_tp():
    ground_truth = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    cluster_labels = [1, 1, 0, 2, 1, 3, 4, 1, 5]
    tp = metrics.tp(ground_truth, cluster_labels)
    print(ground_truth)
    print(cluster_labels)
    print(tp)
    expected = 2
    print(expected)
    assert tp == expected

def test_fn():
    ground_truth = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    cluster_labels = [1, 1, 0, 2, 1, 3, 4, 1, 5]
    fn = metrics.fn(ground_truth, cluster_labels)
    print(ground_truth)
    print(cluster_labels)
    print(fn)
    expected = 16  #four in the first cluster, six in the other two
    print(expected)
    assert fn == expected


def test_tn():
    ground_truth = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    cluster_labels = [1, 1, 0, 2, 1, 3, 4, 1, 5]
    tn = metrics.tn(ground_truth, cluster_labels)
    print(ground_truth)
    print(cluster_labels)
    print(tn)
    expected = sum([4, 4, 6, 3, 6, 6, 6, 3, 6])
    print(expected)
    assert tn == expected



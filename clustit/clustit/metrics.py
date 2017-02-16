
import itertools

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score, completeness_score, precision_score, recall_score


def compute_rates(ground_truth_labels, cluster_labels):
    cl = cluster_labels
    gt = ground_truth_labels
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    n = len(cl)
    for v1,v2 in itertools.product(range(n), range(n)):
        if v1!=v2 and cl[v1] == cl[v2] and gt[v1] != gt[v2]:
            fp += 1
        if v1!=v2 and cl[v1] == cl[v2] and gt[v1] == gt[v2]:
            tp += 1
        if v1!=v2 and cl[v1] != cl[v2] and gt[v1] == gt[v2]:
            fn += 1
        if v1!=v2 and cl[v1] != cl[v2] and gt[v1] != gt[v2]:
            tn += 1
    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)
    tnr = 1 - fpr
    fnr = 1 - tpr
    return fpr, tpr, fnr, tnr



def precision(ground_truth_labels, cluster_labels):
    ntp = tp(ground_truth_labels, cluster_labels)
    nfp = fp(ground_truth_labels, cluster_labels)
    return ntp / (ntp + nfp)


def recall(ground_truth_labels, cluster_labels):
    ntp = tp(ground_truth_labels, cluster_labels)
    nfn = fn(ground_truth_labels, cluster_labels)
    return ntp / (ntp + nfn)






def fp(ground_truth_labels, cluster_labels):
    """ Compute the fraction of edges that are false positives """
    cl = cluster_labels
    gt = ground_truth_labels
    fp = 0
    n = len(cl)
    for v1,v2 in itertools.product(range(n), range(n)):
        if v1!=v2 and cl[v1] == cl[v2] and gt[v1] != gt[v2]:
            fp += 1
    return fp


def tp(ground_truth_labels, cluster_labels):
    """ Compute the fraction of edges that are true positives """
    cl = cluster_labels
    gt = ground_truth_labels
    tp = 0
    n = len(cl)
    for v1,v2 in itertools.product(range(n), range(n)):
        if v1!=v2 and cl[v1] == cl[v2] and gt[v1] == gt[v2]:
            tp += 1
    return tp


def fn(ground_truth_labels, cluster_labels):
    """ Compute the fraction of edges that are false negatives """
    cl = cluster_labels
    gt = ground_truth_labels
    fn = 0
    n = len(cl)
    for v1,v2 in itertools.product(range(n), range(n)):
        if v1!=v2 and cl[v1] != cl[v2] and gt[v1] == gt[v2]:
            fn += 1
    return fn


def tn(ground_truth_labels, cluster_labels):
    """ Compute the fraction of edges that are true negatives """
    cl = cluster_labels
    gt = ground_truth_labels
    tn = 0
    n = len(cl)
    for v1,v2 in itertools.product(range(n), range(n)):
        if v1!=v2 and cl[v1] != cl[v2] and gt[v1] != gt[v2]:
            tn += 1
    return tn









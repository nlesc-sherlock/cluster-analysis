
import itertools

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score, completeness_score





def fpr(ground_truth_labels, cluster_labels):
    """ Compute the false positive rate from the labels given the ground truth labels """
    cl = cluster_labels
    gt = ground_truth_labels
    fp = 0
    n = len(cl)
    for v1,v2 in itertools.product(range(n), range(n)):
        if v1!=v2 and cl[v1] == cl[v2] and gt[v1] != gt[v2]:
            fp += 1
    fpr = fp / (len(cl)**2)
    return fpr

def tpr(ground_truth_labels, cluster_labels):
    """ Compute the true positive rate from the labels given the ground truth labels """
    cl = cluster_labels
    gt = ground_truth_labels
    tp = 0
    n = len(cl)
    for v1,v2 in itertools.product(range(n), range(n)):
        if v1!=v2 and cl[v1] == cl[v2] and gt[v1] == gt[v2]:
            tp += 1
    tpr = tp / (len(cl)**2)
    return tpr


def fnr(ground_truth_labels, cluster_labels):
    """ Compute the false positive rate from the labels given the ground truth labels """
    cl = cluster_labels
    gt = ground_truth_labels
    fn = 0
    n = len(cl)
    for v1,v2 in itertools.product(range(n), range(n)):
        if v1!=v2 and cl[v1] != cl[v2] and gt[v1] == gt[v2]:
            fn += 1
    fnr = fn / (len(cl)**2)
    return fnr


def tnr(ground_truth_labels, cluster_labels):
    """ Compute the true positive rate from the labels given the ground truth labels """
    cl = cluster_labels
    gt = ground_truth_labels
    tn = 0
    n = len(cl)
    for v1,v2 in itertools.product(range(n), range(n)):
        if v1!=v2 and cl[v1] != cl[v2] and gt[v1] != gt[v2]:
            tn += 1
    tnr = tn / (len(cl)**2)
    return tnr









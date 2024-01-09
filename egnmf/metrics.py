"""metrics.py"""
import numpy as np
from pyitlib import discrete_random_variable as drv
from munkres import Munkres


def calc_ac_score(labels_true, labels_pred):
    """calculate unsupervised accuracy score

    Parameters
    ----------
    labels_true: labels from ground truth
    labels_pred: labels form clustering

    Return
    -------
    ac: accuracy score
    """
    nclass = len(np.unique(labels_true))
    labels_size = len(labels_true)
    mat = labels_size * np.ones((nclass, nclass))

    for i in range(labels_size):
        mat[labels_pred[i], labels_true[i]] -= 1.0

    munkres = Munkres()
    mapping = munkres.compute(mat)

    ac = 0.0

    for i in range(labels_size):
        val = mapping[labels_pred[i]][1]
        if val == labels_true[i]:
            ac += 1.0

    ac = ac / labels_size

    return ac


def calc_nmi_score(labels_true, labels_pred):
    """calculate normalized mutual information score

    Parameters
    ----------
    labels_true: labels from ground truth
    labels_pred: labels from clustering

    Return
    -------
    nmi: normalized mutual information score
    """
    H_true = drv.entropy(labels_true, base=2)
    H_pred = drv.entropy(labels_pred, base=2)
    H_joint = drv.entropy_joint([labels_true, labels_pred], base=2)
    mi = H_true + H_pred - H_joint
    nmi = mi / max(H_true, H_pred)
    return nmi

from __future__ import print_function
import argparse
import os
import sys

from scipy import misc
import numpy as np
from collections import defaultdict


def get_curve(known, novel, threshold=None):
    """
    `known` is the positive class (In-distribution, ID),
    `novel` is the negative class (Out-of-distribution, OOD).
    """
    tp, fp = dict(), dict()
    fpr_at_tpr95 = dict()

    # Sort the scores of known and novel (ascending order)
    known.sort()
    novel.sort()

    # Compute the max and min scores (for later analysis)
    end = np.max([np.max(known), np.max(novel)])
    start = np.min([np.min(known), np.min(novel)])

    # Concatenate all scores and sort to traverse thresholds
    all = np.concatenate((known, novel))
    all.sort()

    # Count the number of samples in known (ID) and novel (OOD)
    num_k = known.shape[0]
    num_n = novel.shape[0]

    # Select threshold for computing FPR at TPR=95% (~5% quantile of known)
    if threshold is None:
        threshold = known[round(0.05 * num_k)]

    # Initialize tp (True Positive) and fp (False Positive) curve arrays
    tp = -np.ones([num_k + num_n + 1], dtype=int)
    fp = -np.ones([num_k + num_n + 1], dtype=int)
    tp[0], fp[0] = num_k, num_n  # Initially, all predicted positive

    k, n = 0, 0  # k = traversed known count, n = traversed novel count
    for l in range(num_k + num_n):
        if k == num_k:
            # All known traversed, only novel left
            tp[l+1:] = tp[l]
            fp[l+1:] = np.arange(fp[l] - 1, -1, -1)
            break
        elif n == num_n:
            # All novel traversed, only known left
            tp[l+1:] = np.arange(tp[l] - 1, -1, -1)
            fp[l+1:] = fp[l]
            break
        else:
            # Compare known and novel scores in ascending order
            if novel[n] < known[k]:
                n += 1
                tp[l+1] = tp[l]      # novel is negative, tp unchanged
                fp[l+1] = fp[l] - 1  # novel wrongly counted as positive, decrease fp
            else:
                k += 1
                tp[l+1] = tp[l] - 1  # known correctly recognized as positive, decrease tp
                fp[l+1] = fp[l]      # fp unchanged

    # Post-processing to remove jumps caused by duplicate scores
    j = num_k + num_n - 1
    for l in range(num_k + num_n - 1):
        if all[j] == all[j - 1]:
            tp[j] = tp[j + 1]
            fp[j] = fp[j + 1]
        j -= 1

    # Compute FPR at TPR=95%, equivalent to proportion of novel above threshold predicted positive
    fpr_at_tpr95 = np.sum(novel > threshold) / float(num_n)

    return tp, fp, fpr_at_tpr95


def cal_metric(known, novel, threshold=None):
    # Inputs are in-distribution scores `known` and OOD scores `novel`
    # method optional for different sorting (not used currently)

    # Get TPR curve, FPR curve, and FPR at 95% TPR
    tp, fp, fpr_at_tpr95 = get_curve(known, novel, threshold)

    results = dict()
    mtypes = ['FPR', 'AUROC', 'DTERR', 'AUIN', 'AUOUT']  # 5 common metrics

    # ----------- FPR@95TPR (False Positive Rate at 95% True Positive Rate) -----------
    mtype = 'FPR'
    results[mtype] = fpr_at_tpr95  # Directly returned by get_curve

    # ----------- AUROC (Area Under ROC Curve) -----------
    mtype = 'AUROC'
    # Construct TPR and FPR curves, adding start/end points for complete integration
    tpr = np.concatenate([[1.], tp / tp[0], [0.]])
    fpr = np.concatenate([[1.], fp / fp[0], [0.]])
    # Integrate using trapezoidal rule, note 1 - FPR for correct direction
    results[mtype] = -np.trapz(1. - fpr, tpr)

    # ----------- DTERR (Detection Error: minimum detection error rate) -----------
    mtype = 'DTERR'
    # (FN + FP) / total samples, tp[0] = total positives, fp[0] = total negatives
    # Finds the minimal error rate across thresholds
    results[mtype] = ((tp[0] - tp + fp) / (tp[0] + fp[0])).min()

    # ----------- AUIN (Area Under Inverse Precision-Recall Curve for ID) -----------
    mtype = 'AUIN'
    denom = tp + fp  # Denominator of precision
    denom[denom == 0.] = -1.  # Avoid division by zero
    # Only keep positions with denom > 0 for integration
    pin_ind = np.concatenate([[True], denom > 0., [True]])
    # Precision curve (insert 0.5 at start and 0 at end)
    pin = np.concatenate([[.5], tp / denom, [0.]])
    # Integrate precision vs recall curve (larger area is better)
    results[mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])

    # ----------- AUOUT (Area Under Inverse Precision-Recall Curve for OOD) -----------
    mtype = 'AUOUT'
    # Inverse precision denominator: FP + TN
    denom = tp[0] - tp + fp[0] - fp
    denom[denom == 0.] = -1.
    pout_ind = np.concatenate([[True], denom > 0., [True]])
    # OOD precision curve (recall dimension)
    pout = np.concatenate([[0.], (fp[0] - fp) / denom, [.5]])
    results[mtype] = np.trapz(pout[pout_ind], 1. - fpr[pout_ind])

    return results


def compute_all_metrics(scores_in_all, scores_out_all):
    all_results = defaultdict(dict)

    for forward_name in scores_in_all:
        for score in scores_in_all[forward_name]:
            # Ensure corresponding out scores exist
            if score in scores_out_all.get(forward_name, {}):
                scores_in = scores_in_all[forward_name][score]
                scores_out = scores_out_all[forward_name][score]

                # Compute metrics
                result = cal_metric(scores_in, scores_out)

                # Save results
                all_results[forward_name][score] = result
            else:
                print(f"[Warning] Missing out scores for: {forward_name}, {score}")
    return all_results

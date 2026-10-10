"""
evaluate.py
Metrics for the leave-category-out open-set / zero-day evaluation.

For a given held-out ("zero-day") category:
  - known-attack test set   = benign + attack categories seen at train time
  - unknown-attack test set = the held-out category only

We report:
  AUROC_known    : benign vs. known-attack separability (sanity check --
                   should stay high; a method that regresses here isn't a
                   viable IDS regardless of open-set performance)
  AUROC_unknown  : benign vs. UNKNOWN-attack separability -- the core
                   zero-day detection metric
  UDR (Unknown Detection Rate) : recall on the unknown category at the
                   operating threshold calibrated to 5% FPR on benign
                   (a fixed, deployable operating point, not an
                   idealized area-under-curve number)
  FPR_at_UDR     : the actual false-positive rate on benign at that
                   threshold (sanity companion to UDR)
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score


def threshold_at_fpr(benign_scores: np.ndarray, target_fpr: float = 0.05) -> float:
    """Return the score threshold such that target_fpr fraction of benign
    samples score at or above it (i.e. would be flagged)."""
    return np.quantile(benign_scores, 1 - target_fpr)


def evaluate_split(scores_benign, scores_known_attack, scores_unknown_attack, target_fpr=0.05):
    y_known = np.concatenate([np.zeros(len(scores_benign)), np.ones(len(scores_known_attack))])
    s_known = np.concatenate([scores_benign, scores_known_attack])
    auroc_known = roc_auc_score(y_known, s_known) if len(scores_known_attack) else float("nan")

    y_unknown = np.concatenate([np.zeros(len(scores_benign)), np.ones(len(scores_unknown_attack))])
    s_unknown = np.concatenate([scores_benign, scores_unknown_attack])
    auroc_unknown = roc_auc_score(y_unknown, s_unknown) if len(scores_unknown_attack) else float("nan")

    thresh = threshold_at_fpr(scores_benign, target_fpr)
    fpr_at_thresh = float(np.mean(scores_benign >= thresh))
    udr = float(np.mean(scores_unknown_attack >= thresh)) if len(scores_unknown_attack) else float("nan")

    return {
        "auroc_known": auroc_known,
        "auroc_unknown": auroc_unknown,
        "udr_at_5pct_fpr": udr,
        "actual_fpr_at_threshold": fpr_at_thresh,
    }

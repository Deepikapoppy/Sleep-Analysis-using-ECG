"""
=============================================================================
phase6_comparison_evaluation/p6_metrics.py
Step 2 — Compute evaluation metrics comparing the ECG-derived hypnogram
(Phase 4) against the PSG gold-standard hypnogram (Phase 5).

  _kappa_interpretation   numeric kappa → text label (Landis & Koch scale)
  compute_metrics         overall accuracy, Cohen's kappa, per-stage
                          precision/recall/F1, confusion matrix
=============================================================================
"""

import logging

from sklearn.metrics import (confusion_matrix, classification_report,
                              cohen_kappa_score, accuracy_score)

from phase4_sleep_classification.p4_plots import STAGE_NAMES


# ─────────────────────────────────────────────────────────────────────────────
#  Kappa interpretation
# ─────────────────────────────────────────────────────────────────────────────

def _kappa_interpretation(kappa: float) -> str:
    """Return a text label for a Cohen's kappa value (Landis & Koch scale)."""
    if kappa < 0:
        return "No agreement"
    elif kappa < 0.20:
        return "Slight"
    elif kappa < 0.40:
        return "Fair"
    elif kappa < 0.60:
        return "Moderate"
    elif kappa < 0.80:
        return "Substantial"
    else:
        return "Almost perfect"


# ─────────────────────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(ecg_hyp, psg_hyp, logger: logging.Logger):
    """
    Compute overall accuracy, Cohen's kappa (+ interpretation), per-stage
    precision/recall/F1, and the confusion matrix.

    Only epochs where the PSG label is valid (psg_hyp >= 0) are scored.

    Returns
    -------
    metrics, cm, y_true, y_pred, present_stages
    """
    valid_mask = (psg_hyp >= 0)
    y_true = psg_hyp[valid_mask]
    y_pred = ecg_hyp[valid_mask]

    acc   = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    present_stages = sorted(list(set(y_true) | set(y_pred)))
    stage_labels   = [STAGE_NAMES.get(s, str(s)) for s in present_stages]

    report = classification_report(y_true, y_pred, labels=present_stages,
                                   target_names=stage_labels, output_dict=True)
    cm     = confusion_matrix(y_true, y_pred, labels=present_stages)

    metrics = {
        "n_epochs"            : int(len(y_true)),
        "overall_acc"         : round(float(acc),   4),
        "cohen_kappa"         : round(float(kappa), 4),
        "kappa_interpretation": _kappa_interpretation(kappa),
        "per_stage"           : {},
        "present_stages"      : present_stages,
    }
    for s in present_stages:
        name = STAGE_NAMES.get(s, str(s))
        if name in report:
            metrics["per_stage"][name] = {
                "precision": round(report[name]["precision"], 4),
                "recall"   : round(report[name]["recall"],    4),
                "f1"       : round(report[name]["f1-score"],  4),
                "support"  : int(report[name]["support"]),
            }

    logger.info(f"Overall Accuracy : {acc*100:.2f}%")
    logger.info(f"Cohen's Kappa    : {kappa:.4f}  ({_kappa_interpretation(kappa)})")
    for name, v in metrics["per_stage"].items():
        logger.info(f"  {name:8s}  P={v['precision']:.3f}  R={v['recall']:.3f}  F1={v['f1']:.3f}")

    return metrics, cm, y_true, y_pred, present_stages

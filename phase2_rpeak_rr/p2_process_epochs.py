"""
=============================================================================
phase2_rpeak_rr/p2_process_epochs.py
Step 4 — Run calibrated cascade detection on every epoch.

Changes from original:
  - calibrate_subject() called ONCE before epoch loop
  - detect_rpeaks_all_methods() receives subject_config,
    neighbor_hr_bpm, hr_confidence per epoch
  - Rolling HR context tracked across epochs
=============================================================================
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional

from phase2_rpeak_rr.p2_rpeak_detection import (
    detect_rpeaks_all_methods,
    calibrate_subject,
)
from phase2_rpeak_rr.p2_rr_intervals import (
    compute_rr, correct_rr_artifacts, apply_patient_rr_filter
)


# ─────────────────────────────────────────────────────────────────────────────
#  HR confidence helper
# ─────────────────────────────────────────────────────────────────────────────

def _hr_confidence(row: dict) -> float:
    """
    Score how much to trust the HR from a completed epoch result.
    Returns 0.0 (don't trust) to 1.0 (fully trust).
    """
    if row.get("is_bad", False):
        return 0.0
    if row.get("n_rr", 0) < 5:
        return 0.0
    if row.get("method") == "scipy_prominence":
        return 0.3   # last resort method — low confidence
    hr = row.get("mean_hr", 0.0)
    if not (30 <= hr <= 120):
        return 0.1   # outside normal sleep HR range
    return float(row.get("quality", 0.0))


def _compute_neighbor_hr(recent_rows: list) -> Tuple[Optional[float], float]:
    """
    Confidence-weighted mean HR from last 2-3 valid epochs.
    Returns (neighbor_hr_bpm, hr_confidence).
    """
    if not recent_rows:
        return None, 0.0

    hrs, weights = [], []
    for row in recent_rows:
        conf = _hr_confidence(row)
        hr   = row.get("mean_hr", 0.0)
        if conf > 0.0 and hr and hr > 0:
            hrs.append(hr)
            weights.append(conf)

    if not hrs:
        return None, 0.0

    weights = np.array(weights)
    hrs     = np.array(hrs)
    neighbor_hr   = float(np.average(hrs, weights=weights))
    hr_confidence = float(np.mean(weights))
    return neighbor_hr, hr_confidence


# ─────────────────────────────────────────────────────────────────────────────
#  Main processing loop
# ─────────────────────────────────────────────────────────────────────────────

def process_all_epochs(epochs: np.ndarray,
                       bad_mask: np.ndarray,
                       fs: float,
                       config: dict,
                       logger: logging.Logger
                       ) -> Tuple[List[Dict], Dict]:
    """
    Calibrated cascade R-peak detection on every epoch.

    Stage 1: calibrate_subject() — runs ALL methods on first 4 clean
             epochs to pick primary method for THIS subject.

    Stage 2: per-epoch loop — calibrated cascade with dual gate
             (precision + recall) using rolling neighbour HR context.

    Returns
    -------
    results       : list of dicts — one per epoch
    method_counts : dict {method: n_epochs_won}
    """

    # ── Stage 1: Subject calibration ─────────────────────────────────────────
    logger.info("Running subject calibration (first 4 clean epochs)...")
    subject_config = calibrate_subject(epochs, bad_mask, fs, n_cal=4)
    logger.info(
        f"Calibration complete: "
        f"primary={subject_config['primary']} | "
        f"threshold={subject_config['subj_threshold']} | "
        f"baseline_hr={subject_config['baseline_hr']} | "
        f"nk_score={subject_config['cal_score_nk']} | "
        f"elgendi_score={subject_config['cal_score_elgendi']}"
    )

    # ── Stage 2: Per-epoch detection ──────────────────────────────────────────
    results       = []
    method_counts = {}
    recent_rows   = []   # rolling window of last 3 completed epoch rows

    logger.info(f"Processing {len(epochs)} epochs — calibrated cascade...")

    for i, epoch in enumerate(epochs):

        # Empty row template
        row = {
            "epoch_idx"          : i,
            "is_bad"             : bool(bad_mask[i]),
            "rr_ms"              : [],
            "n_peaks"            : 0,
            "n_rr"               : 0,
            "mean_rr"            : float("nan"),
            "mean_hr"            : float("nan"),
            "n_artifacts"        : 0,
            "method"             : "skipped",
            "quality"            : 0.0,
            "all_method_counts"  : {},
            "r_peak_samples_abs" : [],
            "methods_run"        : 0,
            "stop_reason"        : "skipped",
        }

        if bad_mask[i]:
            results.append(row)
            continue

        # Neighbour HR context from recent valid epochs
        neighbor_hr, hr_conf = _compute_neighbor_hr(recent_rows)

        # Run calibrated cascade
        det = detect_rpeaks_all_methods(
            epoch, fs,
            epoch_idx       = i,
            subject_config  = subject_config,
            neighbor_hr_bpm = neighbor_hr,
            hr_confidence   = hr_conf,
        )

        best_peaks  = det["best_peaks"]
        best_method = det["best_method"]

        method_counts[best_method] = method_counts.get(best_method, 0) + 1

        row["method"]            = best_method
        row["n_peaks"]           = len(best_peaks)
        row["all_method_counts"] = det["all_counts"]
        row["all_peaks"]         = {m: p.tolist() for m, p in det["all_peaks"].items()}  # ← ADD
        row["methods_run"]       = det["methods_run"]
        row["stop_reason"]       = det["stop_reason"]

        # Absolute R-peak sample positions for full-night overlay
        epoch_offset = i * int(fs * config["epoch_sec"])
        row["r_peak_samples_abs"] = (
            (best_peaks + epoch_offset).tolist()
            if len(best_peaks) > 0 else []
        )

        # RR computation + per-epoch artifact correction
        if len(best_peaks) >= 2:
            rr = compute_rr(best_peaks, fs)
            rr_clean, n_art = correct_rr_artifacts(rr)
            row["rr_ms"]       = rr_clean.tolist()
            row["n_rr"]        = len(rr_clean)
            row["n_artifacts"] = n_art
            if len(rr_clean) > 0:
                row["mean_rr"] = float(np.mean(rr_clean))
                row["mean_hr"] = float(60000.0 / row["mean_rr"])
                row["quality"] = float(len(rr_clean) / max(len(rr), 1))

        results.append(row)

        # Update rolling window — keep last 3 valid epochs only
        recent_rows.append(row)
        if len(recent_rows) > 3:
            recent_rows.pop(0)

        if (i + 1) % 50 == 0:
            logger.info(f"  Processed {i+1}/{len(epochs)} epochs...")

    # ── Log cascade depth distribution ────────────────────────────────────────
    depths = [r.get("methods_run", 0) for r in results if not r["is_bad"]]
    if depths:
        from collections import Counter
        dist = Counter(depths)
        logger.info(
            f"Cascade depth: "
            f"1 method={dist.get(1,0)} | "
            f"2 methods={dist.get(2,0)} | "
            f"3 methods={dist.get(3,0)} epochs"
        )

    logger.info(f"Method wins: {method_counts}")

    # ── Patient-level percentile RR filter ────────────────────────────────────
    apply_patient_rr_filter(results, logger)

    return results, method_counts
"""
=============================================================================
phase2_rpeak_rr/p2_rr_intervals.py
Step 3 — RR interval computation + two-tier artifact correction.

Tier 1: fixed physiological bounds (300–2400 ms) + local median deviation
Tier 2: patient-specific percentile bounds (5th–95th) computed full-night
=============================================================================
"""

import logging
import numpy as np
from typing import List, Tuple, Optional


def compute_rr(peaks: np.ndarray, fs: float) -> np.ndarray:
    """Convert R-peak sample indices → RR intervals in milliseconds."""
    if len(peaks) < 2:
        return np.array([])
    return np.diff(peaks) / fs * 1000.0


def correct_rr_artifacts(rr_ms: np.ndarray,
                          rr_min: float = 300.0,
                          rr_max: float = 2400.0,
                          dev_thresh: float = 0.25,
                          patient_p5: Optional[float] = None,
                          patient_p95: Optional[float] = None
                          ) -> Tuple[np.ndarray, int]:
    """
    Two-tier RR artifact correction.

    Tier 1 — physiological bounds + local median deviation.
    Tier 2 — patient-specific percentile bounds (optional).

    Returns
    -------
    rr_clean  : filtered RR array
    n_removed : number of intervals removed
    """
    if len(rr_ms) == 0:
        return rr_ms, 0

    eff_min = rr_min
    eff_max = rr_max
    if patient_p5 is not None and patient_p95 is not None:
        eff_min = max(eff_min, float(patient_p5))
        eff_max = min(eff_max, float(patient_p95))

    mask = (rr_ms >= eff_min) & (rr_ms <= eff_max)
    if mask.sum() > 3:
        local_med = np.median(rr_ms[mask])
        mask &= (np.abs(rr_ms - local_med) / local_med) < dev_thresh

    n_removed = int(len(rr_ms) - mask.sum())
    return rr_ms[mask], n_removed


def apply_patient_rr_filter(results: list,
                             logger: logging.Logger
                             ) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute full-night RR 5th/95th percentiles and re-filter all epochs.
    Mutates `results` in-place. Returns (p5, p95).
    """
    all_rr = []
    for row in results:
        all_rr.extend(row.get("rr_ms", []))

    if len(all_rr) < 30:
        logger.warning("Patient RR filter: too few intervals — skipping.")
        return None, None

    arr      = np.array(all_rr, dtype=float)
    plausible = arr[(arr >= 300) & (arr <= 2400)]
    if len(plausible) < 10:
        logger.warning("Patient RR filter: insufficient plausible RRs — skipping.")
        return None, None

    p5  = float(np.percentile(plausible, 5))
    p95 = float(np.percentile(plausible, 95))
    logger.info(
        f"Patient RR bounds: p5={p5:.1f} ms, p95={p95:.1f} ms  "
        f"({len(plausible)} valid intervals)"
    )

    total_removed = 0
    for row in results:
        rr_orig = np.array(row.get("rr_ms", []), dtype=float)
        if len(rr_orig) == 0:
            continue
        rr_filt, n_art = correct_rr_artifacts(
            rr_orig, patient_p5=p5, patient_p95=p95
        )
        extra             = len(rr_orig) - len(rr_filt)
        total_removed    += extra
        row["rr_ms"]      = rr_filt.tolist()
        row["n_rr"]       = len(rr_filt)
        row["n_artifacts"] = int(row.get("n_artifacts", 0)) + extra
        if len(rr_filt) > 0:
            row["mean_rr"] = float(np.mean(rr_filt))
            row["mean_hr"] = float(60000.0 / row["mean_rr"])
            row["quality"] = float(len(rr_filt) / max(len(rr_orig), 1))
        else:
            row["mean_rr"] = float("nan")
            row["mean_hr"] = float("nan")
            row["quality"] = 0.0

    logger.info(
        f"Patient RR filter: removed {total_removed} additional outliers."
    )
    return p5, p95


def build_tachogram(results: list,
                    epoch_sec: float = 30.0) -> Tuple[np.ndarray, np.ndarray]:
    """Build full-night tachogram arrays from per-epoch results."""
    rr_all, time_all = [], []
    for row in results:
        offset = row["epoch_idx"] * epoch_sec
        rr_ep  = np.array(row["rr_ms"])
        if len(rr_ep) == 0:
            continue
        beat_times = np.cumsum(rr_ep / 1000.0)
        beat_times = beat_times - beat_times[0] + offset
        rr_all.extend(rr_ep.tolist())
        time_all.extend(beat_times.tolist())
    return np.array(time_all), np.array(rr_all)

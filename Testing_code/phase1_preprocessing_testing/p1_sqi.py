"""
=============================================================================
phase1_preprocessing/p1_sqi.py
Step 6 — Signal Quality Index (SQI) per 30-sec epoch
Step 7 — Bad epoch detection

SQI metrics
───────────
kSQI   : Kurtosis SQI    — sharp QRS → high kurtosis → good signal
pSQI   : Power SQI       — QRS band (5–15 Hz) vs total power
basSQI : Baseline SQI    — residual drift
overall: weighted composite (0–1)

Thresholds
──────────
kSQI_norm > 0.25  (kSQI > 5 raw)
pSQI      > 0.30
basSQI    > 0.70
overall   > 0.50  ← primary decision gate

Weights: 0.40 × kSQI_norm + 0.40 × pSQI + 0.20 × basSQI
=============================================================================
"""

import logging
import numpy as np
import pandas as pd
from scipy.signal import medfilt, welch
from scipy.stats import kurtosis as scipy_kurtosis


# ─────────────────────────────────────────────────────────────────────────────
#  Single-epoch SQI
# ─────────────────────────────────────────────────────────────────────────────

def compute_epoch_sqi(epoch: np.ndarray, fs: float) -> dict:
    """
    Compute all SQI metrics for one 30-sec epoch.

    Returns dict with keys:
        kSQI, kSQI_norm, pSQI, basSQI, overall_sqi
    """
    # kSQI — kurtosis
    k         = float(scipy_kurtosis(epoch, fisher=True))
    kSQI_norm = float(np.clip(k / 20.0, 0, 1))

    # pSQI — QRS band power ratio
    nperseg  = min(len(epoch), int(fs * 4))
    f, psd   = welch(epoch, fs=fs, nperseg=nperseg)
    qrs_mask = (f >= 5) & (f <= 15)
    tot_mask = f > 0
    p_qrs    = float(np.trapezoid(psd[qrs_mask], f[qrs_mask])) \
               if qrs_mask.sum() > 0 else 0.0
    p_tot    = float(np.trapezoid(psd[tot_mask], f[tot_mask])) \
               if tot_mask.sum() > 0 else 1.0
    pSQI     = float(np.clip(p_qrs / (p_tot + 1e-10), 0, 1))

    # basSQI — baseline stability
    kern   = int(0.6 * fs)
    kern   = kern if kern % 2 == 1 else kern + 1
    base   = medfilt(epoch, kernel_size=kern)
    basSQI = float(np.clip(1.0 - np.std(base) / (np.std(epoch) + 1e-10), 0, 1))

    overall = 0.40 * kSQI_norm + 0.40 * pSQI + 0.20 * basSQI

    return {
        "kSQI"       : float(k),
        "kSQI_norm"  : kSQI_norm,
        "pSQI"       : pSQI,
        "basSQI"     : basSQI,
        "overall_sqi": float(overall),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  All-epochs SQI
# ─────────────────────────────────────────────────────────────────────────────

def compute_sqi_all_epochs(epochs: np.ndarray,
                            fs: float,
                            logger: logging.Logger) -> pd.DataFrame:
    """
    Compute SQI for every epoch.

    Returns pd.DataFrame with columns:
        epoch_idx, kSQI, kSQI_norm, pSQI, basSQI, overall_sqi
    """
    logger.info(f"Computing SQI for {len(epochs)} epochs...")
    rows = []
    for i, ep in enumerate(epochs):
        row = {"epoch_idx": i}
        row.update(compute_epoch_sqi(ep, fs))
        rows.append(row)
        if (i + 1) % 100 == 0:
            logger.info(f"  SQI progress: {i+1}/{len(epochs)}")

    df    = pd.DataFrame(rows)
    n_bad = int((df["overall_sqi"] < 0.50).sum())
    logger.info(
        f"SQI complete.  Low-quality epochs (<0.50): "
        f"{n_bad}/{len(epochs)}  ({100*n_bad/len(epochs):.1f}%)"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Bad epoch detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_bad_epochs(epochs: np.ndarray,
                      amp_thresh: float = 10.0,
                      flatline_thresh: float = 1e-4,
                      logger: logging.Logger = None) -> np.ndarray:
    """
    Flag over-amplitude and flat-line epochs.

    Parameters
    ----------
    epochs          : (n_epochs, samples_per_epoch) array
    amp_thresh      : max(|epoch|) > this → bad
    flatline_thresh : std(epoch)   < this → bad (flat line)
    logger          : optional

    Returns
    -------
    bad_mask : bool np.ndarray, shape (n_epochs,)
    """
    bad_mask = np.zeros(len(epochs), dtype=bool)
    for i, ep in enumerate(epochs):
        if np.max(np.abs(ep)) > amp_thresh:
            bad_mask[i] = True
        elif np.std(ep) < flatline_thresh:
            bad_mask[i] = True

    n_bad = int(bad_mask.sum())
    if logger:
        logger.info(
            f"Bad epoch detection: {n_bad}/{len(epochs)} "
            f"({100*n_bad/len(epochs):.1f}%) flagged "
            f"(amp_thresh={amp_thresh}, flatline_thresh={flatline_thresh})"
        )
    return bad_mask

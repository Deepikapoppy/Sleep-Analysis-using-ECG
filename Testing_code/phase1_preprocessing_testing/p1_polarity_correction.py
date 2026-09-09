"""
=============================================================================
phase1_preprocessing/p1_polarity_correction.py
Step 4 — Global Polarity Correction  (BUG-5 FIX)

Why this matters
----------------
Some ECG leads are recorded with inverted polarity (negative R-peaks).
This must be detected and corrected ONCE on the FULL cleaned signal,
BEFORE z-score normalisation and epoch segmentation.

If corrected per-epoch, different epochs can end up with opposite polarity —
causing catastrophic failures in R-peak detection (Phase 2) and any
neural network trained on upright QRS complexes.

Method
------
Compare |p01| vs |p99| of the full signal:
  • if |p01| > |p99|  → signal is inverted → negate it
  • otherwise         → polarity is correct → no change
=============================================================================
"""

import logging
import numpy as np


def detect_and_apply_global_polarity(ecg: np.ndarray,
                                      logger: logging.Logger = None):
    """
    Detect and correct ECG polarity on the full signal.

    Parameters
    ----------
    ecg    : 1-D DWT-cleaned ECG array
    logger : optional

    Returns
    -------
    ecg       : np.ndarray — (possibly negated) signal
    inverted  : bool       — True if signal was negated
    """
    p99 = float(np.percentile(ecg, 99))
    p01 = float(np.percentile(ecg, 1))

    inverted = bool(np.abs(p01) > np.abs(p99))

    if inverted:
        ecg = -ecg
        if logger:
            logger.info(
                f"Polarity correction: ECG INVERTED  "
                f"(|p01|={abs(p01):.4f} > |p99|={p99:.4f})  — signal negated."
            )
    else:
        if logger:
            logger.info(
                f"Polarity OK  "
                f"(|p99|={p99:.4f} ≥ |p01|={abs(p01):.4f})  — no change."
            )

    return ecg, inverted

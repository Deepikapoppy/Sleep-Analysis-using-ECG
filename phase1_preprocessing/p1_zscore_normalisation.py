"""
=============================================================================
phase1_preprocessing/p1_zscore_normalisation.py
Step 5 — Z-score Normalisation

Formula:  ecg_norm = (ecg − μ) / (σ + ε)

Applied to the FULL polarity-corrected signal before epoch segmentation.
ε = 1e-8 prevents division by zero for flat-line edge cases.
=============================================================================
"""

import logging
import numpy as np


def normalize(ecg: np.ndarray,
              logger: logging.Logger = None) -> np.ndarray:
    """
    Z-score normalise the full ECG signal.

    Parameters
    ----------
    ecg    : 1-D polarity-corrected ECG array
    logger : optional

    Returns
    -------
    ecg_norm : np.ndarray — zero-mean, unit-variance ECG
    """
    mu    = float(np.mean(ecg))
    sigma = float(np.std(ecg))

    ecg_norm = (ecg - mu) / (sigma + 1e-8)

    if logger:
        logger.info(
            f"Z-score normalisation: μ={mu:.4f}, σ={sigma:.4f}  "
            f"→ output range [{ecg_norm.min():.2f}, {ecg_norm.max():.2f}]"
        )

    return ecg_norm

"""
=============================================================================
phase1_preprocessing/p1_baseline_wander.py
Step 3a — Baseline Wander Removal via DWT Approximation Zeroing.

Method
------
After multilevel DWT decomposition, the approximation coefficients (cA_level)
capture the very-low-frequency content (0 – ~2 Hz at 125 Hz / level 5).
Setting cA_level to zero completely removes the DC drift and slow baseline
wander caused by respiration or electrode movement — without distorting the
QRS complex.

Sub-band @ 125 Hz, level 5:
    A5 : 0.00 – 1.95 Hz   ← SET TO ZERO here
=============================================================================
"""

import logging
import numpy as np
import pywt


def remove_baseline_wander(coeffs: list,
                            level: int,
                            logger: logging.Logger = None) -> list:
    """
    Zero the approximation coefficients (coeffs[0] = cA_level) in-place.

    Parameters
    ----------
    coeffs  : list of arrays from pywt.wavedec()
              coeffs[0]     = cA_level  (lowest freq — baseline)
              coeffs[1]     = cD_level
              ...
              coeffs[level] = cD1       (highest freq)
    level   : DWT decomposition level
    logger  : optional

    Returns
    -------
    coeffs : same list with coeffs[0] zeroed
    """
    before_energy = float(np.sum(coeffs[0] ** 2))
    coeffs[0]     = np.zeros_like(coeffs[0])

    if logger:
        logger.info(
            f"Baseline wander removal: cA{level} zeroed  "
            f"(removed energy = {before_energy:.4f})"
        )
    return coeffs


def get_baseline_band_hz(fs: float, level: int):
    """
    Return (low_hz, high_hz) of the approximation band for reference.
    e.g. at 125 Hz, level 5 → (0.0, 1.95)
    """
    high = fs / (2 ** (level + 1))
    return 0.0, round(high, 2)

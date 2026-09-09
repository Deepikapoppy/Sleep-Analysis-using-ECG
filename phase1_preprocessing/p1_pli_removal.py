"""
=============================================================================
phase1_preprocessing/p1_pli_removal.py
Step 3b — Power Line Interference (PLI) Removal via D1 Zeroing.

Method
------
At 125 Hz with level-5 DWT, the D1 detail band covers 31.25 – 62.5 Hz.
The 50 Hz power-line interference falls entirely within this band.
Setting cD1 (coeffs[level]) to zero suppresses PLI without affecting the
QRS complex (which peaks around 8–15 Hz, in the D3 band).

Sub-band @ 125 Hz, level 5:
    D1 : 31.25 – 62.50 Hz  ← SET TO ZERO here (contains 50 Hz PLI)
=============================================================================
"""

import logging
import numpy as np


def remove_pli(coeffs: list,
               level: int,
               logger: logging.Logger = None) -> list:
    """
    Zero the D1 detail coefficients (coeffs[level] = cD1) in-place.

    Parameters
    ----------
    coeffs  : list of arrays from pywt.wavedec()
    level   : DWT decomposition level
              coeffs[level] == cD1 (highest frequency detail)
    logger  : optional

    Returns
    -------
    coeffs : same list with coeffs[level] zeroed
    """
    pli_idx       = level          # coeffs[level] = cD1
    before_energy = float(np.sum(coeffs[pli_idx] ** 2))
    coeffs[pli_idx] = np.zeros_like(coeffs[pli_idx])

    if logger:
        logger.info(
            f"PLI removal: cD1 (index {pli_idx}) zeroed  "
            f"(removed energy = {before_energy:.4f})"
        )
    return coeffs


def get_pli_band_hz(fs: float) -> tuple:
    """
    Return (low_hz, high_hz) of the D1 band (PLI band) for reference.
    e.g. at 125 Hz → (31.25, 62.5)
    """
    return round(fs / 4.0, 2), round(fs / 2.0, 2)

"""
=============================================================================
phase1_preprocessing/p1_downsample.py
Step 2 — Polyphase downsampling to target_fs (default 125 Hz).

Uses scipy.signal.resample_poly which applies an anti-aliasing FIR filter
automatically before decimation — no aliasing artefacts.
=============================================================================
"""

import logging
import numpy as np
from math import gcd
from scipy.signal import resample_poly


def downsample(ecg: np.ndarray,
               fs_orig: float,
               fs_target: float,
               logger: logging.Logger = None):
    """
    Polyphase downsample ECG from fs_orig → fs_target.

    Parameters
    ----------
    ecg       : 1-D raw ECG numpy array
    fs_orig   : original sampling frequency (Hz)
    fs_target : target sampling frequency (Hz)  — typically 125
    logger    : optional

    Returns
    -------
    ecg_ds : np.ndarray  — downsampled signal
    fs     : float       — fs_target (for downstream convenience)
    """
    if fs_orig == fs_target:
        if logger:
            logger.info("Downsample: already at target Fs — no resampling needed.")
        return ecg, float(fs_target)

    g    = gcd(int(fs_orig), int(fs_target))
    up   = int(fs_target) // g
    down = int(fs_orig)   // g

    if logger:
        logger.info(
            f"Downsample: {fs_orig} Hz → {fs_target} Hz  "
            f"(up={up}, down={down})"
        )

    ecg_ds = resample_poly(ecg, up, down)

    if logger:
        logger.info(
            f"  Downsampled length : {len(ecg_ds)} samples  "
            f"({len(ecg_ds) / fs_target / 3600:.2f} hrs)"
        )

    return ecg_ds, float(fs_target)

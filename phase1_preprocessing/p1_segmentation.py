"""
=============================================================================
phase1_preprocessing/p1_segmentation.py
Step 8 — Segment normalised ECG into 30-second non-overlapping epochs.
=============================================================================
"""

import logging
import numpy as np


def segment_epochs(ecg: np.ndarray,
                   fs: float,
                   epoch_sec: float = 30,
                   logger: logging.Logger = None):
    """
    Split ECG into fixed-length non-overlapping epochs.

    Parameters
    ----------
    ecg       : 1-D normalised ECG array
    fs        : sampling frequency (Hz)
    epoch_sec : epoch length in seconds (default 30)
    logger    : optional

    Returns
    -------
    epochs : np.ndarray  shape (n_epochs, samples_per_epoch)
    spe    : int         samples per epoch
    """
    spe    = int(epoch_sec * fs)
    n_ep   = len(ecg) // spe
    epochs = np.array([ecg[i * spe : (i + 1) * spe] for i in range(n_ep)])

    if logger:
        logger.info(
            f"Segmentation: {n_ep} epochs × {spe} samples "
            f"({epoch_sec} sec @ {fs} Hz)"
        )
    return epochs, spe

"""
=============================================================================
phase1_preprocessing/p1_dwt_denoise.py
Step 3 — DWT Denoising orchestrator.

Calls the three sub-steps in the correct order:
    3a. p1_baseline_wander  → zero cA_level
    3b. p1_pli_removal      → zero cD1
    3c. p1_soft_threshold   → soft-threshold D2…D_level  +  IDWT

Sub-band map @ 125 Hz, level 5
─────────────────────────────
    D1 : 31.25 – 62.50 Hz   ← ZEROED   (50 Hz PLI)
    D2 : 15.63 – 31.25 Hz   ← soft-threshold
    D3 :  7.81 – 15.63 Hz   ← soft-threshold  (QRS peak band)
    D4 :  3.91 –  7.81 Hz   ← soft-threshold
    D5 :  1.95 –  3.91 Hz   ← soft-threshold
    A5 :  0.00 –  1.95 Hz   ← ZEROED   (baseline wander)
=============================================================================
"""

import logging
import numpy as np
import pywt

from phase1_preprocessing.p1_baseline_wander import remove_baseline_wander
from phase1_preprocessing.p1_pli_removal     import remove_pli
from phase1_preprocessing.p1_soft_threshold  import soft_threshold_d2_d5, reconstruct


def _dwt_subbands(fs: float, level: int) -> list:
    """Return [(label, low_hz, high_hz), ...] for every DWT sub-band."""
    bands = []
    for lv in range(1, level + 1):
        hi = fs / (2 ** lv)
        lo = fs / (2 ** (lv + 1))
        bands.append((f"D{lv}", round(lo, 2), round(hi, 2)))
    bands.append((f"A{level}", 0.0, round(fs / (2 ** (level + 1)), 2)))
    return bands


def dwt_filter(ecg: np.ndarray,
               fs: float,
               wavelet: str = "db4",
               level: int = 5,
               logger: logging.Logger = None):
    """
    Full DWT denoising pipeline:
        Decompose → Baseline removal → PLI removal → Soft threshold → IDWT

    Parameters
    ----------
    ecg     : 1-D downsampled ECG (125 Hz)
    fs      : sampling frequency
    wavelet : 'db4' | 'db6' | 'sym4'
    level   : decomposition level (5 recommended at 125 Hz)
    logger  : optional

    Returns
    -------
    ecg_clean   : np.ndarray — denoised ECG (same length as input)
    coeffs_raw  : list       — original coefficients (for plotting)
    coeffs_proc : list       — processed coefficients (for plotting)
    """
    # ── Decompose ─────────────────────────────────────────────────────────────
    coeffs     = pywt.wavedec(ecg, wavelet=wavelet, level=level)
    coeffs_raw = [c.copy() for c in coeffs]   # save originals for plotting

    # Log sub-band table
    if logger:
        logger.info(f"DWT decomposition: wavelet={wavelet}, level={level}")
        for name, lo, hi in reversed(_dwt_subbands(fs, level)):
            if name == f"A{level}":
                tag = "  ← ZEROED   (baseline wander)"
            elif name == "D1":
                tag = "  ← ZEROED   (50 Hz PLI)"
            else:
                tag = "  ← soft-threshold"
            logger.info(f"  {name:4s}: {lo:6.2f} – {hi:6.2f} Hz{tag}")

    # ── Step 3a: Baseline wander removal ─────────────────────────────────────
    coeffs = remove_baseline_wander(coeffs, level, logger)

    # ── Step 3b: PLI removal ─────────────────────────────────────────────────
    coeffs = remove_pli(coeffs, level, logger)

    # ── Step 3c: D2–D5 soft thresholding + IDWT ──────────────────────────────
    coeffs, thresh, sigma = soft_threshold_d2_d5(
        coeffs, level, len(ecg), logger
    )
    ecg_clean = reconstruct(coeffs, wavelet, len(ecg), logger)

    coeffs_proc = coeffs

    if logger:
        snr_db = 10 * np.log10(
            (np.var(ecg) + 1e-10) / (np.var(ecg - ecg_clean) + 1e-10)
        )
        logger.info(
            f"DWT denoising complete — SNR improvement ≈ {snr_db:.1f} dB  "
            f"(σ={sigma:.6f}, λ={thresh:.6f})"
        )

    return ecg_clean, coeffs_raw, coeffs_proc

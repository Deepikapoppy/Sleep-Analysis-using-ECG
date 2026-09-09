"""
=============================================================================
phase1_preprocessing/p1_soft_threshold.py
Step 3c — D2–D5 Soft Thresholding (residual noise removal).

Method
------
After zeroing cA (baseline) and cD1 (PLI), the remaining detail bands
D2 … D_level still contain residual high-frequency noise.
Universal soft thresholding (Donoho & Johnstone, 1994) is applied:

    σ  = median(|cD2|) / 0.6745          ← robust noise estimate from D2
    λ  = σ × √(2 × ln(N))               ← universal threshold
    cDi_proc = sign(cDi) × max(|cDi| − λ, 0)   for i = 2 … level

This preserves QRS morphology while attenuating Gaussian noise.

Sub-bands affected @ 125 Hz, level 5:
    D2 : 15.63 – 31.25 Hz   ← soft-threshold
    D3 :  7.81 – 15.63 Hz   ← soft-threshold (QRS peak energy — gentle)
    D4 :  3.91 –  7.81 Hz   ← soft-threshold
    D5 :  1.95 –  3.91 Hz   ← soft-threshold
=============================================================================
"""

import logging
import numpy as np
import pywt


def soft_threshold_d2_d5(coeffs: list,
                          level: int,
                          ecg_length: int,
                          logger: logging.Logger = None) -> tuple:
    """
    Apply universal soft thresholding to detail bands D2 … D_level.

    Parameters
    ----------
    coeffs     : list from pywt.wavedec() — cA already zeroed, cD1 already zeroed
    level      : DWT decomposition level
    ecg_length : length of original ECG signal (used for threshold calculation)
    logger     : optional

    Returns
    -------
    coeffs : modified list (D2…D_level soft-thresholded)
    thresh : float — the threshold value used (for logging / plotting)
    sigma  : float — estimated noise sigma
    """
    # Noise estimate from D2 (most reliable detail band after PLI removal)
    d2_idx = level - 1          # coeffs[level-1] = cD2
    sigma  = np.median(np.abs(coeffs[d2_idx])) / 0.6745

    # Universal threshold (Donoho & Johnstone)
    thresh = sigma * np.sqrt(2.0 * np.log(max(ecg_length, 1)))

    # Apply soft threshold to D2 … D_level
    # Indices 1 … level-1 correspond to cD_level … cD2
    for i in range(1, level):
        coeffs[i] = pywt.threshold(coeffs[i], value=thresh, mode="soft")

    if logger:
        logger.info(
            f"Soft thresholding D2–D{level}: "
            f"σ={sigma:.6f},  λ={thresh:.6f}"
        )

    return coeffs, float(thresh), float(sigma)


def reconstruct(coeffs: list, wavelet: str, original_length: int,
                logger: logging.Logger = None) -> np.ndarray:
    """
    Inverse DWT reconstruction after all three denoising steps.

    Parameters
    ----------
    coeffs          : processed coefficients list
    wavelet         : wavelet string ('db4', 'db6', 'sym4')
    original_length : length of original ECG (to trim IDWT padding)
    logger          : optional

    Returns
    -------
    ecg_clean : np.ndarray — reconstructed, denoised ECG
    """
    ecg_clean = pywt.waverec(coeffs, wavelet=wavelet)
    ecg_clean = ecg_clean[:original_length]   # trim padding from DWT

    if logger:
        snr_note = "IDWT reconstruction complete"
        logger.info(snr_note)

    return ecg_clean

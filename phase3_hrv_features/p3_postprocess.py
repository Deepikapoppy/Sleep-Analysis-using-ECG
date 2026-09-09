"""
=============================================================================
phase3_hrv_features/p3_postprocess.py
Post-epoch processing:
  1. Derived features (hr_epoch_delta, rmssd_mean_rr_ratio, sdnn_rmssd_ratio)
  2. 3-epoch rolling median on spectral features (RC5 fix)
  3. 10-epoch rolling ULF ratio
  4. Rolling z-score normalisation (FIX-5a)
  5. VLF set to NaN (meaningless in 30-sec epochs)
=============================================================================
"""

import logging
import numpy as np
import pandas as pd
from typing import List


# ─────────────────────────────────────────────────────────────────────────────
#  Spectral smoothing
# ─────────────────────────────────────────────────────────────────────────────

_SPECTRAL_SMOOTH_COLS = [
    "lf_hf_ratio", "lf_power", "hf_power", "lf_nu", "hf_nu",
    "edr_breath_rate", "edr_regularity",
]


def apply_spectral_rolling_median(df: pd.DataFrame,
                                   logger: logging.Logger,
                                   window: int = 3) -> pd.DataFrame:
    """
    3-epoch rolling median on spectral features (RC5 fix).
    Smooths epoch-to-epoch spectral instability without introducing
    interpolation artefacts.
    """
    for col in _SPECTRAL_SMOOTH_COLS:
        if col in df.columns:
            df[col] = (
                df[col]
                .rolling(window=window, center=True, min_periods=1)
                .median()
            )
    logger.info(
        f"RC5 FIX: 3-epoch rolling median on: "
        + ", ".join([c for c in _SPECTRAL_SMOOTH_COLS if c in df.columns])
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  VLF — set to NaN (meaningless in 30-second epochs)
# ─────────────────────────────────────────────────────────────────────────────

def null_vlf(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """VLF power is not meaningful for 30-second epochs. Set to NaN."""
    if "vlf_power" in df.columns:
        df["vlf_power"] = np.nan
        logger.info("VLF power set to NaN — not meaningful in 30-s epochs.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Derived cross-epoch features
# ─────────────────────────────────────────────────────────────────────────────

def compute_derived_features(df: pd.DataFrame,
                               logger: logging.Logger) -> pd.DataFrame:
    """
    Compute features that require looking across epochs:
      hr_epoch_delta      : |HR[i] - HR[i-1]| — inter-epoch HR transition
      ulf_ratio           : 10-epoch rolling ULF proxy from tachogram
    """
    # hr_epoch_delta
    df["hr_epoch_delta"] = df["mean_hr"].diff().abs().fillna(0.0)
    logger.info("Computed hr_epoch_delta (inter-epoch HR change).")

    # ulf_ratio: 10-epoch rolling mean of hf_power as ULF proxy
    # True ULF needs > 10 min; here we use rolling hf as a practical substitute
    if "hf_power" in df.columns:
        df["ulf_ratio"] = (
            df["hf_power"]
            .rolling(window=10, center=True, min_periods=3)
            .mean()
            .fillna(np.nan)
        )
        logger.info("Computed ulf_ratio (10-epoch rolling HF proxy).")

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Z-score normalisation
# ─────────────────────────────────────────────────────────────────────────────

_ZSCORE_FEATURES = [
    # T1
    "mean_hr", "mean_rr", "rmssd", "sdnn", "pnn20", "hr_range",
    "rr_autocorr_lag1", "hf_power", "lf_hf_ratio", "lf_power",
    "resp_rate_est", "perm_en", "dfa_alpha1", "sd1", "sd1_sd2_ratio",
    "sampen", "rmssd_mean_rr_ratio", "sdnn_rmssd_ratio",
    # T2
    "median_rr", "pnn50", "cv", "rr_iqr", "hf_nu", "lf_nu",
    "peak_hf_freq", "total_power", "sd2", "dfa_alpha2",
    "prsa_dc", "prsa_ac", "porta_asymmetry",
    "edr_breath_rate", "edr_regularity", "hr_slope_epoch",
    # T3
    "guzik_asymmetry", "mse_scale2", "mse_scale4", "higuchi_fd",
    "fuzzy_entropy", "recurrence_rate", "rr_entropy_rate",
    "edr_r_amplitude", "edr_qrs_area", "edr_breath_depth",
    "crc_cross_spectrum", "crc_coherence_hf", "crc_phase_sync",
    "r_amplitude_cv", "r_amplitude_mean",
    "hf_peaks_count", "rr_spectral_entropy", "coherence_lf_hf",
    "rr_triangular_index",
    # Derived
    "hr_epoch_delta",
]

_ZSCORE_ROLLING_EPOCHS = 60   # ±30 min window


def add_zscore_features(df: pd.DataFrame,
                         logger: logging.Logger,
                         rolling: bool = True) -> pd.DataFrame:
    """
    For each feature in _ZSCORE_FEATURES add a <feat>_zscore column.

    FIX-5a: Rolling centred window (default 60 epochs = ±30 min) tracks
    ultradian HRV drift so inter-stage separation is preserved across
    the full night rather than collapsed by a global mean.

    σ < 1e-8 → z forced to 0.0
    """
    df_out = df.copy()
    added  = []
    w      = _ZSCORE_ROLLING_EPOCHS

    for feat in _ZSCORE_FEATURES:
        if feat not in df_out.columns:
            continue
        col   = pd.to_numeric(df_out[feat], errors="coerce")
        z_col = f"{feat}_zscore"

        if rolling:
            roll_mu  = col.rolling(window=w, center=True,
                                   min_periods=max(10, w // 4)).mean()
            roll_sig = col.rolling(window=w, center=True,
                                   min_periods=max(10, w // 4)).std(ddof=1)
            global_mu  = float(col.mean(skipna=True))
            global_sig = float(col.std(skipna=True, ddof=1))
            roll_mu    = roll_mu.fillna(global_mu)
            roll_sig   = roll_sig.fillna(global_sig)
            valid_sig  = roll_sig.where(
                np.isfinite(roll_sig) & (roll_sig > 1e-8), other=1.0
            )
            df_out[z_col] = (col - roll_mu) / valid_sig
            df_out[z_col] = df_out[z_col].fillna(0.0)
        else:
            mu  = float(col.mean(skipna=True))
            sig = float(col.std(skipna=True, ddof=1))
            df_out[z_col] = ((col - mu) / sig
                             if (np.isfinite(sig) and sig > 1e-8) else 0.0)
            df_out[z_col] = df_out[z_col].fillna(0.0)

        added.append(z_col)

    mode_str = f"rolling w={w}" if rolling else "global"
    logger.info(
        f"Z-score features added ({len(added)}, mode={mode_str}): "
        + ", ".join(added[:8]) + ("..." if len(added) > 8 else "")
    )
    return df_out

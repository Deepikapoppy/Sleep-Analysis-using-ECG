"""
=============================================================================
phase3_hrv_features/p3_sqi_gate.py
SQI Gate — 4 features, applied BEFORE any tier computation.

Features
────────
  kSQI          Kurtosis-based SQI  (from phase1_sqi.csv)
  pSQI          Power spectral SQI  (from phase1_sqi.csv)
  basSQI        Baseline wander SQI (from phase1_sqi.csv)
  overall_sqi   Combined gate score — <0.50 → epoch flagged bad

If phase1_sqi.csv is missing, all four columns are set to NaN and
the gate is not applied (epochs pass through for downstream tiers).
=============================================================================
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional


SQI_THRESHOLD  = 0.50   # epochs with overall_sqi < this are flagged
SQI_COLS_INPUT = ["kSQI", "kSQI_norm", "pSQI", "basSQI", "overall_sqi"]
SQI_GATE_COLS  = ["kSQI", "pSQI", "basSQI", "overall_sqi"]   # the 4 published features


def apply_sqi_gate(df: pd.DataFrame,
                   sqi_df: Optional[pd.DataFrame],
                   logger: logging.Logger) -> pd.DataFrame:
    """
    Merge Phase 1 SQI scores into feature DataFrame and flag bad SQI epochs.

    Adds columns:
        kSQI, pSQI, basSQI, overall_sqi   — 4 SQI Gate features
        sqi_flag                           — True if overall_sqi < threshold

    Parameters
    ----------
    df      : DataFrame with epoch_idx column (built by calling code)
    sqi_df  : DataFrame from load_phase1_sqi(), or None
    logger  : Logger instance

    Returns
    -------
    df : updated DataFrame with SQI columns added
    """
    if sqi_df is not None:
        avail_cols = ["epoch_idx"] + [c for c in SQI_COLS_INPUT if c in sqi_df.columns]
        df = df.merge(sqi_df[avail_cols], on="epoch_idx", how="left")
        logger.info(
            f"SQI Gate: merged columns {[c for c in avail_cols if c != 'epoch_idx']}"
        )
    else:
        for col in SQI_GATE_COLS:
            df[col] = np.nan
        logger.info("SQI Gate: all SQI columns set to NaN (no phase1_sqi.csv).")

    # ── overall_sqi gate flag ────────────────────────────────────────────────
    if "overall_sqi" in df.columns and not df["overall_sqi"].isna().all():
        df["sqi_flag"] = df["overall_sqi"].fillna(1.0) < SQI_THRESHOLD
        n_flagged = int(df["sqi_flag"].sum())
        logger.info(
            f"SQI Gate: {n_flagged}/{len(df)} epochs flagged "
            f"(overall_sqi < {SQI_THRESHOLD})"
        )
    else:
        df["sqi_flag"] = False
        logger.info("SQI Gate: flag column set to False (no overall_sqi available).")

    return df


def null_spectral_on_bad_sqi(df: pd.DataFrame,
                              spectral_cols: list,
                              logger: logging.Logger) -> pd.DataFrame:
    """
    Per the SQI Gate spec: set all spectral features to NaN for epochs
    where sqi_flag=True. Call this AFTER all tier features are computed.

    Parameters
    ----------
    df            : full feature DataFrame with sqi_flag column
    spectral_cols : list of spectral feature column names to null out
    """
    if "sqi_flag" not in df.columns:
        return df
    bad_mask = df["sqi_flag"].fillna(False).astype(bool)
    n_bad    = int(bad_mask.sum())
    if n_bad == 0:
        return df

    existing = [c for c in spectral_cols if c in df.columns]
    df.loc[bad_mask, existing] = np.nan
    logger.info(
        f"SQI Gate: nulled {len(existing)} spectral features "
        f"for {n_bad} bad-SQI epochs."
    )
    return df


# ── List of spectral columns to null when SQI is bad ─────────────────────────
# Includes Lomb-Scargle bands + all derived spectral / EDR features
SPECTRAL_FEATURE_COLS = [
    "vlf_power", "lf_power", "hf_power", "lf_hf_ratio",
    "total_power", "lf_nu", "hf_nu", "peak_hf_freq", "resp_rate_est",
    # T2 EDR features
    "edr_breath_rate", "edr_regularity", "edr_baseline_wander",
    # T3 CRC features
    "crc_cross_spectrum", "crc_coherence_hf", "crc_phase_sync",
    # T3 freq features
    "hf_peaks_count", "rr_spectral_entropy", "coherence_lf_hf",
    # VLF / ULF — T3
    "ulf_ratio",
]

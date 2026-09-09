"""
=============================================================================
phase4_sleep_classification/p4_preprocess.py
Step 2 — Pre-classification preprocessing.

Functions
─────────
  clip_outliers        99th-pct cap on sd1/sd2/sd1_sd2 to remove extreme spikes
  normalise_features   Robust 5–95 pct normalisation → [0, 1] per feature.
                       Also computes subject-adaptive Wake HR anchors (FIX-5b)
                       stored as __wake_hr_p75 / __sleep_hr_p25 columns.
=============================================================================
"""

import logging
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
#  Outlier clipping
# ─────────────────────────────────────────────────────────────────────────────

def clip_outliers(df: pd.DataFrame,
                  logger: logging.Logger,
                  cols=("sd1", "sd2", "sd1_sd2"),
                  percentile: int = 99) -> pd.DataFrame:
    """
    Cap extreme values at the given percentile for the specified columns.
    Only modifies values that actually exceed the cap.
    """
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        series    = pd.to_numeric(df[col], errors="coerce")
        cap       = np.nanpercentile(series.dropna().values, percentile)
        n_clipped = int((series > cap).sum())
        df[col]   = series.clip(upper=cap)
        if n_clipped:
            logger.info(
                f"clip_outliers: {col} — clipped {n_clipped} epochs "
                f"above {cap:.4f}"
            )
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Robust normalisation
# ─────────────────────────────────────────────────────────────────────────────

def normalise_features(df: pd.DataFrame,
                        logger: logging.Logger) -> pd.DataFrame:
    """
    Robust (5–95 pct) normalisation to [0, 1] for all non-metadata columns.

    FIX-5b: Also captures subject-adaptive Wake calibration anchors BEFORE
    normalisation and stores them as helper columns:
        __wake_hr_p75   : 75th pct of raw mean_hr (BPM units)
        __sleep_hr_p25  : 25th pct of raw mean_hr (BPM units)

    These are consumed by classify_epochs to dynamically calibrate Wake
    scoring for subjects with atypically low baseline HR.

    Columns that are NOT normalised:
        epoch, label, epoch_idx, is_bad, any *_zscore columns
    """
    logger.info(
        "Applying robust (5–95 pct) normalisation to enhance feature contrast."
    )
    cols = [c for c in df.columns
            if c not in ("epoch", "label", "epoch_idx", "is_bad")
            and not c.endswith("_zscore")]
    df_norm = df.copy()

    # FIX-5b: capture raw HR anchors BEFORE normalisation
    if "mean_hr" in df.columns:
        raw_hr = pd.to_numeric(df["mean_hr"], errors="coerce").dropna()
        if len(raw_hr) >= 10:
            df_norm["__wake_hr_p75"]  = float(np.nanpercentile(raw_hr, 75))
            df_norm["__sleep_hr_p25"] = float(np.nanpercentile(raw_hr, 25))
            logger.info(
                f"FIX-5b adaptive Wake anchors: "
                f"wake_hr_p75={df_norm['__wake_hr_p75'].iloc[0]:.1f} bpm  "
                f"sleep_hr_p25={df_norm['__sleep_hr_p25'].iloc[0]:.1f} bpm"
            )
        else:
            df_norm["__wake_hr_p75"]  = np.nan
            df_norm["__sleep_hr_p25"] = np.nan
    else:
        df_norm["__wake_hr_p75"]  = np.nan
        df_norm["__sleep_hr_p25"] = np.nan

    for col in cols:
        col_vals = pd.to_numeric(df_norm[col], errors="coerce")
        non_na   = col_vals.dropna()
        if non_na.size == 0:
            df_norm[col] = 0.5
            continue
        try:
            arr  = non_na.astype(float).values
            low  = float(np.nanpercentile(arr, 5))
            high = float(np.nanpercentile(arr, 95))
        except Exception:
            df_norm[col] = 0.5
            continue

        if (high - low) > 0:
            normed     = (col_vals.astype(float) - low) / (high - low)
            df_norm[col] = normed.clip(0, 1).fillna(0.5)
        else:
            df_norm[col] = 0.5

    return df_norm

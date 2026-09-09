"""
=============================================================================
phase4_sleep_classification/p4_load_phase3.py
Step 1 — Load Phase 3 HRV feature CSV and meta JSON.

Files loaded from <output_dir>/:
    phase3_hrv_features.csv    — 71 features + z-scores per epoch
    phase3_meta.json           — feature list, DWT provenance, tier counts
=============================================================================
"""

import os
import json
import logging
import pandas as pd


def load_features(config: dict, logger: logging.Logger) -> pd.DataFrame:
    """
    Load phase3_hrv_features.csv produced by Phase 3.
    Raises FileNotFoundError if the file is absent — Phase 3 must run first.
    """
    path = os.path.join(config["output_dir"], "phase3_hrv_features.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Phase 3 features not found: {path}\n"
            "Run p3_pipeline.py for this record before Phase 4."
        )
    df = pd.read_csv(path)
    logger.info(f"Loaded HRV features: {df.shape}  ({path})")
    return df


def load_phase3_meta(config: dict, logger: logging.Logger) -> dict:
    """
    Load phase3_meta.json written by Phase 3.
    Logs DWT provenance and v2 feature availability for traceability.
    Returns the meta dict (or {} if absent).
    """
    path = os.path.join(config["output_dir"], "phase3_meta.json")
    if not os.path.exists(path):
        logger.warning("phase3_meta.json not found — DWT provenance unavailable.")
        return {}

    with open(path) as f:
        meta = json.load(f)

    feats  = meta.get("feature_columns", [])
    v2_new = [f for f in ("pnn20", "hr_range", "resp_rate_est", "perm_en",
                           "rmssd_mean_rr_ratio", "median_rr", "rr_iqr")
              if f in feats]

    logger.info("Phase 3 metadata loaded (Phase 4 traceability):")
    logger.info(f"  Phase1 DWT wavelet  : {meta.get('phase1_dwt_wavelet', 'db4')}")
    logger.info(f"  Phase1 DWT level    : {meta.get('phase1_dwt_level', 5)}")
    logger.info(f"  Phase1 SQI mean     : {meta.get('phase1_mean_sqi')}")
    logger.info(f"  n_features          : {meta.get('n_feature_cols')}")
    logger.info(f"  v2 features present : {v2_new}")
    return meta

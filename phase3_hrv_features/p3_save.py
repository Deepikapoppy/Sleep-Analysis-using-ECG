"""
=============================================================================
phase3_hrv_features/p3_save.py
Save Phase 3 outputs and update SQLite.

Files written to <output_dir>/:
    phase3_hrv_features.csv    — 71 features + z-scores per epoch
    phase3_meta.json           — feature list, DWT provenance, tier counts
SQLite: processing_status.phase3_done = 1
=============================================================================
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Database.db_manager import get_record, mark_phase_done, mark_phase_failed, DB_PATH


def save_phase3_results(df: pd.DataFrame,
                         config: dict,
                         logger: logging.Logger,
                         phase1_meta: Optional[dict] = None,
                         db_path: str = DB_PATH) -> dict:
    """
    Write CSV + meta JSON and update SQLite phase3_done=1.
    Returns metrics dict.
    """
    out = config["output_dir"]
    os.makedirs(out, exist_ok=True)

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = os.path.join(out, "phase3_hrv_features.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Features saved → {csv_path}  shape={df.shape}")

    # ── Tier feature counts ───────────────────────────────────────────────────
    from phase3_hrv_features.p3_sqi_gate   import SQI_GATE_COLS
    from phase3_hrv_features.p3_postprocess import _ZSCORE_FEATURES

    sqi_cols = [c for c in SQI_GATE_COLS if c in df.columns]
    t1_cols  = [c for c in _T1_COLS  if c in df.columns]
    t2_cols  = [c for c in _T2_COLS  if c in df.columns]
    t3_cols  = [c for c in _T3_COLS  if c in df.columns]
    z_cols   = [c for c in df.columns if c.endswith("_zscore")]

    # ── Meta JSON ─────────────────────────────────────────────────────────────
    meta3 = {
        "n_epochs"               : int(len(df)),
        "n_feature_cols"         : int(df.shape[1]),
        "feature_columns"        : list(df.columns),
        "sqi_cols"               : sqi_cols,
        "t1_cols"                : t1_cols,
        "t2_cols"                : t2_cols,
        "t3_cols"                : t3_cols,
        "zscore_cols"            : z_cols,
        "sqi_merged"             : "overall_sqi" in df.columns,
        "detection_method_col"   : "detection_method" in df.columns,
        # Phase 1 DWT provenance
        "phase1_preprocessing"   : (phase1_meta or {}).get("preprocessing",    "DWT"),
        "phase1_dwt_wavelet"     : (phase1_meta or {}).get("dwt_wavelet",       "db4"),
        "phase1_dwt_level"       : (phase1_meta or {}).get("dwt_level",          5),
        "phase1_dwt_approx_zeroed": (phase1_meta or {}).get("dwt_approx_zeroed", True),
        "phase1_dwt_pli_zeroed"  : (phase1_meta or {}).get("dwt_pli_zeroed",     True),
        "phase1_dwt_soft_thresh" : (phase1_meta or {}).get("dwt_soft_thresh",    True),
        "phase1_polarity_inverted": (phase1_meta or {}).get("polarity_inverted", False),
        "phase1_fs"              : (phase1_meta or {}).get("fs",                 125),
        "phase1_mean_sqi"        : (phase1_meta or {}).get("mean_sqi",           None),
    }
    meta_path = os.path.join(out, "phase3_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta3, f, indent=2)
    logger.info(f"Phase 3 meta saved → {meta_path}")

    metrics = {
        "n_epochs"     : int(len(df)),
        "n_sqi_cols"   : len(sqi_cols),
        "n_t1_cols"    : len(t1_cols),
        "n_t2_cols"    : len(t2_cols),
        "n_t3_cols"    : len(t3_cols),
        "n_zscore_cols": len(z_cols),
    }

    # ── SQLite ─────────────────────────────────────────────────────────────────
    try:
        import sqlite3
        from Database.db_manager import DB_PATH as _DB_PATH
        db_row = get_record(config["record_name"],
                            config.get("dataset", "slpdb"), db_path)
        if db_row:
            mark_phase_done(db_row["record_id"], 3, out, db_path)
            # Write Phase 3 summary columns
            conn = sqlite3.connect(db_path)
            conn.execute("""
                UPDATE processing_status
                   SET phase3_n_epochs    = ?,
                       phase3_n_features  = ?,
                       phase3_t1_cols     = ?,
                       phase3_t2_cols     = ?,
                       phase3_t3_cols     = ?,
                       phase3_sqi_flagged = ?
                 WHERE record_id = ?
            """, (
                metrics["n_epochs"],
                metrics["n_t1_cols"] + metrics["n_t2_cols"] + metrics["n_t3_cols"]
                + metrics["n_sqi_cols"],
                metrics["n_t1_cols"],
                metrics["n_t2_cols"],
                metrics["n_t3_cols"],
                int(df["sqi_flag"].sum()) if "sqi_flag" in df.columns else 0,
                db_row["record_id"],
            ))
            conn.commit()
            conn.close()
            logger.info("SQLite: phase3_done=1, phase3 summary columns written.")
    except Exception as exc:
        logger.warning(f"SQLite update failed: {exc}")

    return metrics


# ── Column name lists for meta reporting ─────────────────────────────────────
_T1_COLS = [
    "mean_hr", "mean_rr", "rmssd", "pnn20", "sdnn", "min_hr", "max_hr",
    "hr_range", "rr_autocorr_lag1", "hf_power", "lf_hf_ratio", "lf_power",
    "resp_rate_est", "perm_en", "dfa_alpha1", "sd1", "sd1_sd2_ratio",
    "sampen", "rmssd_mean_rr_ratio", "sdnn_rmssd_ratio",
]
_T2_COLS = [
    "median_rr", "pnn50", "cv", "rr_iqr", "hf_nu", "lf_nu",
    "peak_hf_freq", "total_power", "sd2", "dfa_alpha2",
    "prsa_dc", "prsa_ac", "porta_asymmetry",
    "edr_breath_rate", "edr_regularity", "edr_baseline_wander",
    "sleep_cycle_pos", "hr_slope_epoch", "hr_epoch_delta",
]
_T3_COLS = [
    "guzik_asymmetry", "mse_scale2", "mse_scale4", "higuchi_fd",
    "fuzzy_entropy", "recurrence_rate", "rr_entropy_rate", "apen",
    "ulf_ratio", "edr_r_amplitude", "edr_qrs_area", "edr_breath_depth",
    "edr_apnea_index", "crc_cross_spectrum", "crc_coherence_hf",
    "crc_phase_sync", "r_amplitude_cv", "r_amplitude_mean",
    "qrs_duration", "pr_interval", "hf_peaks_count",
    "rr_spectral_entropy", "coherence_lf_hf", "vlf_power",
    "rr_triangular_index", "nn50", "nn20",
]

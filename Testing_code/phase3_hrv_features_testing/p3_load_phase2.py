"""
=============================================================================
phase3_hrv_features_testing/p3_load_phase2.py
Step 1 — Load all Phase 2 (and Phase 1) outputs for a DEVICE session.
IDENTICAL logic to the training pipeline — this only ever reads files by
output_dir path (no dataset-specific branching), so a device session's
Phase 1/2 outputs load exactly the same way an slpdb/hmc record's would.

Files loaded from <output_dir>/:
    phase2_rr_full.json       — per-epoch RR results from Phase 2
    phase1_meta.json          — DWT params, fs, polarity, SQI stats
    phase1_sqi.csv            — per-epoch SQI scores (optional)
=============================================================================
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from typing import Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Database.db_manager import get_record, DB_PATH


def load_phase2_results(config: dict,
                        logger: logging.Logger,
                        db_path: str = DB_PATH) -> list:
    """
    Load phase2_rr_full.json.
    Confirms phase2_done=1 in SQLite before loading.
    """
    out = config["output_dir"]

    # Optional: verify phase2 done in DB
    try:
        db_row = get_record(config["record_name"],
                            config.get("dataset", "device"), db_path)
        if db_row and db_row.get("phase2_done", 0) != 1:
            logger.warning(
                f"phase2_done != 1 for '{config['record_name']}' — "
                "Phase 2 may not have completed successfully."
            )
    except Exception as exc:
        logger.warning(f"SQLite check skipped: {exc}")

    path = os.path.join(out, "phase2_rr_full.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Phase 2 output not found: {path}\n"
            "Run p3_pipeline.py only after Phase 2 completes."
        )
    with open(path) as f:
        results = json.load(f)
    logger.info(f"Loaded {len(results)} epoch RR records from Phase 2.")
    return results


def load_phase1_meta(config: dict, logger: logging.Logger) -> dict:
    """
    Load phase1_meta.json. Logs DWT provenance for traceability.
    Returns full meta dict.
    """
    path = os.path.join(config["output_dir"], "phase1_meta.json")
    if not os.path.exists(path):
        logger.warning("phase1_meta.json not found — DWT provenance unavailable.")
        return {}
    with open(path) as f:
        meta = json.load(f)

    logger.info("Phase 1 DWT metadata:")
    logger.info(f"  preprocessing    : {meta.get('preprocessing', 'DWT')}")
    logger.info(f"  wavelet          : {meta.get('dwt_wavelet', 'db4')}")
    logger.info(f"  level            : {meta.get('dwt_level', 5)}")
    logger.info(f"  approx_zeroed    : {meta.get('dwt_approx_zeroed', True)}")
    logger.info(f"  pli_zeroed       : {meta.get('dwt_pli_zeroed', True)}")
    logger.info(f"  soft_thresh      : {meta.get('dwt_soft_thresh', True)}")
    logger.info(f"  polarity_inverted: {meta.get('polarity_inverted', False)}")
    logger.info(f"  fs               : {meta.get('fs', 125)} Hz")
    logger.info(f"  mean_sqi         : {meta.get('mean_sqi', 'N/A')}")
    logger.info(f"  pct_bad_epochs   : {meta.get('pct_bad', 'N/A')}%")
    return meta


def load_phase1_sqi(config: dict, logger: logging.Logger) -> Optional[pd.DataFrame]:
    """
    Load phase1_sqi.csv (kSQI, pSQI, basSQI, overall_sqi per epoch).
    Returns None if file not found — Phase 3 degrades gracefully.
    """
    path = os.path.join(config["output_dir"], "phase1_sqi.csv")
    if not os.path.exists(path):
        logger.warning("phase1_sqi.csv not found — SQI columns will be NaN.")
        return None
    sqi_df = pd.read_csv(path)
    logger.info(f"Loaded Phase 1 SQI: {len(sqi_df)} epochs, "
                f"cols: {list(sqi_df.columns)}")
    return sqi_df


def load_raw_ecg_epochs(config: dict, logger: logging.Logger) -> Optional[np.ndarray]:
    """
    Load preprocessed_epochs.npy for T3 morphology features.
    Returns None if unavailable (T3 morph features will be NaN).
    """
    path = os.path.join(config["output_dir"], "preprocessed_epochs.npy")
    if not os.path.exists(path):
        logger.warning("preprocessed_epochs.npy not found — T3 morph features will be NaN.")
        return None
    epochs = np.load(path)
    logger.info(f"Loaded ECG epochs for T3 morphology: shape={epochs.shape}")
    return epochs

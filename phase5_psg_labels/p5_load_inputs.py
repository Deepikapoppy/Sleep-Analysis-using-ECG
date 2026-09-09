"""
=============================================================================
phase5_psg_labels/p5_load_inputs.py
Step 1 — Load all upstream inputs needed by Phase 5.

Functions
─────────
  load_phase1_meta      phase1_meta.json  → DWT provenance + SQI stats
  load_phase3_meta      phase3_meta.json  → v2 feature availability
  load_phase1_sqi       phase1_sqi.csv    → per-epoch SQI (optional)
  load_psg_annotations  raw annotations   → SLPDB (.st) or HMC (text)
  get_original_fs       original recording sample-rate via wfdb / EDF
=============================================================================
"""

import os
import json
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import wfdb


# ─────────────────────────────────────────────────────────────────────────────
#  Upstream metadata
# ─────────────────────────────────────────────────────────────────────────────

def load_phase1_meta(config: dict, logger: logging.Logger) -> dict:
    """
    Load phase1_meta.json — DWT provenance + SQI statistics.
    Logs key fields for full pipeline traceability.
    Returns {} if the file is absent.
    """
    path = os.path.join(config["output_dir"], "phase1_meta.json")
    if not os.path.exists(path):
        logger.warning("phase1_meta.json not found — DWT provenance unavailable.")
        return {}
    with open(path) as f:
        meta = json.load(f)

    logger.info("Phase 1 metadata loaded (Phase 5 traceability):")
    logger.info(f"  preprocessing    : {meta.get('preprocessing', 'DWT')}")
    logger.info(f"  DWT wavelet      : {meta.get('dwt_wavelet', 'db4')}")
    logger.info(f"  DWT level        : {meta.get('dwt_level', 5)}")
    logger.info(f"  polarity_inverted: {meta.get('polarity_inverted', False)}")
    logger.info(f"  mean_sqi         : {meta.get('mean_sqi')}")
    logger.info(f"  pct_bad_epochs   : {meta.get('pct_bad')}%")
    logger.info(f"  n_epochs         : {meta.get('n_epochs')}")
    return meta


def load_phase3_meta(config: dict, logger: logging.Logger) -> dict:
    """
    Load phase3_meta.json — v2 feature availability.
    Returns {} if the file is absent.
    """
    path = os.path.join(config["output_dir"], "phase3_meta.json")
    if not os.path.exists(path):
        logger.warning("phase3_meta.json not found — v2 feature info unavailable.")
        return {}
    with open(path) as f:
        meta = json.load(f)

    feats  = meta.get("feature_columns", [])
    v2_new = [f for f in ("pnn20", "hr_range", "resp_rate_est", "perm_en",
                           "rmssd_mean_rr_ratio", "median_rr", "rr_iqr")
              if f in feats]
    logger.info("Phase 3 metadata loaded (Phase 5 traceability):")
    logger.info(f"  n_features      : {meta.get('n_feature_cols')}")
    logger.info(f"  sqi_merged      : {meta.get('sqi_merged')}")
    logger.info(f"  v2 features     : {v2_new}")
    return meta


def load_phase1_sqi(config: dict,
                     logger: logging.Logger) -> Optional[pd.DataFrame]:
    """
    Load phase1_sqi.csv for bad-epoch and SQI overlays in plots.
    Returns None if file is absent — Phase 5 degrades gracefully.
    """
    path = os.path.join(config["output_dir"], "phase1_sqi.csv")
    if not os.path.exists(path):
        logger.warning("phase1_sqi.csv not found — SQI overlay unavailable.")
        return None
    sqi_df = pd.read_csv(path)
    logger.info(f"Loaded Phase 1 SQI: {len(sqi_df)} epochs, "
                f"cols: {list(sqi_df.columns)}")
    return sqi_df


# ─────────────────────────────────────────────────────────────────────────────
#  PSG annotations
# ─────────────────────────────────────────────────────────────────────────────

def load_psg_annotations(config: dict, logger: logging.Logger):
    """
    Load raw PSG annotations.
    SLPDB  → WFDB .st file (wfdb.rdann).
    HMC    → text file parsed via Phase 0 helpers.
    Returns the raw annotation object to be consumed by build_psg_hypnogram.
    """
    dataset = config.get("dataset", "slpdb")
    from phase0_dataset_management.p0_config import get_record_path
    record_path = get_record_path(config)

    if dataset == "hmc":
        from phase0_dataset_management import (
            _get_hmc_ann_path, parse_hmc_annotations,
        )
        ann_path = _get_hmc_ann_path(record_path)
        logger.info(f"Loading HMC PSG annotations from: {ann_path}")
        try:
            ann_records = parse_hmc_annotations(ann_path, logger)
        except Exception as exc:
            logger.error(f"Failed to read HMC annotations: {exc}")
            raise
        logger.info(f"  Total annotations : {len(ann_records)}")
        return ann_records

    # SLPDB — WFDB .st annotation file
    logger.info(f"Loading SLPDB PSG annotations from: {record_path}.st")
    try:
        ann = wfdb.rdann(record_path, "st")
    except Exception as exc:
        logger.error(f"Failed to read WFDB annotations: {exc}")
        raise

    logger.info(f"  Total annotations : {len(ann.sample)}")
    logger.info(f"  Unique labels     : {list(set(ann.aux_note))}")
    return ann


# ─────────────────────────────────────────────────────────────────────────────
#  Original sample-rate
# ─────────────────────────────────────────────────────────────────────────────

def get_original_fs(config: dict,
                     logger: logging.Logger) -> Tuple[float, int]:
    """
    Return (fs_orig, sig_len) for the raw recording.
    Uses wfdb for SLPDB and the Phase 0 EDF loader for HMC.
    """
    dataset = config.get("dataset", "slpdb")
    from phase0_dataset_management.p0_config import get_record_path
    record_path = get_record_path(config)

    if dataset == "hmc":
        from phase0_dataset_management import load_edf_signal
        edf_path = record_path + ".edf"
        logger.info(f"Loading EDF to read original Fs: {edf_path}")
        try:
            signal, fs, *rest = load_edf_signal(
                edf_path,
                config.get("hmc_ecg_keywords", ["ECG", "EKG"]),
                logger,
            )
        except Exception as exc:
            logger.error(f"Failed to read EDF: {exc}")
            raise
        logger.info(f"Original recording Fs = {fs} Hz")
        return float(fs), len(signal)

    record  = wfdb.rdrecord(record_path)
    fs_orig = record.fs
    logger.info(f"Original recording Fs = {fs_orig} Hz")
    return float(fs_orig), record.sig_len

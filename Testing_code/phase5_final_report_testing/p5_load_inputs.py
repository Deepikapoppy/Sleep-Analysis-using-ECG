"""
=============================================================================
phase5_final_report_testing/p5_load_inputs.py
Step 1 — Load upstream outputs for one device session.

Unlike training's Phase 5 (which loads real PSG annotation files), this
package has nothing to parse for ground truth — device sessions never
carry PSG labels. Instead it loads:
  - phase1_meta.json    (Phase 1 testing — DWT provenance, n_epochs)
  - phase1_sqi.csv       (Phase 1 testing — per-epoch SQI, for the overlay plot)
  - phase3_hrv_features.csv (Phase 3 testing — HRV features per epoch)
  - phase4b_predict_meta.json (Phase 4b testing — model info, stage counts)
  - phase4b_ecg_hypnogram_rf.npy       (Phase 4b testing — smoothed prediction)
  - phase4b_ecg_hypnogram_rf_raw.npy   (Phase 4b testing — raw prediction)

All paths are read from config["output_dir"] — the same per-session folder
every prior testing phase already writes into.
=============================================================================
"""

import os
import json
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _require(path: str, what: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{what} not found: {path}\n"
            f"Make sure the prerequisite phase has already run for this "
            f"session (Phase 1 / Phase 3 / Phase 4b testing)."
        )


def load_phase1_meta(config: dict, logger: logging.Logger) -> dict:
    path = os.path.join(config["output_dir"], "phase1_meta.json")
    _require(path, "phase1_meta.json")
    with open(path) as f:
        meta = json.load(f)
    logger.info(
        f"phase1_meta: n_epochs={meta.get('n_epochs')}, "
        f"dwt={meta.get('dwt_wavelet')} L={meta.get('dwt_level')}, "
        f"mean_sqi={meta.get('mean_sqi')}"
    )
    return meta


def load_phase1_sqi(config: dict, logger: logging.Logger):
    """Returns None (not a hard requirement) if phase1_sqi.csv is missing —
    the SQI overlay plot is skipped gracefully in that case, same as
    training's Phase 5 behaviour."""
    path = os.path.join(config["output_dir"], "phase1_sqi.csv")
    if not os.path.exists(path):
        logger.warning("phase1_sqi.csv not found — SQI overlay plot will be skipped.")
        return None
    df = pd.read_csv(path)
    logger.info(f"phase1_sqi: {len(df)} epoch rows loaded")
    return df


def load_phase3_features(config: dict, logger: logging.Logger):
    """Returns None (not a hard requirement) if phase3_hrv_features.csv is
    missing — the HRV-by-stage boxplot is skipped gracefully in that case."""
    path = os.path.join(config["output_dir"], "phase3_hrv_features.csv")
    if not os.path.exists(path):
        logger.warning("phase3_hrv_features.csv not found — HRV summary plot will be skipped.")
        return None
    df = pd.read_csv(path)
    logger.info(f"phase3_hrv_features: {len(df)} epoch rows, {df.shape[1]} columns")
    return df


def load_phase4b_predictions(config: dict, logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Hard requirement — there is nothing to report on without a prediction.

    Returns
    -------
    pred_raw    : np.ndarray, int stage codes 0-4, pre-smoothing
    pred_smooth : np.ndarray, int stage codes 0-4, post-smoothing (the one
                  used for every plot / summary in this package)
    predict_meta: dict — phase4b_predict_meta.json content (model info,
                  stage counts, feature alignment diagnostics)
    """
    out = config["output_dir"]
    raw_path    = os.path.join(out, "phase4b_ecg_hypnogram_rf_raw.npy")
    smooth_path = os.path.join(out, "phase4b_ecg_hypnogram_rf.npy")
    meta_path   = os.path.join(out, "phase4b_predict_meta.json")

    _require(raw_path,    "phase4b_ecg_hypnogram_rf_raw.npy (Phase 4b testing output)")
    _require(smooth_path, "phase4b_ecg_hypnogram_rf.npy (Phase 4b testing output)")
    _require(meta_path,   "phase4b_predict_meta.json (Phase 4b testing output)")

    pred_raw    = np.load(raw_path)
    pred_smooth = np.load(smooth_path)
    with open(meta_path) as f:
        predict_meta = json.load(f)

    logger.info(
        f"phase4b predictions: {len(pred_smooth)} epochs, "
        f"model={predict_meta.get('model_type')}, "
        f"stages(smoothed)={predict_meta.get('stage_counts_smooth')}"
    )
    return pred_raw, pred_smooth, predict_meta


def load_all(config: dict, logger: logging.Logger) -> Dict:
    """Convenience wrapper — loads everything this package needs at once."""
    phase1_meta = load_phase1_meta(config, logger)
    sqi_df      = load_phase1_sqi(config, logger)
    hrv_df      = load_phase3_features(config, logger)
    pred_raw, pred_smooth, predict_meta = load_phase4b_predictions(config, logger)

    return {
        "phase1_meta" : phase1_meta,
        "sqi_df"      : sqi_df,
        "hrv_df"      : hrv_df,
        "pred_raw"    : pred_raw,
        "pred_smooth" : pred_smooth,
        "predict_meta": predict_meta,
    }

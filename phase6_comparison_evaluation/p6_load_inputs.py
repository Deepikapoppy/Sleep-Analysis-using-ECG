"""
=============================================================================
phase6_comparison_evaluation/p6_load_inputs.py
Step 1 — Load all upstream inputs needed by Phase 6.

Functions
─────────
  load_pipeline_metadata   phase5_meta.json (bundles Phase 1 DWT provenance
                            + Phase 3 v2 feature info) → falls back to
                            phase1_meta.json if Phase 5 meta is absent
  apply_final_smoothing    K=3 (90s) median filter on the ECG hypnogram
  load_all                 ECG hypnogram (Phase 4) + PSG hypnogram (Phase 5)
                            + HRV features (Phase 3)
  align                    trims all three arrays/frames to a common length

Prerequisites
─────────────
Phase 4 output (ecg_hypnogram_smooth.npy or ecg_hypnogram_raw.npy) required.
Phase 5 output (psg_hypnogram.npy) required.
Phase 3 output (phase3_hrv_features.csv) required.
=============================================================================
"""

import os
import json
import logging

import numpy as np
import pandas as pd
from scipy.signal import medfilt as _scipy_medfilt


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline metadata  (Phase 5 meta bundles Phase 1 & Phase 3 provenance)
# ─────────────────────────────────────────────────────────────────────────────

def load_pipeline_metadata(config: dict, logger: logging.Logger) -> dict:
    """
    Load phase5_meta.json written by Phase 5 (which itself bundles Phase 1
    DWT provenance and Phase 3 v2 feature info).
    Falls back to phase1_meta.json directly if Phase 5 meta is absent.
    Returns {} if neither file exists.
    """
    p5_path = os.path.join(config["output_dir"], "phase5_meta.json")
    p1_path = os.path.join(config["output_dir"], "phase1_meta.json")

    meta = {}
    if os.path.exists(p5_path):
        with open(p5_path) as f:
            meta = json.load(f)
        logger.info("Pipeline metadata loaded from phase5_meta.json:")
    elif os.path.exists(p1_path):
        with open(p1_path) as f:
            meta = json.load(f)
        logger.warning("phase5_meta.json absent — fell back to phase1_meta.json.")
    else:
        logger.warning("No pipeline metadata found — DWT provenance unavailable.")
        return {}

    logger.info(f"  DWT wavelet      : {meta.get('phase1_dwt_wavelet', 'db4')}")
    logger.info(f"  DWT level        : {meta.get('phase1_dwt_level', 5)}")
    logger.info(f"  polarity_inverted: {meta.get('phase1_polarity_inverted', False)}")
    logger.info(f"  Phase 1 SQI mean : {meta.get('phase1_mean_sqi')}")
    logger.info(f"  Phase 3 n_feat   : {meta.get('phase3_n_features')}")
    return meta


# ─────────────────────────────────────────────────────────────────────────────
#  Final post-load smoothing
# ─────────────────────────────────────────────────────────────────────────────

def apply_final_smoothing(ecg_hyp: np.ndarray,
                           logger: logging.Logger,
                           kernel: int = 3) -> np.ndarray:
    """K=3 (90 s) median filter — removes single-epoch jitter."""
    kernel   = kernel if kernel % 2 == 1 else kernel + 1
    smoothed = _scipy_medfilt(ecg_hyp.astype(float), kernel_size=kernel).astype(int)
    n_changed = int(np.sum(smoothed != ecg_hyp))
    logger.info(
        f"Phase 6 final smoothing (K={kernel}): "
        f"{n_changed} epochs changed ({100 * n_changed / max(len(ecg_hyp), 1):.1f}%)"
    )
    return smoothed


# ─────────────────────────────────────────────────────────────────────────────
#  Load ECG hypnogram (Phase 4), PSG hypnogram (Phase 5), HRV features (Phase 3)
# ─────────────────────────────────────────────────────────────────────────────

def load_all(config: dict, logger: logging.Logger):
    """
    Returns (ecg_hyp, psg_hyp, hrv_df).

    ecg_hyp : prefers ecg_hypnogram_smooth.npy, falls back to
              ecg_hypnogram_raw.npy (Phase 4 output)
    psg_hyp : psg_hypnogram.npy (Phase 5 output)
    hrv_df  : phase3_hrv_features.csv (Phase 3 output)
    """
    out = config["output_dir"]

    smooth_path = os.path.join(out, "ecg_hypnogram_smooth.npy")
    raw_path    = os.path.join(out, "ecg_hypnogram_raw.npy")

    if os.path.exists(smooth_path):
        ecg = np.load(smooth_path)
        logger.info("Loaded ECG hypnogram: ecg_hypnogram_smooth.npy")
    elif os.path.exists(raw_path):
        ecg = np.load(raw_path)
        logger.warning("ecg_hypnogram_smooth.npy not found — using raw.")
    else:
        raise FileNotFoundError(f"No ECG hypnogram in {out}. Run Phase 4 first.")

    try:
        psg = np.load(os.path.join(out, "psg_hypnogram.npy"))
        hrv = pd.read_csv(os.path.join(out, "phase3_hrv_features.csv"))
    except Exception as e:
        raise FileNotFoundError(f"Missing Phase 5/3 outputs in {out}: {e}")

    logger.info(f"ECG: {len(ecg)} epochs | PSG: {len(psg)} epochs")
    ecg = apply_final_smoothing(ecg, logger, kernel=3)
    return ecg, psg, hrv


# ─────────────────────────────────────────────────────────────────────────────
#  Align lengths
# ─────────────────────────────────────────────────────────────────────────────

def align(ecg_hyp: np.ndarray,
          psg_hyp: np.ndarray,
          hrv: pd.DataFrame,
          logger: logging.Logger):
    """Trim ecg_hyp, psg_hyp and hrv to the shortest common length."""
    n = min(len(ecg_hyp), len(psg_hyp), len(hrv))
    logger.info(f"Aligned to {n} common epochs.")
    return ecg_hyp[:n], psg_hyp[:n], hrv.iloc[:n].reset_index(drop=True)

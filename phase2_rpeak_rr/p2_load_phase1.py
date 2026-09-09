"""
=============================================================================
phase2_rpeak_rr/p2_load_phase1.py
Step 1 — Load all Phase-1 outputs from disk.

Files loaded from <output_dir>/ (written by Phase 1):
    preprocessed_epochs.npy   — (n_epochs, samples_per_epoch)
    bad_epoch_mask.npy         — bool array (n_epochs,)
    clean_ecg_full.npy         — full normalised ECG
    raw_ecg_ds_full.npy        — downsampled pre-DWT ECG (for comparison)
    phase1_meta.json           — fs, wavelet, polarity, SQI stats

SQLite is also checked to confirm phase1_done=1 before loading.
=============================================================================
"""

import os
import sys
import json
import logging
import numpy as np
from typing import Optional, Tuple, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Database.db_manager import get_record, DB_PATH


def load_phase1_outputs(config: dict,
                        logger: logging.Logger,
                        db_path: str = DB_PATH) -> Tuple:
    """
    Load all Phase-1 output files for the current record.

    Returns
    -------
    epochs      : np.ndarray  (n_epochs, spe)
    bad_mask    : np.ndarray  bool (n_epochs,)
    fs          : float
    meta        : dict  (full phase1_meta.json contents)
    """
    out = config["output_dir"]

    # ── Check SQLite phase1_done flag ─────────────────────────────────────────
    db_row = get_record(config["record_name"],
                        config.get("dataset", "slpdb"), db_path)
    if db_row is None:
        logger.warning(
            f"Record '{config['record_name']}' not found in SQLite — "
            "proceeding anyway."
        )

    # ── Load numpy arrays ─────────────────────────────────────────────────────
    epochs_path = os.path.join(out, "preprocessed_epochs.npy")
    mask_path   = os.path.join(out, "bad_epoch_mask.npy")

    if not os.path.exists(epochs_path):
        raise FileNotFoundError(
            f"Phase 1 output not found: {epochs_path}\n"
            "Run Phase 1 (p1_pipeline.py) before Phase 2."
        )

    epochs   = np.load(epochs_path)
    bad_mask = np.load(mask_path)

    # ── Load meta JSON ─────────────────────────────────────────────────────────
    meta_path = os.path.join(out, "phase1_meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    fs                = float(meta["fs"])
    polarity_inverted = bool(meta.get("polarity_inverted", False))
    dwt_wavelet       = meta.get("dwt_wavelet", "db4")
    dwt_level         = meta.get("dwt_level", 5)

    logger.info(f"Loaded {len(epochs)} epochs  |  fs={fs} Hz  |  "
                f"bad={int(bad_mask.sum())}")
    logger.info(f"Phase 1: DWT wavelet={dwt_wavelet}, level={dwt_level}  |  "
                f"polarity_inverted={polarity_inverted}")

    return epochs, bad_mask, fs, meta


def load_raw_ecg_for_viz(config: dict) -> Optional[np.ndarray]:
    """Load raw downsampled ECG (before DWT cleaning) for comparison plots."""
    path = os.path.join(config["output_dir"], "raw_ecg_ds_full.npy")
    if os.path.exists(path):
        return np.load(path)
    return None


def load_clean_ecg_for_viz(config: dict) -> Optional[np.ndarray]:
    """Load DWT-cleaned + z-scored full ECG saved by Phase 1."""
    path = os.path.join(config["output_dir"], "clean_ecg_full.npy")
    if os.path.exists(path):
        return np.load(path)
    return None

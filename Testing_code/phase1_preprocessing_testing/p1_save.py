"""
=============================================================================
phase1_preprocessing_testing/p1_save.py
Step 9 — Save all Phase-1 outputs to disk + SQLite status update.
IDENTICAL logic to the training pipeline — record_name is simply a device
session id (e.g. "ADM937394258") instead of an slpdb/hmc record name, and
everything else (arrays, SQI csv, meta json) is written the same way.

Files written to <output_dir>/
    preprocessed_epochs.npy   — (n_epochs, samples_per_epoch)
    bad_epoch_mask.npy         — bool array (n_epochs,)
    clean_ecg_full.npy         — normalised full-length ECG
    raw_ecg_ds_full.npy        — downsampled (pre-DWT) full-length ECG
    phase1_sqi.csv             — SQI metrics per epoch
    phase1_meta.json           — all parameters + summary stats
=============================================================================
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import shutil
from typing import Optional


def _prepare_array_for_save(arr: np.ndarray, logger: logging.Logger) -> np.ndarray:
    """Reduce disk usage for large numerical arrays when possible."""
    if isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.floating):
        if arr.dtype == np.float64:
            logger.warning(
                "Saving floating-point array as float32 to reduce disk usage "
                f"(shape={arr.shape}, dtype={arr.dtype})"
            )
            return arr.astype(np.float32)
    return arr


def save_preprocessed(epochs: np.ndarray,
                       bad_mask: np.ndarray,
                       clean_ecg: np.ndarray,
                       raw_ecg_ds: np.ndarray,
                       sqi_df: pd.DataFrame,
                       fs: float,
                       config: dict,
                       logger: logging.Logger,
                       polarity_inverted: bool = False,
                       dwt_wavelet: str = "db4",
                       dwt_level: int = 5,
                       recording_start_time: Optional[str] = None) -> dict:
    """
    Persist all Phase-1 artefacts and return meta dict.
    """
    out = config["output_dir"]
    os.makedirs(out, exist_ok=True)

    # ── Check free disk space before attempting large writes
    try:
        arrs = [epochs, bad_mask, clean_ecg, raw_ecg_ds]
        prepared_arrs = [_prepare_array_for_save(a, logger) for a in arrs]
        total_bytes = sum(int(getattr(a, "nbytes", 0)) for a in prepared_arrs)
        # small safety buffer + CSV/JSON overhead
        required = int(total_bytes * 1.10) + 5_242_880
        free_bytes = shutil.disk_usage(out).free
        if free_bytes < required:
            logger.warning(
                "Low free disk space detected; writing reduced-size arrays "
                f"(need ~{required} bytes, have {free_bytes} bytes)"
            )
    except OSError:
        # propagate OS errors (including disk errors) clearly
        raise
    except Exception:
        # if estimation fails for any reason, continue and let np.save handle errors
        pass

    # ── numpy arrays ──────────────────────────────────────────────────────────
    try:
        np.save(os.path.join(out, "preprocessed_epochs.npy"), prepared_arrs[0])
        np.save(os.path.join(out, "bad_epoch_mask.npy"),      prepared_arrs[1])
        np.save(os.path.join(out, "clean_ecg_full.npy"),      prepared_arrs[2])
        np.save(os.path.join(out, "raw_ecg_ds_full.npy"),     prepared_arrs[3])
    except OSError as e:
        logger.error(f"Failed to write numpy output files: {e}")
        raise

    # ── SQI CSV ───────────────────────────────────────────────────────────────
    sqi_df.to_csv(os.path.join(out, "phase1_sqi.csv"), index=False)

    # ── meta JSON ─────────────────────────────────────────────────────────────
    meta = {
        "record"              : config.get("record_name"),
        "dataset"             : config.get("dataset", "device"),
        "n_epochs"            : int(len(epochs)),
        "samples_per_ep"      : int(epochs.shape[1]),
        "fs"                  : int(fs),
        "epoch_sec"           : config["epoch_sec"],
        "n_bad_epochs"        : int(bad_mask.sum()),
        "pct_bad"             : round(100 * bad_mask.sum() / len(epochs), 2),
        "mean_sqi"            : round(float(sqi_df["overall_sqi"].mean()), 3),
        "pct_low_sqi"         : round(float(
                                    (sqi_df["overall_sqi"] < 0.50).mean() * 100
                                ), 2),
        "polarity_inverted"   : bool(polarity_inverted),
        "preprocessing"       : "DWT",
        "dwt_wavelet"         : dwt_wavelet,
        "dwt_level"           : int(dwt_level),
        "dwt_approx_zeroed"   : True,
        "dwt_pli_zeroed"      : True,
        "dwt_soft_thresh"     : True,
        "recording_start_time": recording_start_time,
    }
    with open(os.path.join(out, "phase1_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Saved preprocessed_epochs.npy  → {out}")
    logger.info(f"Saved raw_ecg_ds_full.npy      → {out}")
    logger.info(f"Saved phase1_sqi.csv           → {out}")
    logger.info(f"Saved phase1_meta.json         → {out}")
    logger.info(f"  DWT: wavelet={dwt_wavelet}, level={dwt_level}")
    logger.info(f"  Polarity inverted: {polarity_inverted}")
    return meta

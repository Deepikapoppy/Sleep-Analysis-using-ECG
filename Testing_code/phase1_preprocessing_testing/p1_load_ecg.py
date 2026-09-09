"""
=============================================================================
phase1_preprocessing_testing/p1_load_ecg.py
Step 1 — Load raw ECG signal for a DEVICE session (Test_Dataset JSON chunks).

slpdb/hmc loading has been intentionally REMOVED from this testing package —
production code for those datasets lives in phase1_preprocessing/p1_load_ecg.py.
This file only ever loads "device" sessions, using the exact same session
manifest built by phase0_dataset_management_testing.

A device "record" is a SESSION (many ~30 s chunk JSON files grouped by
admissionId, sorted by window_start_ms), not one file — see
phase0_dataset_management_testing/p0_scan_datasets.py for the grouping logic.
This loader reuses that manifest so Phase 0 and Phase 1 never disagree about
which chunks belong to which session or what order they're in.
=============================================================================
"""

import os
import sys
import json
import logging
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase0_dataset_management_testing.p0_scan_datasets import get_device_chunk_files
from phase0_dataset_management_testing.p0_metadata_extraction import _fs_from_chunk


def load_ecg(config: dict, logger: logging.Logger):
    """
    Load and concatenate one device session's ECG chunks.

    Gaps between consecutive chunks beyond `device_gap_tolerance_ms` are
    NaN-padded (not silently stitched) so Phase 1's own bad-epoch/SQI logic
    (Steps 6/7, unchanged from training) naturally flags them — the padding
    is then median-filled here so no NaN ever reaches the DWT step, exactly
    like the HMC loader in production handles its own NaNs.

    Returns
    -------
    ecg_raw : np.ndarray  — 1-D concatenated raw ECG for the whole session
    fs_orig : float       — median fs across chunks (Hz)
    record  : lightweight stub exposing .fs / .sig_len / .sig_name / .p_signal
    """
    session_id = config["record_name"]
    ecg_key    = config.get("device_ecg_key", "ECG_CH_A")
    gap_tol    = config.get("device_gap_tolerance_ms", 500)
    fs_tol_pct = config.get("device_fs_tolerance_pct", 2.0)

    chunks = get_device_chunk_files(config, session_id, logger)
    if not chunks:
        raise FileNotFoundError(
            f"[DEVICE] No chunk files found for session '{session_id}'. "
            f"Check local_test_device_path/local_device_path and "
            f"device_id_key in CONFIG."
        )

    logger.info(f"[DEVICE] Loading session '{session_id}': {len(chunks)} chunk(s)")

    per_chunk_fs = []
    pieces       = []
    n_gaps       = 0

    for i, c in enumerate(chunks):
        with open(c["path"], "r", encoding="utf-8") as f:
            full = json.load(f)
        seg  = np.asarray(full.get(ecg_key, []), dtype="float32")
        fs_c = _fs_from_chunk(c)
        if fs_c:
            per_chunk_fs.append(fs_c)

        if i > 0:
            prev   = chunks[i - 1]
            gap_ms = c["window_start_ms"] - prev["window_end_ms"]
            if gap_ms > gap_tol and per_chunk_fs:
                fs_est    = per_chunk_fs[-1]
                n_missing = int(round((gap_ms / 1000) * fs_est))
                if n_missing > 0:
                    pieces.append(np.full(n_missing, np.nan, dtype="float32"))
                    n_gaps += 1
                    logger.warning(
                        f"  Gap {gap_ms:.0f} ms between "
                        f"{os.path.basename(prev['path'])} → "
                        f"{os.path.basename(c['path'])} — "
                        f"padded {n_missing} NaN sample(s)"
                    )
        pieces.append(seg)

    ecg_raw = np.concatenate(pieces).astype("float32") if pieces else np.array([], dtype="float32")
    if not per_chunk_fs:
        raise ValueError(f"[DEVICE] Could not determine fs for session '{session_id}'")

    fs_orig  = float(np.median(per_chunk_fs))
    outliers = [f for f in per_chunk_fs
               if abs(f - fs_orig) / fs_orig * 100 > fs_tol_pct]
    if outliers:
        logger.warning(
            f"  {len(outliers)}/{len(per_chunk_fs)} chunk(s) deviate "
            f">{fs_tol_pct}% from median fs ({fs_orig:.2f} Hz)"
        )

    # Median-fill remaining NaNs (gap padding + any in-chunk corrupt samples)
    # — same treatment production's HMC loader gives its own NaNs.
    if np.isnan(ecg_raw).any():
        ecg_raw = np.where(np.isfinite(ecg_raw), ecg_raw, np.nanmedian(ecg_raw))

    logger.info(f"  Raw ECG shape : {ecg_raw.shape}")
    logger.info(f"  Original Fs   : {fs_orig:.4f} Hz")
    logger.info(f"  Duration      : {len(ecg_raw)/fs_orig/3600:.4f} hours "
               f"({n_gaps} gap(s) padded)")

    class _DeviceRecord:
        pass
    record          = _DeviceRecord()
    record.fs       = fs_orig
    record.sig_len  = len(ecg_raw)
    record.sig_name = [ecg_key]
    record.p_signal = ecg_raw.reshape(-1, 1)

    return ecg_raw, fs_orig, record


def extract_recording_start_time(config: dict, record,
                                  logger: logging.Logger):
    """
    Extract recording start as 'YYYY-MM-DD HH:MM:SS' string, taken from the
    first chunk's window_start_ms. Returns None if unavailable.

    (Device sessions can span midnight and aren't guaranteed same-day like
    slpdb/hmc, so — unlike the production loader's 'HH:MM:SS' — the date is
    kept here too.)
    """
    session_id = config["record_name"]
    try:
        chunks = get_device_chunk_files(config, session_id, logger)
        if chunks and chunks[0].get("window_start_ms"):
            dt = datetime.fromtimestamp(chunks[0]["window_start_ms"] / 1000)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as exc:
        logger.warning(f"Could not extract recording start time: {exc}")

    logger.info("Recording start time unavailable — timestamps will be relative.")
    return None

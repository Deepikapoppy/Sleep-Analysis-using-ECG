"""
=============================================================================
phase1_preprocessing/p1_load_ecg.py
Step 1 — Load raw ECG signal from disk (SLPDB WFDB or HMC EDF).
=============================================================================
"""

import os
import sys
import logging
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase0_dataset_management.p0_config import get_record_path
from phase0_dataset_management.p0_metadata_extraction import load_edf_signal


def load_ecg(config: dict, logger: logging.Logger):
    """
    Load raw ECG signal.

    • SLPDB  → reads WFDB record (.hea / .dat)
    • HMC    → reads EDF file, finds ECG channel by keyword

    Returns
    -------
    ecg_raw  : np.ndarray  — 1-D raw ECG
    fs_orig  : float       — original sampling frequency (Hz)
    record   : wfdb.Record or _HMCRecord stub
    """
    dataset = config.get("dataset", "slpdb")

    if dataset == "hmc":
        edf_path = get_record_path(config) + ".edf"
        logger.info(f"[HMC] Loading EDF ECG from: {edf_path}")

        keywords = config.get("hmc_ecg_keywords", ["ECG", "EKG"])
        ecg_raw, fs_orig, channel_names, ecg_idx, _ = load_edf_signal(
            edf_path, keywords, logger
        )
        ecg_raw = np.where(np.isfinite(ecg_raw), ecg_raw, np.nanmedian(ecg_raw))

        logger.info(f"  Raw ECG shape : {ecg_raw.shape}")
        logger.info(f"  Original Fs   : {fs_orig} Hz")
        logger.info(f"  Duration      : {len(ecg_raw)/fs_orig/3600:.2f} hours")

        class _HMCRecord:
            pass
        record           = _HMCRecord()
        record.fs        = fs_orig
        record.sig_len   = len(ecg_raw)
        record.sig_name  = channel_names
        record.p_signal  = ecg_raw.reshape(-1, 1)
        return ecg_raw, fs_orig, record

    else:
        import wfdb
        record_path = get_record_path(config)
        logger.info(f"[SLPDB] Loading ECG from: {record_path}")

        record  = wfdb.rdrecord(record_path)
        ecg_raw = record.p_signal[:, config["ecg_channel"]].astype(np.float64)
        fs_orig = record.fs
        ecg_raw = np.where(np.isfinite(ecg_raw), ecg_raw, np.nanmedian(ecg_raw))

        logger.info(f"  Raw ECG shape : {ecg_raw.shape}")
        logger.info(f"  Original Fs   : {fs_orig} Hz")
        logger.info(f"  Duration      : {len(ecg_raw)/fs_orig/3600:.2f} hours")
        return ecg_raw, fs_orig, record


def extract_recording_start_time(config: dict, record,
                                  logger: logging.Logger):
    """
    Extract recording start as 'HH:MM:SS' string.
    Returns None if unavailable.

    • SLPDB : reads record.base_datetime / record.base_time
    • HMC   : reads EDF header via pyedflib
    """
    dataset = config.get("dataset", "slpdb")
    try:
        if dataset == "hmc":
            import pyedflib
            edf_path = get_record_path(config) + ".edf"
            f  = pyedflib.EdfReader(edf_path)
            dt = f.getStartdatetime()
            f.close()
            if dt:
                return dt.strftime("%H:%M:%S")
        else:
            bt = getattr(record, "base_datetime", None)
            if bt is None:
                bt = getattr(record, "base_time", None)
            if bt is not None and hasattr(bt, "strftime"):
                return bt.strftime("%H:%M:%S")
    except Exception as exc:
        logger.warning(f"Could not extract recording start time: {exc}")

    logger.info("Recording start time unavailable — timestamps will be relative.")
    return None

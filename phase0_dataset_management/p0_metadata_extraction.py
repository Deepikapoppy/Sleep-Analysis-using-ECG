"""
=============================================================================
phase0_dataset_management/p0_metadata_extraction.py
Reads signal headers (WFDB / EDF) for each registered record and stores
the extracted metadata in the record_metadata SQLite table.

Also saves a per-record record_info.json to the output directory for
backward compatibility with downstream phases.
=============================================================================
"""

import os
import sys
import json
import glob
import logging
from typing import Optional, Tuple, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Database.db_manager import (
    get_all_records, upsert_metadata, mark_phase_done, mark_phase_failed, DB_PATH
)


# ─────────────────────────────────────────────────────────────────────────────
#  EDF helpers  (HMC)
# ─────────────────────────────────────────────────────────────────────────────

def _find_ecg_channel(channel_names, keywords):
    for idx, name in enumerate(channel_names):
        for kw in keywords:
            if kw.lower() in name.lower():
                return idx, name
    return 0, channel_names[0] if channel_names else "ch0"


def load_edf_signal(edf_path: str, ecg_keywords: list,
                    logger: logging.Logger = None):
    """
    Read EDF header + ECG channel via pyedflib.
    Returns (signal_array, fs, channel_names, ecg_idx).
    """
    try:
        import pyedflib
    except ImportError:
        raise ImportError("Install pyedflib:  pip install pyedflib")

    f             = pyedflib.EdfReader(edf_path)
    channel_names = list(f.getSignalLabels())
    fs_list       = list(f.getSampleFrequencies())
    ecg_idx, ecg_name = _find_ecg_channel(channel_names, ecg_keywords)
    fs            = int(fs_list[ecg_idx])
    if logger:
        logger.info(f"  EDF channels : {channel_names}")
        logger.info(f"  ECG channel  : [{ecg_idx}] '{ecg_name}'  @ {fs} Hz")
    signal = f.readSignal(ecg_idx).astype("float64")
    try:
        start_dt = f.getStartdatetime()
        rec_start = start_dt.strftime("%H:%M:%S") if start_dt else None
    except Exception:
        rec_start = None
    f._close()
    del f
    return signal, fs, channel_names, ecg_idx, rec_start


def _get_hmc_ann_path(base_path: str) -> Optional[str]:
    for suffix in ("_sleepscoring.txt", "_sleepscoring"):
        p = base_path + suffix
        if os.path.exists(p):
            return p
    return None


def parse_hmc_annotations(ann_path: str, logger=None):
    """
    Parse HMC sleepscoring text — returns list of (onset_sec, dur_sec, label).
    Supports Format A (tab/csv with header) and Format C (plain list).
    """
    import re
    with open(ann_path, "r", encoding="utf-8", errors="replace") as fh:
        lines = [ln.rstrip("\n\r") for ln in fh if ln.strip()]
    data_lines = [ln for ln in lines if not ln.startswith("#")]
    if not data_lines:
        raise ValueError(f"Empty annotation file: {ann_path}")

    first     = data_lines[0].lower()
    has_header = any(kw in first for kw in
                     ("sleep stage", "start", "duration", "onset"))
    delim      = "\t" if "\t" in data_lines[0] else ","
    records    = []

    def _hms_to_sec(s):
        parts = s.strip().split(":")
        try:
            if len(parts) == 3:
                return int(parts[0])*3600 + int(parts[1])*60 + float(parts[2])
            if len(parts) == 2:
                return int(parts[0])*60 + float(parts[1])
            return float(parts[0])
        except ValueError:
            return 0.0

    if has_header and delim in data_lines[0]:
        header    = [h.strip().lower() for h in data_lines[0].split(delim)]
        start_idx = next((i for i, h in enumerate(header)
                          if "start" in h or "onset" in h), None)
        dur_idx   = next((i for i, h in enumerate(header)
                          if "duration" in h), None)
        label_idx = next((i for i, h in enumerate(header)
                          if any(k in h for k in
                                 ("stage", "description", "annotation", "event"))), 0)
        onset_sec = 0.0
        for ln in data_lines[1:]:
            cols = ln.split(delim)
            if len(cols) <= label_idx:
                continue
            label = cols[label_idx].strip()
            if start_idx is not None and start_idx < len(cols):
                val = cols[start_idx].strip()
                onset_sec = _hms_to_sec(val) if ":" in val else float(val)
            dur = 30.0
            if dur_idx is not None and dur_idx < len(cols):
                try:
                    dur = float(cols[dur_idx].strip())
                except ValueError:
                    pass
            records.append((onset_sec, dur, label))
            onset_sec += dur
    else:
        onset_sec = 0.0
        for ln in data_lines:
            records.append((onset_sec, 30.0, ln.strip()))
            onset_sec += 30.0

    if logger:
        logger.info(f"  Parsed {len(records)} HMC annotations from: {ann_path}")
    return records


# ─────────────────────────────────────────────────────────────────────────────
#  WFDB helpers  (SLPDB)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_slpdb_metadata(record_path: str, config: dict,
                             logger: logging.Logger) -> dict:
    import wfdb
    logger.info(f"[SLPDB] Reading header: {record_path}")
    record   = wfdb.rdrecord(record_path)
    ann      = wfdb.rdann(record_path, "st")
    fs       = record.fs
    n_samp   = record.sig_len
    dur_hr   = round(n_samp / fs / 3600, 3)
    channels = record.sig_name
    unique_labels = list(set(ann.aux_note))
    # Try to get recording start time
    rec_start = None
    bt = getattr(record, "base_datetime", None) or getattr(record, "base_time", None)
    if bt is not None and hasattr(bt, "strftime"):
        rec_start = bt.strftime("%H:%M:%S")

    return {
        "original_fs"    : float(fs),
        "target_fs"      : float(config.get("target_fs", 125)),
        "n_samples"      : int(n_samp),
        "duration_hr"    : dur_hr,
        "channels"       : channels,
        "ecg_channel_idx": config.get("ecg_channel", 0),
        "ann_labels"     : unique_labels,
        "n_annotations"  : len(ann.sample),
        "recording_start": rec_start,
    }


def _extract_hmc_metadata(base_path: str, config: dict,
                           logger: logging.Logger) -> dict:
    edf_path = base_path + ".edf"
    logger.info(f"[HMC] Reading EDF header: {edf_path}")
    keywords = config.get("hmc_ecg_keywords", ["ECG", "EKG"])
    signal, fs, channel_names, ecg_idx, rec_start = load_edf_signal(
        edf_path, keywords, logger
    )
    n_samp   = len(signal)
    dur_hr   = round(n_samp / fs / 3600, 3)
    ann_path = _get_hmc_ann_path(base_path)
    ann_records  = parse_hmc_annotations(ann_path, logger) if ann_path else []
    unique_labels = list({r[2] for r in ann_records})

    return {
        "original_fs"    : float(fs),
        "target_fs"      : float(config.get("target_fs", 125)),
        "n_samples"      : int(n_samp),
        "duration_hr"    : dur_hr,
        "channels"       : channel_names,
        "ecg_channel_idx": ecg_idx,
        "ann_labels"     : unique_labels,
        "n_annotations"  : len(ann_records),
        "recording_start": rec_start,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Public: extract & store metadata for all registered records
# ─────────────────────────────────────────────────────────────────────────────

def extract_all_metadata(config: dict,
                         logger: logging.Logger,
                         output_root: str = "results",
                         db_path: str = DB_PATH) -> None:
    """
    For every record registered in SQLite:
      1. Read signal header
      2. Upsert into record_metadata table
      3. Write record_info.json to per-subject output dir
    """
    dataset    = config.get("dataset", "slpdb")
    all_records = get_all_records(dataset, db_path)

    if not all_records:
        logger.warning("No records found in DB — run p0_register_records first.")
        return

    logger.info(f"Extracting metadata for {len(all_records)} {dataset} records...")

    base_dir = (config["local_slpdb_path"] if dataset == "slpdb"
                else config["local_hmc_path"])

    for row in all_records:
        rec_name  = row["record_name"]
        record_id = row["record_id"]
        base_path = os.path.join(base_dir, rec_name)

        try:
            if dataset == "slpdb":
                meta = _extract_slpdb_metadata(base_path, config, logger)
            else:
                meta = _extract_hmc_metadata(base_path, config, logger)

            upsert_metadata(record_id, meta, db_path)

            # Save JSON for backward compatibility
            from phase0_dataset_management.p0_config import make_subject_dirs
            out_dir = make_subject_dirs(rec_name, output_root, dataset)
            info_path = os.path.join(out_dir, "record_info.json")
            with open(info_path, "w", encoding="utf-8") as f:
                json.dump({"dataset": dataset, "record": rec_name, **meta},
                          f, indent=2, default=str)
            logger.info(f"  [{rec_name}] metadata saved → {info_path}")

        except Exception as exc:
            logger.error(f"  [{rec_name}] metadata extraction FAILED: {exc}")
            mark_phase_failed(record_id, phase=0, error=str(exc), db_path=db_path)


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from phase0_dataset_management.p0_config  import CONFIG
    from phase0_dataset_management.p0_logging import setup_logger
    logger = setup_logger("metadata", log_dir="logs")
    extract_all_metadata(CONFIG, logger)

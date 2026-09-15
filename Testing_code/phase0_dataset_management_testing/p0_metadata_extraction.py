"""
=============================================================================
phase0_dataset_management_testing/p0_metadata_extraction.py
DEVICE-ONLY metadata extraction. Reads the chunk manifest for each
registered session, validates fs consistency / gaps, and stores extracted
metadata in the record_metadata SQLite table. slpdb/hmc (WFDB/EDF) metadata
extraction has been intentionally removed from this testing package.

Also saves a per-session record_info.json to the output directory for
backward compatibility with downstream phases.
=============================================================================
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict

import numpy as np

from Database.db_manager import (
    get_all_records, upsert_metadata, mark_phase_failed, DB_PATH
)


# ─────────────────────────────────────────────────────────────────────────────
#  DEVICE helpers  (JSON chunk sessions — no ground truth)
# ─────────────────────────────────────────────────────────────────────────────

def _fs_from_chunk(meta: dict) -> Optional[float]:
    """fs implied by one chunk's sample_count / duration_s."""
    try:
        sc, dur = meta.get("sample_count"), meta.get("duration_s")
        if not sc or not dur:
            return None
        return sc / dur
    except (TypeError, ZeroDivisionError):
        return None


def _extract_device_metadata(session_id: str, chunks: List[Dict],
                              config: dict, logger: logging.Logger) -> dict:
    """
    chunks: ordered (by window_start_ms) chunk-meta dicts from
    build_device_manifest() — path + timing/count fields only, no ECG arrays.

    Validates per-chunk fs consistency and flags inter-chunk gaps, but does
    NOT load the full ECG signal (that happens later, in the Phase 0/1
    loader that actually needs the samples).
    """
    gap_tol    = config.get("device_gap_tolerance_ms", 500)
    fs_tol_pct = config.get("device_fs_tolerance_pct", 2.0)

    per_chunk_fs = []
    gaps         = []

    for i, c in enumerate(chunks):
        fs_c = _fs_from_chunk(c)
        if fs_c:
            per_chunk_fs.append(fs_c)

        if i > 0:
            prev   = chunks[i - 1]
            gap_ms = c["window_start_ms"] - prev["window_end_ms"]
            if gap_ms > gap_tol:
                gaps.append({
                    "after_chunk" : os.path.basename(prev["path"]),
                    "before_chunk": os.path.basename(c["path"]),
                    "gap_ms"      : gap_ms,
                })

    if not per_chunk_fs:
        raise ValueError(
            f"[DEVICE] No valid sample_count/duration_s in any chunk "
            f"for session '{session_id}'"
        )

    fs_median = float(np.median(per_chunk_fs))
    outliers  = [round(f, 2) for f in per_chunk_fs
                if abs(f - fs_median) / fs_median * 100 > fs_tol_pct]
    if outliers and logger:
        logger.warning(
            f"[DEVICE] {session_id}: {len(outliers)}/{len(per_chunk_fs)} "
            f"chunk(s) deviate >{fs_tol_pct}% from median fs "
            f"({fs_median:.2f} Hz): {outliers[:5]}"
        )
    if gaps and logger:
        logger.warning(
            f"[DEVICE] {session_id}: {len(gaps)} gap(s) beyond "
            f"{gap_tol} ms detected between chunks"
        )

    # Pull patient/device fields from the first chunk/session JSON.
    with open(chunks[0]["path"], "r", encoding="utf-8") as f:
        first_full = json.load(f)
    first_record = first_full[0] if isinstance(first_full, list) and first_full else first_full
    device_fields = {}
    for k in ("patientId", "patientName", "admissionId", "facilityId",
             "deviceId", "age", "gender", "assignedDoctor",
             "firmware_version", "rhythmType"):
        if isinstance(first_record, dict) and k in first_record:
            device_fields[f"device_{k}"] = first_record.get(k)

    first_start_ms = chunks[0]["window_start_ms"]
    last_end_ms    = chunks[-1]["window_end_ms"]
    span_hr        = round((last_end_ms - first_start_ms) / 1000 / 3600, 4)
    summed_hr      = round(sum(c["duration_s"] for c in chunks) / 3600, 4)
    n_samples_total = int(sum(c["sample_count"] for c in chunks))

    rec_start = None
    if first_start_ms:
        rec_start = datetime.fromtimestamp(first_start_ms / 1000).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    logger.info(
        f"  [DEVICE] {session_id}: {len(chunks)} chunks, fs≈{fs_median:.2f} Hz, "
        f"span={span_hr:.3f}h (summed={summed_hr:.3f}h), {len(gaps)} gap(s)"
    )

    return {
        "original_fs"       : fs_median,
        "target_fs"         : float(config.get("target_fs", 125)),
        "n_samples"         : n_samples_total,
        "duration_hr"       : span_hr,          # includes gaps
        "duration_hr_summed": summed_hr,        # excludes gaps
        "channels"          : [config.get("device_ecg_key", "ECG_CH_A")],
        "ecg_channel_idx"   : 0,
        "ann_labels"        : [],               # device format has no ground truth
        "n_annotations"     : 0,
        "recording_start"   : rec_start,
        "n_chunks"          : len(chunks),
        "n_gaps"            : len(gaps),
        "gaps"              : gaps,
        **device_fields,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Public: extract & store metadata for all registered sessions
# ─────────────────────────────────────────────────────────────────────────────

def extract_all_metadata(config: dict,
                         logger: logging.Logger,
                         output_root: str = "results_test",
                         db_path: str = DB_PATH) -> None:
    """
    For every device session registered in SQLite:
      1. Read the chunk manifest (fs / gap validation, no full signal load)
      2. Upsert into record_metadata table
      3. Write record_info.json to per-subject output dir
    """
    all_records = get_all_records("device", db_path)

    if not all_records:
        logger.warning("No records found in DB — run p0_register_records first.")
        return

    logger.info(f"Extracting metadata for {len(all_records)} device session(s)...")

    manifest = config.get("_device_manifest")
    if manifest is None:
        from .p0_scan_datasets import build_device_manifest
        manifest = build_device_manifest(config, logger)
        config["_device_manifest"] = manifest

    for row in all_records:
        rec_name  = row["record_name"]
        record_id = row["record_id"]

        try:
            chunks = manifest.get(rec_name, [])
            if not chunks:
                raise ValueError(f"No chunks found for device session '{rec_name}'")
            meta = _extract_device_metadata(rec_name, chunks, config, logger)

            upsert_metadata(record_id, meta, db_path)

            from .p0_config import make_subject_dirs
            out_dir = make_subject_dirs(rec_name, output_root, "device")
            info_path = os.path.join(out_dir, "record_info.json")
            with open(info_path, "w", encoding="utf-8") as f:
                json.dump({"dataset": "device", "record": rec_name, **meta},
                          f, indent=2, default=str)
            logger.info(f"  [{rec_name}] metadata saved → {info_path}")

        except Exception as exc:
            logger.error(f"  [{rec_name}] metadata extraction FAILED: {exc}")
            mark_phase_failed(record_id, phase=0, error=str(exc), db_path=db_path)


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from .p0_config import CONFIG
    from .p0_logging import setup_logger
    logger = setup_logger("metadata", log_dir="logs")
    extract_all_metadata(CONFIG, logger, db_path="test_pipeline.db")

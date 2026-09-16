"""
=============================================================================
phase0_dataset_management_testing/p0_scan_datasets.py
DEVICE-ONLY scanner. Scans local device JSON chunk folders and groups them
into sessions. slpdb/hmc scanning has been intentionally removed from this
testing package.

Accepted JSON structure (the only ECG chunk format this pipeline supports —
confirmed identical across every device/firmware seen so far, e.g. ezbelt
session ADM1094607798.json and ezflex session ADM1345459698.json):

    [
      {
        "utcTimestamp": {"$date": "2026-09-04T16:51:59.195Z"},
        "admissionId":  "ADM1345459698",
        "packetNo":     1,
        "_id":          {"$oid": "..."},
        "facilityId":   "CF1315821527",
        "value":        [[<ecg sample>, <ecg sample>, ...]]
      },
      ...
    ]

    i.e. one JSON file holding a LIST of packet records for the same
    admissionId, each with a nested value[0] ECG sample array. A "record"
    here is therefore a SESSION (chunk files are grouped/sorted by
    `device_id_key`, default "admissionId", and by window_start_ms — a
    session may span multiple files/hour-folders, or be a single file).

    Non-ECG device files (alerts — {"category": "alert", "streamAlert": ...},
    SPO2_UNFILTERED_*, NISO101_*, etc.) may sit in the same folders. They
    don't match the structure above (no admissionId+value list) and are
    silently skipped — see _read_device_chunk_meta / build_device_manifest.
=============================================================================
"""

import os
import glob
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional

from .p0_config import get_active_device_path


def _parse_iso8601_ms(value) -> Optional[int]:
    """Parse either ISO-8601 timestamps or numeric millisecond values."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return int(float(s))
        except ValueError:
            pass
        try:
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is not None:
                return int(dt.timestamp() * 1000)
            return int(dt.timestamp() * 1000)
        except ValueError:
            return None
    if isinstance(value, dict):
        value = value.get("$date")
        return _parse_iso8601_ms(value)
    return None


def _coerce_value_array(value):
    """Return the numeric ECG payload from one packet record's "value" field."""
    if value is None:
        return []
    if isinstance(value, list):
        if not value:
            return []
        if isinstance(value[0], list):
            first = value[0]
            return list(first)
        if isinstance(value[0], (int, float)):
            return list(value)
    return [value] if isinstance(value, (int, float)) else []


def _is_packetized_json(payload) -> bool:
    """True if payload matches the accepted device ECG structure (see module
    docstring): a non-empty list of dict records, each with an admissionId
    and a nested value[] ECG sample array."""
    if not isinstance(payload, list) or not payload:
        return False
    first = payload[0]
    if not isinstance(first, dict):
        return False
    if "admissionId" not in first or "value" not in first:
        return False
    value = first.get("value")
    if not isinstance(value, list) or not value:
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def scan_datasets(config: dict,
                  logger: logging.Logger = None) -> List[str]:
    """
    Discover all device sessions. Returns a sorted list of session ids.
    The chunk manifest is cached onto config["_device_manifest"] for
    downstream phases to reuse without re-scanning the disk.

    Raises FileNotFoundError if the folder does not exist.
    Raises RuntimeError if no sessions are found.
    """
    return _scan_device(config, logger)


# ─────────────────────────────────────────────────────────────────────────────
#  DEVICE scanner — groups many small JSON chunks into sessions
# ─────────────────────────────────────────────────────────────────────────────

def _read_device_chunk_meta(path: str, id_key: str) -> Optional[Dict]:
    """
    Read only the small scalar fields of one device JSON chunk/session.

    Only the packetized structure is accepted (see module docstring): a
    JSON file holding a list of packet records, each with utcTimestamp and
    a nested value[] ECG packet array for the same admissionId. Anything
    else (a bare dict, an alert file, an empty/malformed list, etc.)
    returns None and is skipped by the caller — this pipeline has never
    seen a real device file in any other shape.
    """
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)

    if _is_packetized_json(d):
        samples = 0
        timestamps = []
        for record in d:
            if not isinstance(record, dict):
                continue
            value = record.get("value")
            packet = _coerce_value_array(value)
            if packet:
                samples += len(packet)
            ts = _parse_iso8601_ms(record.get("utcTimestamp"))
            if ts is not None:
                timestamps.append(ts)

        if not timestamps or samples == 0:
            return None

        start_ms = min(timestamps)
        end_ms = max(timestamps)
        duration_s = max((end_ms - start_ms) / 1000.0, 1.0 / 125.0)
        return {
            "path"           : path,
            "session_id"     : str(d[0].get(id_key, "UNKNOWN")),
            "window_start_ms": start_ms,
            "window_end_ms"  : end_ms,
            "duration_s"     : duration_s,
            "sample_count"   : int(samples),
        }

    return None


def build_device_manifest(config: dict,
                          logger: logging.Logger = None) -> Dict[str, List[Dict]]:
    """
    Group all device JSON chunk files under the active device path (test or
    production, via get_active_device_path) by session id
    (config["device_id_key"], default "admissionId"), sorted per-session by
    window_start_ms.

    Returns
    -------
    {session_id: [chunk_meta_dict, ...]}   — chunk_meta has no ECG arrays,
    just path + timing/count fields (see _read_device_chunk_meta).
    """
    path       = get_active_device_path(config)
    path_key   = ("local_test_device_path" if config.get("use_test_device_path")
                  else "local_device_path")
    _check_dir(path, "DEVICE", path_key)
    id_key  = config.get("device_id_key", "admissionId")
    pattern = config.get("device_file_glob", "ECG_*.json")

    files = set(glob.glob(os.path.join(path, "**", pattern), recursive=True))
    files |= set(glob.glob(os.path.join(path, "**", "*.json"), recursive=True))
    files = sorted(files)

    if logger:
        logger.info(f"[DEVICE] Scanned: {path}")
        logger.info(f"  Found {len(files)} JSON file(s) checked for device payloads '{pattern}'")

    sessions  = {}
    bad_files = []
    for fp in files:
        try:
            meta = _read_device_chunk_meta(fp, id_key)
            if meta is None:
                continue
            if meta["window_start_ms"] is None or meta["sample_count"] is None:
                bad_files.append(fp)
                continue
            sessions.setdefault(meta["session_id"], []).append(meta)
        except Exception as exc:
            bad_files.append(fp)
            if logger:
                logger.warning(f"  Could not parse device chunk '{fp}': {exc}")

    for sid in sessions:
        sessions[sid] = sorted(sessions[sid], key=lambda m: m["window_start_ms"])

    if logger:
        logger.info(f"  Grouped into {len(sessions)} session(s) "
                    f"(keyed by '{id_key}')")
        for sid, chunks in sessions.items():
            logger.info(f"    {sid}: {len(chunks)} chunk(s)")
        if bad_files:
            logger.warning(
                f"  {len(bad_files)} file(s) unreadable / missing required "
                f"fields, skipped: {bad_files[:5]}{'...' if len(bad_files) > 5 else ''}"
            )

    return sessions


def get_device_chunk_files(config: dict, session_id: str,
                           logger: logging.Logger = None) -> List[Dict]:
    """
    Convenience accessor: return the ordered chunk-meta list for one session,
    reusing config["_device_manifest"] if already cached, otherwise building
    it on demand.
    """
    manifest = config.get("_device_manifest")
    if manifest is None:
        manifest = build_device_manifest(config, logger)
        config["_device_manifest"] = manifest
    return manifest.get(session_id, [])


def _scan_device(config: dict, logger) -> List[str]:
    sessions = build_device_manifest(config, logger)
    if not sessions:
        raise RuntimeError(
            f"No valid device JSON chunks found in: {get_active_device_path(config)}\n"
            f"Expected JSON files holding a list of packet records, each with "
            f"'{config.get('device_id_key', 'admissionId')}' and a nested "
            f"'value' ECG sample array (see p0_scan_datasets.py module docstring)."
        )
    config["_device_manifest"] = sessions      # cache for Phase 0/1 loaders
    return sorted(sessions.keys())


# ─────────────────────────────────────────────────────────────────────────────
#  Helper
# ─────────────────────────────────────────────────────────────────────────────

def _check_dir(path: str, label: str, config_key: str) -> None:
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"{label} folder not found: {path}\n"
            f"Update CONFIG['{config_key}'] in p0_config.py."
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from .p0_config import CONFIG
    from .p0_logging import setup_logger

    logger = setup_logger("scan", log_dir="logs")
    try:
        recs = scan_datasets(CONFIG, logger)
        print(f"device: {len(recs)} session(s) — {recs}")
    except Exception as exc:
        print(f"device: {exc}")
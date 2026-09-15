"""
=============================================================================
phase0_dataset_management_testing/p0_scan_datasets.py
DEVICE-ONLY scanner. Scans local device JSON chunk folders and groups them
into sessions. slpdb/hmc scanning has been intentionally removed from this
testing package.

Layout supported (any folder nesting):
    device_json_or_test_dataset/
        <any nesting, e.g. ADM937394258/2026-06-30_04/>/
            ECG_ADM937394258_035937.json   ← ~30 s chunk, has admissionId,
            ECG_ADM937394258_040007.json      window_start_ms/window_end_ms,
            ...                                duration_s, sample_count

    Other sensor files (SPO2_UNFILTERED_*, NISO101_*, *_data_*) may sit in
    the same folders — device_file_glob="ECG_*.json" excludes them.

    Chunk files are grouped into one "record" (session) by `device_id_key`
    (default "admissionId") and sorted by window_start_ms. A record here
    is therefore a SESSION (potentially spanning many hour-folders), not a
    single file.
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
    """Return the numeric ECG payload from a legacy dict or a new packet entry."""
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


def _is_new_packetized_json(payload) -> bool:
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

    Supports both the legacy chunk format (single ECG_*.json dict with
    admissionId/window_start_ms/... fields) and the newer packetized format
    (one JSON file with a list of records, each containing utcTimestamp and
    a nested value[] ECG packet array for the same admissionId).
    """
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)

    if isinstance(d, dict):
        return {
            "path"           : path,
            "session_id"     : str(d.get(id_key, "UNKNOWN")),
            "window_start_ms": d.get("window_start_ms"),
            "window_end_ms"  : d.get("window_end_ms"),
            "duration_s"     : d.get("duration_s"),
            "sample_count"   : d.get("sample_count"),
        }

    if _is_new_packetized_json(d):
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
            f"Expected files matching '{config.get('device_file_glob', 'ECG_*.json')}' "
            f"containing '{config.get('device_id_key', 'admissionId')}' "
            f"and 'window_start_ms'/'sample_count' fields."
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

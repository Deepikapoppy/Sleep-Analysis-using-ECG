"""
=============================================================================
phase0_dataset_management_testing/p0_register_records.py
DEVICE-ONLY registration. Takes discovered sessions and writes them into
SQLite. slpdb/hmc registration has been intentionally removed from this
testing package.

A device "record" is a session made of many chunk files (no single
signal_file/annotation_file pair). Since there's no single signal_file per
session, the ordered chunk-path list is stored as a JSON string in the
signal_file column. annotation_file is always None — this format never
carries ground truth.

NOTE: registering into SQLite at all is OPTIONAL for testing — 
run_phase0_record() already handles db_row is None gracefully (record_id
stays None, DB writes are skipped). Call this only if you want sessions
tracked for traceability. Always pass an explicit db_path (e.g.
"test_pipeline.db") to keep test runs isolated from your production DB.
=============================================================================
"""

import os
import json
import logging
from typing import List

from Database.create_database import create_database, DB_PATH
from Database.db_manager import (
    register_dataset,
    register_record,
    update_dataset_record_count,
)

from .p0_config import get_active_device_path


def register_all_records(config: dict,
                         records: List[str],
                         logger: logging.Logger = None,
                         db_path: str = DB_PATH) -> int:
    """
    Register device sessions into SQLite.

    Parameters
    ----------
    config  : CONFIG dict
    records : list of session id strings from p0_scan_datasets
    logger  : optional logger
    db_path : path to .db file — pass an isolated test path explicitly

    Returns
    -------
    dataset_id : int
    """
    conn = create_database(db_path)
    conn.close()

    base_dir   = get_active_device_path(config)
    dataset_id = register_dataset("device", "json_chunks", base_dir, db_path)
    if logger:
        logger.info(f"Dataset 'device' → dataset_id={dataset_id}")

    manifest = config.get("_device_manifest")
    if manifest is None:
        from .p0_scan_datasets import build_device_manifest
        manifest = build_device_manifest(config, logger)
        config["_device_manifest"] = manifest

    registered = 0
    skipped    = 0
    for session_id in records:
        chunks      = manifest.get(session_id, [])
        chunk_paths = [c["path"] for c in chunks]
        signal_file = json.dumps(chunk_paths)   # ordered list, as text

        record_id = register_record(
            dataset_id      = dataset_id,
            record_name     = session_id,
            signal_file     = signal_file,
            annotation_file = None,
            db_path         = db_path,
        )
        if record_id:
            registered += 1
        else:
            skipped += 1

    update_dataset_record_count(dataset_id, len(records), db_path)

    if logger:
        logger.info(
            f"Registration complete: {registered} inserted, "
            f"{skipped} already existed  |  dataset_id={dataset_id}"
        )

    return dataset_id


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from .p0_config import CONFIG
    from .p0_logging import setup_logger
    from .p0_scan_datasets import scan_datasets

    logger  = setup_logger("register", log_dir="logs")
    records = scan_datasets(CONFIG, logger)
    did     = register_all_records(CONFIG, records, logger, db_path="test_pipeline.db")
    print(f"dataset_id = {did},  {len(records)} sessions registered.")

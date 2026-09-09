"""
=============================================================================
phase0_dataset_management/p0_register_records.py
Takes the list of discovered records and writes them into SQLite:
  - ensures datasets row exists
  - inserts each record into records table
  - creates a blank processing_status row for every record
=============================================================================
"""

import os
import sys
import logging
from typing import List

# Allow running as a standalone script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Database.create_database import create_database, DB_PATH
from Database.db_manager import (
    register_dataset,
    register_record,
    update_dataset_record_count,
)


_DATASET_FORMATS = {
    "slpdb": "wfdb",
    "hmc"  : "edf",
}

_SIGNAL_EXT = {
    "slpdb": ".hea",
    "hmc"  : ".edf",
}

_ANN_EXT = {
    "slpdb": ".st",
    "hmc"  : "_sleepscoring.txt",
}


def register_all_records(config: dict,
                         records: List[str],
                         logger: logging.Logger = None,
                         db_path: str = DB_PATH) -> int:
    """
    Register a list of record names into SQLite.

    Parameters
    ----------
    config  : CONFIG dict (needs dataset, local_*_path keys)
    records : list of record name strings from p0_scan_datasets
    logger  : optional logger
    db_path : path to .db file

    Returns
    -------
    dataset_id : int
    """
    # Ensure schema exists
    conn = create_database(db_path)
    conn.close()

    dataset  = config.get("dataset", "slpdb")
    fmt      = _DATASET_FORMATS.get(dataset, "unknown")
    base_dir = (config["local_slpdb_path"] if dataset == "slpdb"
                else config["local_hmc_path"])
    sig_ext  = _SIGNAL_EXT[dataset]
    ann_ext  = _ANN_EXT[dataset]

    # ── 1. Register dataset ──────────────────────────────────────────────────
    dataset_id = register_dataset(dataset, fmt, base_dir, db_path)
    if logger:
        logger.info(f"Dataset '{dataset}' → dataset_id={dataset_id}")

    # ── 2. Register each record ───────────────────────────────────────────────
    registered = 0
    skipped    = 0
    for rec_name in records:
        signal_file = os.path.join(base_dir, rec_name + sig_ext)
        # For HMC try both annotation suffixes
        if dataset == "hmc":
            ann_txt   = os.path.join(base_dir, rec_name + "_sleepscoring.txt")
            ann_notxt = os.path.join(base_dir, rec_name + "_sleepscoring")
            ann_file  = ann_txt if os.path.exists(ann_txt) else ann_notxt
        else:
            ann_file = os.path.join(base_dir, rec_name + ann_ext)
            if not os.path.exists(ann_file):
                ann_file = None

        record_id = register_record(
            dataset_id  = dataset_id,
            record_name = rec_name,
            signal_file = signal_file,
            annotation_file = ann_file,
            db_path     = db_path,
        )
        if record_id:
            registered += 1
        else:
            skipped += 1

    # ── 3. Update total count ─────────────────────────────────────────────────
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
    from phase0_dataset_management.p0_config  import CONFIG
    from phase0_dataset_management.p0_logging import setup_logger
    from phase0_dataset_management.p0_scan_datasets import scan_datasets

    logger  = setup_logger("register", log_dir="logs")
    records = scan_datasets(CONFIG, logger)
    did     = register_all_records(CONFIG, records, logger)
    print(f"dataset_id = {did},  {len(records)} records registered.")

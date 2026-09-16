"""
=============================================================================
run_phase0_test.py
Run this file directly (python run_phase0_test.py) from the folder that
CONTAINS phase0_dataset_management_testing/ (i.e. your project root).

This exercises the full Phase 0 pipeline (scan -> register -> metadata ->
inspect/plot) against whichever path CONFIG points at, using the
use_test_device_path toggle so you never touch your production device path.

Database: every step below is passed CONFIG["db_path"] explicitly (see
p0_config.py — TEST_DB_PATH / CONFIG["db_path"], defaults to
"test_pipeline.db"). This is deliberate: register_all_records(),
extract_all_metadata() and run_phase0_record() all have function defaults
too, but those defaults must never be relied on here — passing db_path
explicitly at every call site is the only way to guarantee test runs can
never silently fall through to your production Sleepanalysis.db.

Why not just run the individual p0_*.py files directly?
Because phase0_dataset_management_testing/ uses package-relative imports
(from .p0_config import ...) so its modules can be renamed/copied freely
without accidentally importing your production phase0_dataset_management
folder by name. Relative imports only work when the file is run AS PART OF
THE PACKAGE, which this driver does correctly (running p0_scan_datasets.py
directly with `python p0_scan_datasets.py` will fail with "attempted
relative import with no known parent package" — that's expected, use this
driver or `python -m phase0_dataset_management_testing.p0_scan_datasets`
instead).
=============================================================================
"""

import os
import sys

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
TESTING_ROOT = os.path.dirname(PACKAGE_DIR)
PROJECT_ROOT = os.path.dirname(TESTING_ROOT)

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, TESTING_ROOT)

from phase0_dataset_management_testing import (
    CONFIG, scan_datasets, register_all_records, extract_all_metadata,
    run_phase0_record, make_subject_dirs, setup_logger,
)

if __name__ == "__main__":
    logger  = setup_logger("phase0_test", log_dir="logs")
    db_path = CONFIG["db_path"]

    print(f"dataset      = {CONFIG['dataset']}")
    print(f"use_test_device_path = {CONFIG.get('use_test_device_path')}")
    print(f"db_path      = {db_path}")

    records = scan_datasets(CONFIG, logger)
    print(f"Records/sessions found: {records}")

    # Registration is optional for device/test data — run_phase0_record()
    # handles db_row is None gracefully. Comment this out if you don't
    # want SQLite touched at all during testing.
    register_all_records(CONFIG, records, logger, db_path=db_path)

    for rec in records:
        CONFIG["record_name"] = rec
        CONFIG["output_dir"]  = make_subject_dirs(
            rec, CONFIG["results_dir"], CONFIG["dataset"]
        )
        run_phase0_record(CONFIG, logger, db_path=db_path)
        print(f"Done: {rec} -> {CONFIG['output_dir']}")

    # Metadata extraction (fs/gap validation, record_metadata table,
    # record_info.json per session) — reads whatever was just registered
    # above, so it must run after the registration step, against the same
    # db_path.
    extract_all_metadata(CONFIG, logger, output_root=CONFIG["results_dir"],
                          db_path=db_path)
"""
=============================================================================
run_phase1_test.py
Run this file directly (python run_phase1_test.py) from your project root —
the same folder that contains phase0_dataset_management_testing/ and
phase1_preprocessing_testing/.

Chains: scan (Phase 0 testing) -> register (optional) -> Phase 1 full
pipeline, for every device session found under CONFIG's active device path
(local_test_device_path if use_test_device_path=True, else local_device_path).

Uses the SAME phase0_dataset_management_testing.CONFIG for both phases —
Phase 1 needs every device_* key (device_ecg_key, device_id_key,
device_gap_tolerance_ms, etc.) that Phase 0 already defines, plus
target_fs/epoch_sec/dwt_wavelet/dwt_level, all of which are already there.

Why a driver script and not `python p1_pipeline.py` directly?
Same reason as run_phase0_test.py — phase1_preprocessing_testing/ uses
package-relative imports (from .p1_logging import ...), which only resolve
when the file is run AS PART OF THE PACKAGE. This driver does that correctly.
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
    CONFIG, scan_datasets, register_all_records, make_subject_dirs,
)
from phase1_preprocessing_testing.p1_pipeline import run_phase1_record
from phase1_preprocessing_testing.p1_logging   import setup_logger

TEST_DB_PATH = "test_pipeline.db"   # isolated from your production DB

if __name__ == "__main__":
    logger = setup_logger("phase1_test", log_dir="logs", log_file="phase1_test.log")

    print(f"dataset               = {CONFIG['dataset']}")
    print(f"use_test_device_path  = {CONFIG.get('use_test_device_path')}")

    records = scan_datasets(CONFIG, logger)
    print(f"Sessions found: {records}")

    # Optional — skip this line entirely if you don't want SQLite touched
    # during testing. run_phase1_record() already handles db_row is None.
    register_all_records(CONFIG, records, logger, db_path=TEST_DB_PATH)

    for session_id in records:
        CONFIG["record_name"] = session_id
        CONFIG["output_dir"]  = make_subject_dirs(
            session_id, CONFIG["results_dir"], CONFIG["dataset"]
        )
        meta = run_phase1_record(CONFIG, logger, db_path=TEST_DB_PATH)
        print(f"Done: {session_id} -> {CONFIG['output_dir']}  "
             f"({meta['n_epochs']} epochs, {meta['pct_bad']}% bad)")

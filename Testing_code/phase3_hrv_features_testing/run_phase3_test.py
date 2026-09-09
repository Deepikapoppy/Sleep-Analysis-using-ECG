"""
=============================================================================
run_phase3_test.py
Run this file directly (python run_phase3_test.py) from your project root —
the folder that contains phase0_dataset_management_testing/,
phase1_preprocessing_testing/, phase2_rpeak_rr_testing/, and
phase3_hrv_features_testing/.

Requires Phase 2 to have already run for each session (run_phase2_test.py) —
Phase 3 loads phase2_rr_full.json / phase1_meta.json / phase1_sqi.csv /
preprocessed_epochs.npy from config["output_dir"]; it does not re-run
Phase 1 or Phase 2.

Chains: scan (Phase 0 testing) -> Phase 3 full pipeline (skip-and-continue
on failure, e.g. a session Phase 2 never completed for).

All outputs land under results_test/device/plots/per_subject/<session>/
alongside the Phase 1/2 files already there. SQLite writes (optional,
phase3_done=1 + summary columns) go to the isolated test_pipeline.db,
never your production DB.
=============================================================================
"""

import os
import sys

# Make the project root, the Testing_code package root, and the current test
# package directory importable when this script is run directly. This allows
# sibling test packages and the main Database package to resolve correctly.
TESTING_ROOT = os.path.dirname(os.path.abspath(__file__))
TESTING_CODE_ROOT = os.path.dirname(TESTING_ROOT)
PROJECT_ROOT = os.path.dirname(TESTING_CODE_ROOT)
for path in (PROJECT_ROOT, TESTING_CODE_ROOT, TESTING_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from phase0_dataset_management_testing import CONFIG, scan_datasets, make_subject_dirs
from phase3_hrv_features_testing.p3_pipeline import run_phase3_record
from phase3_hrv_features_testing.p3_logging   import setup_logger

TEST_DB_PATH = "test_pipeline.db"   # isolated from your production DB

if __name__ == "__main__":
    logger = setup_logger("phase3_test", log_dir="logs", log_file="phase3_test.log")

    print(f"dataset               = {CONFIG['dataset']}")
    print(f"use_test_device_path  = {CONFIG.get('use_test_device_path')}")

    records = scan_datasets(CONFIG, logger)
    print(f"Sessions found: {records}")

    for session_id in records:
        CONFIG["record_name"] = session_id
        CONFIG["output_dir"]  = make_subject_dirs(
            session_id, CONFIG["results_dir"], CONFIG["dataset"]
        )
        try:
            metrics = run_phase3_record(CONFIG, logger, db_path=TEST_DB_PATH)
            print(f"Done: {session_id} -> {CONFIG['output_dir']}  "
                 f"({metrics['n_epochs']} epochs, "
                 f"T1={metrics['n_t1_cols']} T2={metrics['n_t2_cols']} "
                 f"T3={metrics['n_t3_cols']} z-scores={metrics['n_zscore_cols']})")
        except Exception as exc:
            # e.g. a session whose Phase 2 never completed
            print(f"SKIPPED: {session_id} -> {exc}")
            continue

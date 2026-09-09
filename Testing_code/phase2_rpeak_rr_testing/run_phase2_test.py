"""
=============================================================================
run_phase2_test.py
Run this file directly (python run_phase2_test.py) from your project root —
the folder that contains phase0_dataset_management_testing/,
phase1_preprocessing_testing/, and phase2_rpeak_rr_testing/.

Requires Phase 1 to have already run for each session (run_phase1_test.py) —
Phase 2 loads preprocessed_epochs.npy / bad_epoch_mask.npy / phase1_meta.json
from config["output_dir"], it does not touch raw ECG or re-run Phase 1.

Chains: scan (Phase 0 testing) -> Phase 2 full pipeline (skip-and-continue
on failure, e.g. a session Phase 1 never completed for) -> consolidated CSV.
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
from phase2_rpeak_rr_testing.p2_pipeline import run_phase2_record
from phase2_rpeak_rr_testing.p2_logging   import setup_logger
from phase2_rpeak_rr_testing.p2_consolidated_csv import build_consolidated_csv

TEST_DB_PATH = "test_pipeline.db"   # isolated from your production DB

if __name__ == "__main__":
    logger = setup_logger("phase2_test", log_dir="logs", log_file="phase2_test.log")

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
            metrics = run_phase2_record(CONFIG, logger, db_path=TEST_DB_PATH)
            print(f"Done: {session_id} -> {CONFIG['output_dir']}  "
                 f"(mean HR={metrics['mean_hr_overall']} bpm, "
                 f"top method={metrics['top_method']})")
        except Exception as exc:
            # e.g. a session whose Phase 1 never completed (too short, etc.)
            print(f"SKIPPED: {session_id} -> {exc}")
            continue

    print()
    print("Building consolidated CSV across all sessions...")
    df = build_consolidated_csv(CONFIG, logger,
                                output_root=CONFIG["results_dir"],
                                db_path=TEST_DB_PATH)
    print(f"Consolidated: {len(df)} session(s) with completed Phase 2 output.")

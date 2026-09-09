"""
=============================================================================
run_phase5_test.py
Run this file directly (python run_phase5_test.py) from your project root.

Final-report phase for device sessions: takes the RF model's predictions
(from run_phase4b_test.py) and generates hypnogram / distribution / SQI /
HRV-by-stage / final-summary plots, WITHOUT any accuracy or kappa — device
sessions have no PSG ground truth to evaluate against, so that comparison
is fundamentally impossible for this dataset, not merely omitted here.

Requires Phase 4b to have already run for each session (run_phase4b_test.py),
which itself requires Phase 3 (run_phase3_test.py) and Phase 1
(run_phase1_test.py). Phase 3's HRV features and Phase 1's phase1_sqi.csv
are optional — their respective plots are skipped gracefully if absent.
=============================================================================
"""

import os
import sys

TESTING_ROOT = os.path.dirname(os.path.abspath(__file__))
TESTING_ROOT_PARENT = os.path.dirname(TESTING_ROOT)
PROJECT_ROOT = os.path.dirname(TESTING_ROOT_PARENT)
for path in [PROJECT_ROOT, TESTING_ROOT_PARENT, TESTING_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

from phase0_dataset_management_testing import CONFIG, scan_datasets, make_subject_dirs
from phase5_final_report_testing.p5_pipeline import run_phase5_record
from phase5_final_report_testing.p5_logging  import setup_logger

TEST_DB_PATH = "test_pipeline.db"   # isolated from your production DB

if __name__ == "__main__":
    logger = setup_logger("phase5_test", log_dir="logs", log_file="phase5_test.log")

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
            meta = run_phase5_record(CONFIG, logger, db_path=TEST_DB_PATH)
            print(f"Done: {session_id} -> {CONFIG['output_dir']}  "
                 f"({meta['n_epochs']} epochs, "
                 f"stages={meta['stage_minutes_predicted']})")
        except Exception as exc:
            # e.g. a session whose Phase 4b never completed
            print(f"SKIPPED: {session_id} -> {exc}")
            continue

"""
=============================================================================
run_phase4b_test.py
Run this file directly (python run_phase4b_test.py) from your project root.

This is the DEPLOYMENT path — loads your PRETRAINED RF model and predicts
sleep stages for every device test session, directly from Phase 3 features.
It does NOT use phase4_sleep_classification (rule-based) at all.

Requires Phase 3 to have already run for each session (run_phase3_test.py).

*** EDIT MODEL_PATH BELOW before running ***
It must point at the phase4b_rf_model.pkl produced by training
(phase4b_ml_classification/p4b_ml_classify.py), under whichever dataset
(hmc or slpdb) you actually trained on — never "device".
=============================================================================
"""

import os
import sys

TESTING_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTING_ROOT)
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
for path in [REPO_ROOT, PROJECT_ROOT, TESTING_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

from phase0_dataset_management_testing import CONFIG, scan_datasets, make_subject_dirs
from phase4b_ml_classification_testing.p4b_pipeline import run_phase4b_predict_record
from phase4b_ml_classification_testing.p4b_logging   import setup_logger

TEST_DB_PATH = "test_pipeline.db"   # isolated from your production DB

# ── EDIT THIS ────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(
    REPO_ROOT, "results", "hmc", "consolidated", "phase4b_rf_model.pkl"
)
# Change "hmc" to "slpdb" (or your pooled/combined path) if that's what you
# actually trained the deployed model on.
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger = setup_logger("phase4b_test", log_dir="logs", log_file="phase4b_test.log")

    if not os.path.exists(MODEL_PATH):
        print(f"MODEL NOT FOUND: {MODEL_PATH}")
        print("Edit MODEL_PATH at the top of this script before running.")
        sys.exit(1)

    print(f"dataset               = {CONFIG['dataset']}")
    print(f"use_test_device_path  = {CONFIG.get('use_test_device_path')}")
    print(f"model_path            = {MODEL_PATH}")

    records = scan_datasets(CONFIG, logger)
    print(f"Sessions found: {records}")

    for session_id in records:
        CONFIG["record_name"] = session_id
        CONFIG["output_dir"]  = make_subject_dirs(
            session_id, CONFIG["results_dir"], CONFIG["dataset"]
        )
        try:
            result = run_phase4b_predict_record(
                CONFIG, MODEL_PATH, logger, db_path=TEST_DB_PATH
            )
            print(f"Done: {session_id} -> {CONFIG['output_dir']}  "
                 f"({result['n_epochs']} epochs, "
                 f"predicted={result['stage_counts_smooth']})")
        except Exception as exc:
            # e.g. a session whose Phase 3 never completed
            print(f"SKIPPED: {session_id} -> {exc}")
            continue

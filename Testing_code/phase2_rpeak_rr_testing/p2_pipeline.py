"""
=============================================================================
phase2_rpeak_rr_testing/p2_pipeline.py
Master orchestrator for Phase 2, DEVICE sessions only.
slpdb/hmc handling has been intentionally REMOVED from this testing
package — production code for those datasets lives in
phase2_rpeak_rr/p2_pipeline.py. config["record_name"] here is always a
device session id (e.g. "ADM937394258") and config["dataset"] is always
"device". Requires Phase 1 (phase1_preprocessing_testing) to have already
run successfully for the session — reads its .npy/.json outputs from
config["output_dir"].

Steps (unchanged step order/logic from training)
─────────────────────────────────────────────
1  p2_load_phase1      → load epochs, bad_mask, fs, meta from Phase 1 disk
2  p2_rpeak_detection   → calibrated cascade (neurokit → elgendi2010 → scipy)
3  p2_rr_intervals      → RR computation + 2-tier artifact correction
4  p2_process_epochs    → combine into per-epoch result rows
5  p2_plots             → 7 diagnostic plots incl. method comparison
6  p2_save              → .csv / .npy / .json + SQLite phase2_done=1 (optional)
=============================================================================
"""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .p2_logging      import setup_logger, log_phase_header, log_step
from .p2_load_phase1  import (
    load_phase1_outputs, load_raw_ecg_for_viz, load_clean_ecg_for_viz
)
from .p2_process_epochs import process_all_epochs
from .p2_rr_intervals   import build_tachogram
from .p2_plots           import (
    plot_method_comparison,
    plot_rpeak_examples,
    plot_raw_clean_rpeaks,
    plot_tachogram,
    plot_hr_histogram,
    plot_detection_quality,
    plot_method_breakdown,
)
from .p2_save            import save_phase2_results
from Database.db_manager import mark_phase_failed, get_record, DB_PATH


def run_phase2_record(config: dict,
                      logger: logging.Logger,
                      db_path: str = DB_PATH) -> dict:
    """
    Run full Phase-2 pipeline for config['record_name'] (a device session id).

    Returns
    -------
    metrics : dict — same content as phase2_metrics.json
    """
    rec_name  = config["record_name"]
    dataset   = config.get("dataset", "device")

    db_row    = get_record(rec_name, dataset, db_path)   # None if not registered — fine
    record_id = db_row["record_id"] if db_row else None

    try:
        # ── Step 1: Load Phase 1 outputs ──────────────────────────────────────
        log_step(logger, 1, "Load Phase-1 outputs from disk")
        epochs, bad_mask, fs, meta = load_phase1_outputs(config, logger, db_path)
        raw_ecg_ds  = load_raw_ecg_for_viz(config)
        clean_ecg   = load_clean_ecg_for_viz(config)

        log_phase_header(logger, rec_name, dataset, len(epochs), fs)

        # ── Step 2–4: Detect R-peaks + compute RR + process all epochs ───────
        log_step(logger, 2, "R-Peak Detection (3 methods)")
        results, method_counts = process_all_epochs(
            epochs, bad_mask, fs, config, logger
        )

        # ── Step 5: Build tachogram ───────────────────────────────────────────
        log_step(logger, 3, "Build Tachogram")
        tachogram_t, tachogram_rr = build_tachogram(results, config["epoch_sec"])

        # ── Step 6: Plots ─────────────────────────────────────────────────────
        log_step(logger, 4, "Generating Plots")
        plot_method_comparison(epochs, results, fs, config, logger)
        plot_rpeak_examples(epochs, results, fs, config, logger)
        plot_raw_clean_rpeaks(epochs, results, fs, config, logger,
                               raw_ecg_ds=raw_ecg_ds, clean_ecg_full=clean_ecg)
        plot_tachogram(tachogram_t, tachogram_rr, config, logger)
        plot_hr_histogram(results, config, logger)
        plot_detection_quality(results, config, logger)
        plot_method_breakdown(method_counts, config, logger)

        # ── Step 7: Save ──────────────────────────────────────────────────────
        log_step(logger, 5, "Save Outputs + Update SQLite (optional)")
        metrics = save_phase2_results(
            results, tachogram_t, tachogram_rr,
            method_counts, config, logger, meta=meta, db_path=db_path
        )

        logger.info(
            f"✓ Phase 2 complete — {rec_name} | "
            f"valid epochs={metrics['valid_epochs']} | "
            f"mean HR={metrics['mean_hr_overall']} bpm | "
            f"top method={metrics['top_method']}"
        )
        return metrics

    except Exception as exc:
        logger.error(f"✗ Phase 2 FAILED — {rec_name}: {exc}")
        if record_id:
            mark_phase_failed(record_id, 2, str(exc), db_path)
        raise


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from phase0_dataset_management_testing import CONFIG, scan_datasets, make_subject_dirs
    from Database.db_manager import get_all_records

    TEST_DB_PATH = "test_pipeline.db"   # isolated from your production DB

    logger = setup_logger("phase2_test", log_dir="logs", log_file="phase2_test.log")

    # Prefer sessions already registered in the isolated test DB; fall back
    # to a fresh disk scan if nothing's registered yet — same fallback
    # pattern as phase1_preprocessing_testing's __main__.
    records = get_all_records(CONFIG["dataset"], db_path=TEST_DB_PATH)
    session_ids = ([row["record_name"] for row in records] if records
                   else scan_datasets(CONFIG, logger))

    for session_id in session_ids:
        CONFIG["record_name"] = session_id
        CONFIG["output_dir"]  = make_subject_dirs(
            session_id, CONFIG["results_dir"], CONFIG["dataset"]
        )
        try:
            run_phase2_record(CONFIG, logger, db_path=TEST_DB_PATH)
        except Exception as exc:
            # Skip-and-continue for batch runs, same as production's __main__
            # — one bad/too-short session shouldn't halt the whole batch.
            logger.error(f"Skipping {session_id}: {exc}")
            continue

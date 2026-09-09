"""
=============================================================================
phase4_sleep_classification/p4_pipeline.py
Master orchestrator for Phase 4 — Rule-Based Sleep Stage Classification.

Execution order
───────────────
1  p4_load_phase3    → load phase3_hrv_features.csv + phase3_meta.json
2  p4_preprocess     → clip outliers, robust 5–95 pct normalisation
3  p4_classify       → score all epochs, apply time-of-night priors,
                        adaptive Wake boost, softmax → argmax → raw stages
4  p4_classify       → smooth_hypnogram (median filter + min-run merger)
5  p4_plots          → 5 diagnostic plots
6  p4_save           → npy + CSV + JSON + SQLite phase4_done=1

Prerequisites
─────────────
Phase 3 must be complete for the record (phase3_hrv_features.csv must exist).
=============================================================================
"""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase4_sleep_classification.p4_logging   import setup_logger, log_phase_header, log_step
from phase4_sleep_classification.p4_load_phase3 import load_features, load_phase3_meta
from phase4_sleep_classification.p4_preprocess  import clip_outliers, normalise_features
from phase4_sleep_classification.p4_classify    import classify_epochs, smooth_hypnogram
from phase4_sleep_classification.p4_plots       import (
    plot_score_matrix, plot_ecg_hypnogram,
    plot_stage_duration, plot_v2_stage_contributions,
)
from phase4_sleep_classification.p4_save        import save_phase4_results
from Database.db_manager                        import mark_phase_failed, get_record, DB_PATH


def run_phase4_record(config: dict,
                       logger: logging.Logger,
                       db_path: str = DB_PATH) -> dict:
    """
    Run the full Phase 4 pipeline for config['record_name'].

    Returns
    -------
    metrics : dict with n_epochs and per-stage minute counts
    """
    rec_name  = config["record_name"]
    dataset   = config.get("dataset", "slpdb")
    db_row    = get_record(rec_name, dataset, db_path)
    record_id = db_row["record_id"] if db_row else None

    try:
        # ── Step 1: Load Phase 3 outputs ─────────────────────────────────────
        log_step(logger, 1, "Load Phase 3 HRV features + meta")
        load_phase3_meta(config, logger)        # logs DWT provenance
        df = load_features(config, logger)
        log_phase_header(logger, rec_name, dataset, len(df))

        # ── Step 2: Pre-process ───────────────────────────────────────────────
        log_step(logger, 2, "Clip outliers + robust normalisation")
        df     = clip_outliers(df, logger)
        normed = normalise_features(df, logger)

        # ── Step 3: Classify ──────────────────────────────────────────────────
        log_step(logger, 3, "Rule-based scoring + priors + argmax")
        raw_stages, scores = classify_epochs(normed, config, logger)

        # ── Step 4: Smooth ────────────────────────────────────────────────────
        log_step(logger, 4, "Temporal smoothing (median + min-run)")
        smooth_stages = smooth_hypnogram(raw_stages, kernel_size=5, logger=logger)

        # ── Step 5: Plots ─────────────────────────────────────────────────────
        log_step(logger, 5, "Generate diagnostic plots")
        plot_score_matrix(scores,     config, logger)
        plot_ecg_hypnogram(raw_stages, smooth_stages, config, logger)
        plot_stage_duration(raw_stages,    "Raw",      config, logger)
        plot_stage_duration(smooth_stages, "Smoothed", config, logger)
        plot_v2_stage_contributions(df, smooth_stages, config, logger)

        # ── Step 6: Save ──────────────────────────────────────────────────────
        log_step(logger, 6, "Save outputs + update SQLite")
        metrics = save_phase4_results(
            raw_stages, smooth_stages, scores,
            config, logger, db_path=db_path,
        )

        logger.info(
            f"✓ Phase 4 complete — {rec_name} | "
            f"epochs={metrics['n_epochs']} | "
            f"smooth stages={metrics['stage_counts_sm']}"
        )
        return metrics

    except Exception as exc:
        logger.error(f"✗ Phase 4 FAILED — {rec_name}: {exc}")
        if record_id:
            mark_phase_failed(record_id, 4, str(exc), db_path)
        raise


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from phase0_dataset_management.p0_config import CONFIG, make_subject_dirs
    from Database.db_manager import get_all_records

    logger  = setup_logger("phase4", log_dir="logs", log_file="phase4.log")
    records = get_all_records(CONFIG["dataset"])

    for row in records:
        CONFIG["record_name"] = row["record_name"]
        CONFIG["output_dir"]  = make_subject_dirs(
            row["record_name"],
            CONFIG.get("results_dir", "results"),
            CONFIG["dataset"],
        )
        try:
            run_phase4_record(CONFIG, logger)
        except Exception as exc:
            logger.error(f"Skipping {row['record_name']}: {exc}")
            continue

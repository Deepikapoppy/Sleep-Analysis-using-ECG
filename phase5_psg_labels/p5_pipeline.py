"""
=============================================================================
phase5_psg_labels/p5_pipeline.py
Master orchestrator for Phase 5 — PSG Gold Standard Labels.

Execution order
───────────────
1  p5_load_inputs     → phase1_meta, phase3_meta, phase1_sqi,
                         PSG annotations, original Fs
2  p5_build_hypnogram → FIX-3 offset correction + epoch alignment
3  p5_plots           → 5 diagnostic plots
4  p5_save            → npy + CSV + stats JSON + phase5_meta.json
                         + duration summary CSV + SQLite phase5_done=1

Prerequisites
─────────────
Phase 1 must be complete (phase1_meta.json must exist for n_epochs).
Phase 3 output (phase3_hrv_features.csv) is optional — plot 5 is
  skipped gracefully if absent.
Phase 4 output (phase4_ecg_hypnogram.csv) is optional — ECG rows in
  the duration summary are skipped gracefully if absent.

Note: Phase 5 (PSG labels) is independent of Phase 4 (ECG staging)
and can be run in parallel with Phase 4, or before it.  The duration
summary simply shows NaN ECG rows when Phase 4 is not yet done.
=============================================================================
"""

import os
import sys
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase5_psg_labels.p5_logging       import setup_logger, log_phase_header, log_step
from phase5_psg_labels.p5_load_inputs   import (
    load_phase1_meta, load_phase3_meta, load_phase1_sqi,
    load_psg_annotations, get_original_fs,
)
from phase5_psg_labels.p5_build_hypnogram import build_psg_hypnogram
from phase5_psg_labels.p5_plots          import (
    plot_psg_hypnogram, plot_psg_stage_distribution,
    plot_annotation_timeline, plot_psg_sqi_overlay,
    plot_psg_hrv_summary,
)
from phase5_psg_labels.p5_save           import (
    save_psg, save_phase5_meta,
    collect_duration_row, write_duration_summary, update_sqlite,
)
from Database.db_manager                 import mark_phase_failed, get_record, DB_PATH


def run_phase5_record(config: dict,
                       logger: logging.Logger,
                       db_path: str = DB_PATH) -> tuple:
    """
    Run the full Phase 5 pipeline for config['record_name'].

    Returns
    -------
    (psg_hyp, duration_rows)
    """
    rec_name  = config["record_name"]
    dataset   = config.get("dataset", "slpdb")
    db_row    = get_record(rec_name, dataset, db_path)
    record_id = db_row["record_id"] if db_row else None

    try:
        # ── Step 1: Load inputs ───────────────────────────────────────────────
        log_step(logger, 1, "Load upstream metadata + PSG annotations")
        phase1_meta = load_phase1_meta(config, logger)
        phase3_meta = load_phase3_meta(config, logger)
        sqi_df      = load_phase1_sqi(config, logger)
        ann              = load_psg_annotations(config, logger)
        fs_orig, sig_len = get_original_fs(config, logger)

        # n_ecg_epochs from phase1_meta.json (authoritative)
        meta1_path   = os.path.join(config["output_dir"], "phase1_meta.json")
        with open(meta1_path) as f:
            n_ecg_epochs = json.load(f)["n_epochs"]

        log_phase_header(logger, rec_name, dataset, n_ecg_epochs)

        # ── Step 2: Build PSG hypnogram ───────────────────────────────────────
        log_step(logger, 2, "Map labels + epoch alignment (FIX-3 offset)")
        psg_hyp, ann_times, ann_stages = build_psg_hypnogram(
            ann, fs_orig, n_ecg_epochs, config, logger
        )

        # ── Step 3: Plots ─────────────────────────────────────────────────────
        log_step(logger, 3, "Generate diagnostic plots (5)")
        plot_psg_hypnogram(psg_hyp, config, logger)
        plot_psg_stage_distribution(psg_hyp, config, logger)
        plot_annotation_timeline(ann_times, ann_stages, config, logger)
        plot_psg_sqi_overlay(psg_hyp, sqi_df, config, logger)
        plot_psg_hrv_summary(psg_hyp, config, logger)

        # ── Step 4: Save ──────────────────────────────────────────────────────
        log_step(logger, 4, "Save outputs + duration summary + SQLite")
        stats = save_psg(psg_hyp, config, logger)
        save_phase5_meta(psg_hyp, config, logger,
                         phase1_meta=phase1_meta, phase3_meta=phase3_meta)
        rows  = collect_duration_row(psg_hyp, config, logger)
        write_duration_summary(rows, config, logger)
        update_sqlite(psg_hyp, stats, config, logger, db_path=db_path)

        logger.info(f"✓ Phase 5 complete — {rec_name} | epochs={len(psg_hyp)}")
        return psg_hyp, rows

    except Exception as exc:
        logger.error(f"✗ Phase 5 FAILED — {rec_name}: {exc}")
        if record_id:
            mark_phase_failed(record_id, 5, str(exc), db_path)
        raise


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from phase0_dataset_management.p0_config import CONFIG, make_subject_dirs
    from Database.db_manager import get_all_records

    logger   = setup_logger("phase5", log_dir="logs", log_file="phase5.log")
    records  = get_all_records(CONFIG["dataset"])
    all_rows = []

    for row in records:
        CONFIG["record_name"] = row["record_name"]
        CONFIG["output_dir"]  = make_subject_dirs(
            row["record_name"],
            CONFIG.get("results_dir", "results"),
            CONFIG["dataset"],
        )
        try:
            _, rows = run_phase5_record(CONFIG, logger)
            all_rows += rows
        except Exception as exc:
            logger.error(f"Skipping {row['record_name']}: {exc}")
            continue

    logger.info("Phase 5 — all subjects done.")

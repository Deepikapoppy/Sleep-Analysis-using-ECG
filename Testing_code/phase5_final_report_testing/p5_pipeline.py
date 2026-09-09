"""
=============================================================================
phase5_final_report_testing/p5_pipeline.py
Master orchestrator — Phase 5 Final Report for one device session.

This is the deployment-side replacement for training's Phase 5 + Phase 6
combined: it takes the pretrained model's predictions (Phase 4b testing)
and produces the same *style* of reporting as training — hypnogram,
stage distribution, SQI overlay, HRV-by-stage boxplots, final summary
dashboard — built on the PREDICTED stage per epoch. Nothing here computes
accuracy / kappa / confusion matrix, because device sessions have no PSG
ground truth to score against; that comparison is fundamentally impossible
for this dataset, not merely skipped.

Execution order
───────────────
1  Load Phase 1 meta + SQI, Phase 3 features, Phase 4b predictions
2  Generate 5 plots (hypnogram, distribution, SQI overlay, HRV summary,
   final summary dashboard)
3  Save duration summary CSV + phase5_meta.json + update SQLite

Prerequisites
─────────────
Phase 1 testing (phase1_meta.json) must exist.
Phase 4b testing (phase4b_ecg_hypnogram_rf.npy + _raw.npy + predict_meta.json)
  must exist — this is the hard requirement, there is nothing to report
  without a prediction.
Phase 3 testing (phase3_hrv_features.csv) and Phase 1's phase1_sqi.csv are
  optional — their plots are skipped gracefully if absent.
=============================================================================
"""

import logging

from .p5_logging     import setup_logger, log_phase_header, log_step
from .p5_load_inputs import load_all
from .p5_plots        import (
    plot_predicted_hypnogram, plot_stage_distribution,
    plot_sqi_overlay, plot_hrv_summary, plot_final_summary,
)
from .p5_save          import save_duration_summary, save_phase5_meta, update_sqlite

from Database.db_manager import get_record, mark_phase_failed, DB_PATH


def run_phase5_record(config: dict, logger: logging.Logger,
                      db_path: str = DB_PATH) -> dict:
    """
    Run the full Phase 5 final-report pipeline for config['record_name'].

    Returns
    -------
    meta : dict — same content as phase5_meta.json
    """
    rec_name  = config["record_name"]
    dataset   = config.get("dataset", "device")
    db_row    = get_record(rec_name, dataset, db_path)
    record_id = db_row["record_id"] if db_row else None

    try:
        log_step(logger, 1, "Load Phase 1 / Phase 3 / Phase 4b outputs")
        data = load_all(config, logger)
        phase1_meta  = data["phase1_meta"]
        sqi_df       = data["sqi_df"]
        hrv_df       = data["hrv_df"]
        pred_smooth  = data["pred_smooth"]
        predict_meta = data["predict_meta"]

        log_phase_header(logger, rec_name, dataset, len(pred_smooth))

        log_step(logger, 2, "Generate plots (5)")
        rec_start = phase1_meta.get("recording_start_time")
        plot_predicted_hypnogram(pred_smooth, config, logger,
                                 recording_start_time=rec_start)
        plot_stage_distribution(pred_smooth, config, logger)
        plot_sqi_overlay(pred_smooth, sqi_df, config, logger)
        plot_hrv_summary(pred_smooth, hrv_df, config, logger)
        plot_final_summary(pred_smooth, phase1_meta, predict_meta, config, logger)

        log_step(logger, 3, "Save duration summary + meta + SQLite")
        save_duration_summary(pred_smooth, config, logger)
        meta = save_phase5_meta(pred_smooth, phase1_meta, predict_meta, config, logger)
        update_sqlite(pred_smooth, config, logger, db_path=db_path)

        logger.info(
            f"✓ Phase 5 final report complete — {rec_name} | "
            f"epochs={meta['n_epochs']} | "
            f"stages={meta['stage_minutes_predicted']}"
        )
        return meta

    except Exception as exc:
        logger.error(f"✗ Phase 5 final report FAILED — {rec_name}: {exc}")
        if record_id:
            mark_phase_failed(record_id, 5, str(exc), db_path)
        raise

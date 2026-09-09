"""
=============================================================================
phase4b_ml_classification_testing/p4b_pipeline.py
Master orchestrator for Phase 4b DEPLOYMENT prediction on a device session.

This is NOT phase4_sleep_classification_testing (the rule-based scorer) —
that package is unused in the deployment path. This loads your pretrained
RandomForestClassifier bundle and predicts directly from Phase 3 features.

Execution order
───────────────
1  Load Phase 3 features for the device session (phase3_hrv_features.csv)
2  Load the pretrained model bundle (model, imputer, feature_cols)
3  Align this session's columns to feature_cols exactly (missing → NaN,
   filled by the TRAINING imputer; extra columns dropped)
4  Predict raw stage per epoch
5  Smooth (median filter + min-run merger — same utility Phase 4's
   rule-based pipeline uses, reused here as pure post-processing)
6  Save raw + smoothed hypnogram .npy, a meta JSON, and (optional) mark
   phase4b_done in the isolated test_pipeline.db — no accuracy/kappa are
   computed or stored, since there is no ground truth for device sessions

Prerequisites
─────────────
Phase 3 must be complete for the session (phase3_hrv_features.csv must
exist under config["output_dir"]), and a trained phase4b_rf_model.pkl must
exist (produced by phase4b_ml_classification/p4b_ml_classify.py — training,
on your labeled hmc/slpdb data, NOT on device sessions).
=============================================================================
"""

import os
import sys
import json
import logging
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .p4b_logging import setup_logger, log_phase_header, log_step
from .p4b_predict  import (
    load_model_bundle, align_features, predict_session, STAGE_NAMES
)

# Reused as-is from your production packages — both are dataset-agnostic
# utilities with zero rule-based scoring or hmc/slpdb-specific logic:
from phase4_sleep_classification.p4_load_phase3 import load_features, load_phase3_meta
from phase4_sleep_classification.p4_classify     import smooth_hypnogram

from Database.db_manager import get_record, get_connection, mark_phase_failed, DB_PATH


# ─────────────────────────────────────────────────────────────────────────────
#  SQLite: ensure phase4b columns exist (mirrors training script's helper,
#  minus accuracy/kappa — no ground truth for device sessions)
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_phase4b_predict_columns(db_path: str = DB_PATH) -> None:
    with get_connection(db_path) as conn:
        existing = {row[1] for row in
                    conn.execute("PRAGMA table_info(processing_status)")}
        additions = {
            "phase4b_done"  : "INTEGER DEFAULT 0",
            "phase4b_at"    : "TEXT",
            "phase4b_output": "TEXT",
            "phase4b_model_path": "TEXT",
        }
        for col, coltype in additions.items():
            if col not in existing:
                conn.execute(
                    f"ALTER TABLE processing_status ADD COLUMN {col} {coltype}"
                )
        conn.commit()


def _mark_phase4b_predict(record_id: int, output_dir: str, model_path: str,
                           db_path: str = DB_PATH) -> None:
    from datetime import datetime
    with get_connection(db_path) as conn:
        conn.execute(
            """UPDATE processing_status
               SET phase4b_done=1, phase4b_at=?, phase4b_output=?,
                   phase4b_model_path=?, updated_at=?
               WHERE record_id=?""",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), output_dir,
             model_path, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             record_id),
        )
        conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_phase4b_predict_record(config: dict,
                                model_path: str,
                                logger: logging.Logger,
                                db_path: str = DB_PATH) -> dict:
    """
    Run Phase 4b prediction for config['record_name'] (a device session id).

    Parameters
    ----------
    config     : CONFIG dict (needs record_name, output_dir, dataset)
    model_path : path to a trained phase4b_rf_model.pkl — YOU must set this
                 explicitly, it is not derived from config["dataset"]
                 (config["dataset"]="device" has no trained model of its
                 own; the model lives under whichever dataset — hmc or
                 slpdb — it was trained on)
    logger     : logger
    db_path    : isolated test DB path

    Returns
    -------
    result : dict — n_epochs, predicted stage distribution, output paths
    """
    rec_name  = config["record_name"]
    dataset   = config.get("dataset", "device")
    db_row    = get_record(rec_name, dataset, db_path)   # None if not registered — fine
    record_id = db_row["record_id"] if db_row else None

    try:
        # ── Step 1: Load Phase 3 features ─────────────────────────────────────
        log_step(logger, 1, "Load Phase 3 HRV features")
        load_phase3_meta(config, logger)   # logs DWT provenance, harmless if absent
        df = load_features(config, logger)

        # ── Step 2: Load pretrained model bundle ──────────────────────────────
        log_step(logger, 2, "Load pretrained model bundle")
        bundle = load_model_bundle(model_path, logger)
        feat_cols = bundle["feature_cols"]

        log_phase_header(logger, rec_name, dataset, len(df), model_path)

        # ── Step 3: Align features to the model's training columns ───────────
        log_step(logger, 3, "Align features to model's training columns")
        X, align_info = align_features(df, feat_cols, logger)

        # ── Step 4: Predict ───────────────────────────────────────────────────
        log_step(logger, 4, "Predict")
        preds_raw = predict_session(X, bundle, logger)

        # ── Step 5: Smooth ────────────────────────────────────────────────────
        log_step(logger, 5, "Temporal smoothing (median + min-run)")
        preds_smooth = smooth_hypnogram(
            preds_raw, kernel_size=5, min_run_epochs=2, logger=logger
        )

        # ── Step 6: Save ──────────────────────────────────────────────────────
        log_step(logger, 6, "Save outputs + update SQLite (optional)")
        out = config["output_dir"]
        os.makedirs(out, exist_ok=True)

        np.save(os.path.join(out, "phase4b_ecg_hypnogram_rf_raw.npy"), preds_raw)
        np.save(os.path.join(out, "phase4b_ecg_hypnogram_rf.npy"),     preds_smooth)

        raw_counts    = Counter(preds_raw)
        smooth_counts = Counter(preds_smooth)
        meta = {
            "record"          : rec_name,
            "dataset"         : dataset,
            "n_epochs"        : int(len(preds_smooth)),
            "model_path"      : model_path,
            "model_type"      : type(bundle["model"]).__name__,
            "n_feature_cols"  : len(feat_cols),
            "stage_counts_raw": {STAGE_NAMES.get(k, str(k)): int(v)
                                  for k, v in sorted(raw_counts.items())},
            "stage_counts_smooth": {STAGE_NAMES.get(k, str(k)): int(v)
                                     for k, v in sorted(smooth_counts.items())},
            "feature_alignment": align_info,
            "note": "No accuracy/kappa — device sessions have no ground truth.",
        }
        meta_path = os.path.join(out, "phase4b_predict_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Prediction meta saved → {meta_path}")

        if record_id:
            _ensure_phase4b_predict_columns(db_path)
            _mark_phase4b_predict(record_id, out, model_path, db_path)
            logger.info("SQLite: phase4b_done=1 (prediction, no ground truth)")

        logger.info(
            f"✓ Phase 4b prediction complete — {rec_name} | "
            f"epochs={meta['n_epochs']} | "
            f"smoothed stages={meta['stage_counts_smooth']}"
        )
        return meta

    except Exception as exc:
        logger.error(f"✗ Phase 4b prediction FAILED — {rec_name}: {exc}")
        if record_id:
            mark_phase_failed(record_id, 4, str(exc), db_path)   # reuses phase4 slot
        raise


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from phase0_dataset_management_testing import CONFIG, scan_datasets, make_subject_dirs
    from Database.db_manager import get_all_records

    TEST_DB_PATH = "test_pipeline.db"

    # ── EDIT THIS: path to your trained model ──────────────────────────────
    # Produced by phase4b_ml_classification/p4b_ml_classify.py (training).
    # Points at whichever dataset the model was actually trained on —
    # hmc had more subjects (154 vs 18) and better OOF kappa in your
    # earlier diagnostics run, so it's the default guess here — change if
    # you trained on slpdb instead, or on a pooled/combined set.
    MODEL_PATH = os.path.join("results", "hmc", "consolidated", "phase4b_rf_model.pkl")

    logger = setup_logger("phase4b_test", log_dir="logs", log_file="phase4b_test.log")

    db_records = get_all_records(CONFIG["dataset"], db_path=TEST_DB_PATH)
    session_ids = ([row["record_name"] for row in db_records] if db_records
                   else scan_datasets(CONFIG, logger))

    for session_id in session_ids:
        CONFIG["record_name"] = session_id
        CONFIG["output_dir"]  = make_subject_dirs(
            session_id, CONFIG["results_dir"], CONFIG["dataset"]
        )
        try:
            run_phase4b_predict_record(CONFIG, MODEL_PATH, logger, db_path=TEST_DB_PATH)
        except Exception as exc:
            logger.error(f"Skipping {session_id}: {exc}")
            continue

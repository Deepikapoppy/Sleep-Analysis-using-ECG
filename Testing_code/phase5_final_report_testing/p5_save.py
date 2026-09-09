"""
=============================================================================
phase5_final_report_testing/p5_save.py
Step 3 — Persist outputs + update SQLite for one device session.

Files written to config["output_dir"]/
    phase5_duration_summary.csv   stage, minutes (predicted)
    phase5_meta.json              model info, DWT provenance, stage durations

SQLite note: the existing processing_status schema has phase5_* columns
that were designed for PSG gold-standard durations (phase5_n_psg_epochs,
phase5_wake_min, phase5_n1_min, ... phase5_rem_min). Since device sessions
have no PSG, these are repurposed here to store the PREDICTED stage
durations instead — no schema change needed, just a different meaning for
this dataset. This is documented in phase5_meta.json's "note" field so it's
never ambiguous later.
=============================================================================
"""

import os
import json
import logging
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd

from Database.db_manager import get_record, mark_phase_done, get_connection, DB_PATH

from .p5_plots import STAGE_NAMES


def save_duration_summary(pred_smooth: np.ndarray, config: dict,
                          logger: logging.Logger) -> str:
    counts = Counter(pred_smooth)
    rows = [
        {"stage": STAGE_NAMES[s], "minutes": round(counts.get(s, 0) * 30 / 60, 2)}
        for s in sorted(STAGE_NAMES.keys())
    ]
    df  = pd.DataFrame(rows)
    out = os.path.join(config["output_dir"], "phase5_duration_summary.csv")
    df.to_csv(out, index=False)
    logger.info(f"Duration summary saved → {out}")
    return out


def save_phase5_meta(pred_smooth: np.ndarray, phase1_meta: dict,
                     predict_meta: dict, config: dict,
                     logger: logging.Logger) -> dict:
    counts = Counter(pred_smooth)
    stage_min = {STAGE_NAMES[s]: round(counts.get(s, 0) * 30 / 60, 2)
                for s in sorted(STAGE_NAMES.keys())}

    meta = {
        "record"             : config.get("record_name"),
        "dataset"             : config.get("dataset", "device"),
        "n_epochs"            : int(len(pred_smooth)),
        "model_path"          : predict_meta.get("model_path"),
        "model_type"          : predict_meta.get("model_type"),
        "n_feature_cols"      : predict_meta.get("n_feature_cols"),
        "stage_minutes_predicted": stage_min,
        "phase1_dwt_wavelet"  : phase1_meta.get("dwt_wavelet", "db4"),
        "phase1_dwt_level"    : phase1_meta.get("dwt_level", 5),
        "phase1_mean_sqi"     : phase1_meta.get("mean_sqi"),
        "note": (
            "No PSG ground truth for device sessions — this file reports "
            "the RF model's PREDICTED stage durations only. No accuracy / "
            "kappa / confusion matrix exist for this dataset."
        ),
    }
    out = os.path.join(config["output_dir"], "phase5_meta.json")
    with open(out, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Phase 5 meta saved → {out}")
    return meta


def update_sqlite(pred_smooth: np.ndarray, config: dict, logger: logging.Logger,
                  db_path: str = DB_PATH) -> None:
    """
    Mark phase5_done=1 and repurpose the existing PSG-duration columns to
    store PREDICTED durations for device sessions (documented above).
    """
    try:
        db_row = get_record(config["record_name"], config.get("dataset", "device"), db_path)
        if not db_row:
            logger.info("Session not registered in SQLite — skipping DB update.")
            return

        record_id = db_row["record_id"]
        mark_phase_done(record_id, 5, config["output_dir"], db_path)

        counts = Counter(pred_smooth)
        mins   = {s: counts.get(s, 0) * 30 / 60 for s in range(5)}

        with get_connection(db_path) as conn:
            conn.execute(
                """UPDATE processing_status
                   SET phase5_n_psg_epochs = ?,
                       phase5_wake_min     = ?,
                       phase5_n1_min       = ?,
                       phase5_n2_min       = ?,
                       phase5_n3_min       = ?,
                       phase5_rem_min      = ?
                   WHERE record_id = ?""",
                (int(len(pred_smooth)), mins[0], mins[1], mins[2], mins[3], mins[4],
                 record_id),
            )
            conn.commit()
        logger.info(
            "SQLite: phase5_done=1, predicted stage durations written "
            "(repurposing PSG-duration columns — no PSG exists for device)."
        )
    except Exception as exc:
        logger.warning(f"SQLite update failed: {exc}")

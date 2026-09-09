"""
=============================================================================
phase4_sleep_classification/p4_save.py
Step 5 — Save Phase 4 outputs and update SQLite.

Files written to <output_dir>/:
    ecg_hypnogram_raw.npy       — raw stage array
    ecg_hypnogram_smooth.npy    — smoothed stage array
    stage_scores.npy            — (n_epochs, 5) score matrix
    phase4_ecg_hypnogram.csv    — epoch-level hypnogram table
    phase4_stage_stats.json     — stage counts and durations (minutes)

SQLite: processing_status.phase4_done = 1
=============================================================================
"""

import os
import sys
import json
import logging
import sqlite3
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Database.db_manager import get_record, mark_phase_done, DB_PATH

STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


def save_phase4_results(raw_stages,
                         smooth_stages,
                         scores: np.ndarray,
                         config: dict,
                         logger: logging.Logger,
                         db_path: str = DB_PATH) -> dict:
    """
    Persist all Phase 4 outputs to disk and mark phase4_done = 1 in SQLite.

    Returns
    -------
    metrics : dict with stage counts
    """
    out = config["output_dir"]
    os.makedirs(out, exist_ok=True)

    # ── NumPy arrays ──────────────────────────────────────────────────────────
    np.save(os.path.join(out, "ecg_hypnogram_raw.npy"),    raw_stages)
    np.save(os.path.join(out, "ecg_hypnogram_smooth.npy"), smooth_stages)
    np.save(os.path.join(out, "stage_scores.npy"),         scores)

    # ── CSV hypnogram ─────────────────────────────────────────────────────────
    df_hyp = pd.DataFrame({
        "epoch_idx"   : np.arange(len(raw_stages)),
        "time_sec"    : np.arange(len(raw_stages)) * 30,
        "time_hr"     : np.arange(len(raw_stages)) * 30 / 3600,
        "raw_stage"   : raw_stages,
        "smooth_stage": smooth_stages,
        "stage_label" : [STAGE_NAMES[s] for s in smooth_stages],
        **{f"score_{STAGE_NAMES[i]}": scores[:, i] for i in range(5)},
    })
    csv_path = os.path.join(out, "phase4_ecg_hypnogram.csv")
    df_hyp.to_csv(csv_path, index=False)
    logger.info(f"ECG hypnogram saved → {csv_path}")

    # ── Stage stats JSON ──────────────────────────────────────────────────────
    counts = Counter(smooth_stages)
    stats  = {
        STAGE_NAMES[s]: {
            "epochs" : int(v),
            "minutes": round(v * 30 / 60, 1),
        }
        for s, v in counts.items()
    }
    stats_path = os.path.join(out, "phase4_stage_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Stage stats (smooth): {stats}")

    metrics = {
        "n_epochs"        : int(len(smooth_stages)),
        "stage_counts_raw": {STAGE_NAMES[s]: int(v)
                              for s, v in Counter(raw_stages).items()},
        "stage_counts_sm" : {STAGE_NAMES[s]: int(v)
                              for s, v in counts.items()},
    }

    # ── SQLite ────────────────────────────────────────────────────────────────
    try:
        db_row = get_record(config["record_name"],
                            config.get("dataset", "slpdb"), db_path)
        if db_row:
            mark_phase_done(db_row["record_id"], 4, out, db_path)
            conn = sqlite3.connect(db_path)
            conn.execute("""
                UPDATE processing_status
                   SET phase4_n_epochs  = ?,
                       phase4_wake_min  = ?,
                       phase4_n1_min    = ?,
                       phase4_n2_min    = ?,
                       phase4_n3_min    = ?,
                       phase4_rem_min   = ?
                 WHERE record_id = ?
            """, (
                metrics["n_epochs"],
                stats.get("Wake", {}).get("minutes", 0.0),
                stats.get("N1",   {}).get("minutes", 0.0),
                stats.get("N2",   {}).get("minutes", 0.0),
                stats.get("N3",   {}).get("minutes", 0.0),
                stats.get("REM",  {}).get("minutes", 0.0),
                db_row["record_id"],
            ))
            conn.commit()
            conn.close()
            logger.info("SQLite: phase4_done=1, phase4 summary columns written.")
    except Exception as exc:
        logger.warning(f"SQLite update failed: {exc}")

    return metrics

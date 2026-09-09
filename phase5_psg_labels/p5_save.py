"""
=============================================================================
phase5_psg_labels/p5_save.py
Step 4 — Persist Phase 5 outputs and update SQLite.

Functions
─────────
  save_psg               psg_hypnogram.npy + phase5_psg_hypnogram.csv
                         + phase5_psg_stats.json
  save_phase5_meta       phase5_meta.json  (DWT provenance + PSG stats,
                         consumed by Phase 6)
  collect_duration_row   per-subject row: ECG Raw / ECG Smoothed / PSG mins
  write_duration_summary append to cumulative sleep_stage_duration_summary.csv

SQLite: processing_status.phase5_done = 1
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
from phase4_sleep_classification.p4_plots import STAGE_NAMES

STAGE_ORDER = [0, 1, 2, 3, 4]


# ─────────────────────────────────────────────────────────────────────────────
#  Save PSG hypnogram + stats
# ─────────────────────────────────────────────────────────────────────────────

def save_psg(psg_hyp: np.ndarray,
              config: dict,
              logger: logging.Logger) -> dict:
    """
    Write:
      psg_hypnogram.npy          numpy array (n_epochs,)
      phase5_psg_hypnogram.csv   epoch-level table
      phase5_psg_stats.json      stage counts and durations

    Returns the stats dict.
    """
    out = config["output_dir"]
    os.makedirs(out, exist_ok=True)

    # .npy
    np.save(os.path.join(out, "psg_hypnogram.npy"), psg_hyp)

    # CSV
    df_psg = pd.DataFrame({
        "epoch_idx"  : np.arange(len(psg_hyp)),
        "time_sec"   : np.arange(len(psg_hyp)) * 30,
        "time_hr"    : np.arange(len(psg_hyp)) * 30 / 3600,
        "psg_stage"  : psg_hyp,
        "stage_label": [STAGE_NAMES.get(int(s), "?") for s in psg_hyp],
    })
    csv_path = os.path.join(out, "phase5_psg_hypnogram.csv")
    df_psg.to_csv(csv_path, index=False)
    logger.info(f"PSG hypnogram saved → {csv_path}")

    # Stats JSON
    counts = Counter(psg_hyp)
    stats  = {
        STAGE_NAMES.get(s, str(s)): {
            "epochs" : int(v),
            "minutes": round(v * 30 / 60, 1),
        }
        for s, v in counts.items()
    }
    with open(os.path.join(out, "phase5_psg_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"PSG stage stats: {stats}")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
#  Meta JSON
# ─────────────────────────────────────────────────────────────────────────────

def save_phase5_meta(psg_hyp: np.ndarray,
                      config: dict,
                      logger: logging.Logger,
                      phase1_meta: Optional[dict] = None,
                      phase3_meta: Optional[dict] = None) -> dict:
    """
    Write phase5_meta.json with full DWT provenance and PSG statistics
    so Phase 6 (metrics) can load all upstream context in one place.
    """
    counts = Counter(psg_hyp)
    out    = config["output_dir"]

    meta5 = {
        # PSG stats
        "n_psg_epochs"   : int(len(psg_hyp)),
        "psg_stage_counts": {
            STAGE_NAMES.get(s, str(s)): int(v) for s, v in counts.items()
        },
        # Phase 1 DWT provenance
        "phase1_preprocessing"      : (phase1_meta or {}).get("preprocessing",     "DWT"),
        "phase1_dwt_wavelet"        : (phase1_meta or {}).get("dwt_wavelet",        "db4"),
        "phase1_dwt_level"          : (phase1_meta or {}).get("dwt_level",           5),
        "phase1_dwt_approx_zeroed"  : (phase1_meta or {}).get("dwt_approx_zeroed",  True),
        "phase1_dwt_pli_zeroed"     : (phase1_meta or {}).get("dwt_pli_zeroed",      True),
        "phase1_dwt_soft_thresh"    : (phase1_meta or {}).get("dwt_soft_thresh",     True),
        "phase1_polarity_inverted"  : (phase1_meta or {}).get("polarity_inverted",  False),
        "phase1_fs"                 : (phase1_meta or {}).get("fs",                  125),
        "phase1_mean_sqi"           : (phase1_meta or {}).get("mean_sqi",            None),
        "phase1_pct_bad"            : (phase1_meta or {}).get("pct_bad",             None),
        # Phase 3 v2 feature info
        "phase3_n_features"         : (phase3_meta or {}).get("n_feature_cols",      None),
        "phase3_sqi_merged"         : (phase3_meta or {}).get("sqi_merged",          False),
        "phase3_feature_columns"    : (phase3_meta or {}).get("feature_columns",     []),
    }
    meta_path = os.path.join(out, "phase5_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta5, f, indent=2)
    logger.info(f"Phase 5 meta saved → {meta_path}")
    return meta5


# ─────────────────────────────────────────────────────────────────────────────
#  Duration summary helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_overall_dir(config: dict) -> str:
    overall = os.path.join(
        config.get("results_dir", "results"),
        config.get("dataset", "slpdb"),
        "plots", "overall",
    )
    os.makedirs(overall, exist_ok=True)
    return overall


def collect_duration_row(psg_hyp: np.ndarray,
                          config: dict,
                          logger: logging.Logger) -> list:
    """
    Build per-subject duration rows for:
      ECG Raw      — from phase4_ecg_hypnogram.csv  raw_stage column
      ECG Smoothed — from phase4_ecg_hypnogram.csv  smooth_stage column
      PSG          — from psg_hyp argument
    """
    subject   = config["record_name"]
    out_dir   = config["output_dir"]
    epoch_min = 30 / 60

    def to_mins(stages):
        c = Counter(stages)
        return {STAGE_NAMES[s]: round(c.get(s, 0) * epoch_min, 1)
                for s in STAGE_ORDER}

    rows = []

    # Phase 4 ECG hypnogram (renamed from phase5 in old code)
    p4_csv = os.path.join(out_dir, "phase4_ecg_hypnogram.csv")
    if os.path.exists(p4_csv):
        df4 = pd.read_csv(p4_csv)
        for col, label in [("raw_stage", "ECG Raw"), ("smooth_stage", "ECG Smoothed")]:
            if col in df4.columns:
                mins      = to_mins(df4[col].values.astype(int))
                total_min = round(sum(mins.values()), 1)
                rows.append({
                    "Subject": subject, "Type": label, **mins,
                    "Total (min)": total_min,
                    "Total (hrs)": round(total_min / 60, 2),
                })
            else:
                logger.warning(f"Column '{col}' missing in {p4_csv}")
    else:
        logger.warning(f"phase4_ecg_hypnogram.csv not found for {subject}")

    # PSG row
    mins      = to_mins(psg_hyp)
    total_min = round(sum(mins.values()), 1)
    rows.append({
        "Subject": subject, "Type": "PSG", **mins,
        "Total (min)": total_min,
        "Total (hrs)": round(total_min / 60, 2),
    })
    return rows


def write_duration_summary(all_rows: list,
                            config: dict,
                            logger: logging.Logger) -> None:
    """
    Append / update the cumulative sleep_stage_duration_summary.csv
    in results/<dataset>/plots/overall/.
    Existing rows for updated subjects are replaced; all others preserved.
    """
    overall_dir = _get_overall_dir(config)
    csv_path    = os.path.join(overall_dir, "sleep_stage_duration_summary.csv")
    COLS        = ["Subject", "Type", "Wake", "N1", "N2", "N3", "REM",
                   "Total (min)", "Total (hrs)"]
    type_order  = {"ECG Raw": 0, "ECG Smoothed": 1, "PSG": 2}

    df_new = pd.DataFrame(all_rows, columns=COLS)

    if os.path.exists(csv_path):
        df_old              = pd.read_csv(csv_path)
        subjects_updated    = df_new["Subject"].unique().tolist()
        df_old              = df_old[~df_old["Subject"].isin(subjects_updated)]
        df_out              = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_out = df_new

    df_out["_ord"] = df_out["Type"].map(type_order).fillna(9)
    df_out = df_out.sort_values(["Subject", "_ord"]).drop(columns="_ord")
    df_out.to_csv(csv_path, index=False)
    logger.info(f"Duration summary saved → {csv_path}")
    logger.info(f"\n{df_out.to_string(index=False)}")


# ─────────────────────────────────────────────────────────────────────────────
#  SQLite update
# ─────────────────────────────────────────────────────────────────────────────

def update_sqlite(psg_hyp: np.ndarray,
                   stats: dict,
                   config: dict,
                   logger: logging.Logger,
                   db_path: str = DB_PATH) -> None:
    """Mark phase5_done = 1 and write PSG stage duration summary columns."""
    try:
        db_row = get_record(config["record_name"],
                            config.get("dataset", "slpdb"), db_path)
        if db_row:
            mark_phase_done(db_row["record_id"], 5, config["output_dir"], db_path)
            conn = sqlite3.connect(db_path)
            conn.execute("""
                UPDATE processing_status
                   SET phase5_n_psg_epochs = ?,
                       phase5_wake_min     = ?,
                       phase5_n1_min       = ?,
                       phase5_n2_min       = ?,
                       phase5_n3_min       = ?,
                       phase5_rem_min      = ?
                 WHERE record_id = ?
            """, (
                int(len(psg_hyp)),
                stats.get("Wake", {}).get("minutes", 0.0),
                stats.get("N1",   {}).get("minutes", 0.0),
                stats.get("N2",   {}).get("minutes", 0.0),
                stats.get("N3",   {}).get("minutes", 0.0),
                stats.get("REM",  {}).get("minutes", 0.0),
                db_row["record_id"],
            ))
            conn.commit()
            conn.close()
            logger.info("SQLite: phase5_done=1, phase5 PSG summary columns written.")
    except Exception as exc:
        logger.warning(f"SQLite update failed: {exc}")

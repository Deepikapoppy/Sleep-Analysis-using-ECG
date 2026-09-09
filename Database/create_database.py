"""
=============================================================================
create_database.py
Creates the SleepAnalysis SQLite database with full schema:
  - datasets          : one row per dataset (slpdb / hmc)
  - records           : one row per subject/recording file
  - record_metadata   : per-record signal metadata (fs, duration, channels)
  - processing_status : tracks pipeline phase completion per record

UPDATED: Phase 2 result columns added to processing_status:
  - phase2_method_used    : which R-peak detector won most epochs
  - phase2_mean_hr        : overall mean heart rate (bpm)
  - phase2_valid_epochs   : number of epochs with valid RR intervals

UPDATED: Phase 4 result columns added to processing_status:
  - phase4_n_epochs  : total epochs classified
  - phase4_wake_min  : Wake duration (minutes, smoothed hypnogram)
  - phase4_n1_min    : N1 duration (minutes)
  - phase4_n2_min    : N2 duration (minutes)
  - phase4_n3_min    : N3 duration (minutes)
  - phase4_rem_min   : REM duration (minutes)

UPDATED: Phase 5 result columns added to processing_status:
  - phase5_n_psg_epochs : total PSG epochs aligned
  - phase5_wake_min     : PSG Wake duration (minutes)
  - phase5_n1_min       : PSG N1 duration (minutes)
  - phase5_n2_min       : PSG N2 duration (minutes)
  - phase5_n3_min       : PSG N3 duration (minutes)
  - phase5_rem_min      : PSG REM duration (minutes)

UPDATED: Phase 6 result columns added to processing_status:
  (Phase 6 = Hypnogram Comparison & Evaluation, ECG vs PSG; renamed from
   the old "Phase 7")
  - phase6_overall_acc          : overall accuracy (ECG vs PSG)
  - phase6_cohen_kappa          : Cohen's kappa (ECG vs PSG)
  - phase6_kappa_interpretation : text label for kappa (e.g. "Substantial")
  - phase6_n_epochs             : valid epochs scored
  - phase6_per_stage_f1         : JSON string {stage_name: f1, ...}
=============================================================================
"""

import sqlite3
import os
import logging
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "Sleepanalysis.db")


def get_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [%(levelname)s]  %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger("create_database")


def create_database(db_path: str = DB_PATH) -> sqlite3.Connection:
    """
    Create all tables in the SQLite database.
    Safe to call multiple times — uses IF NOT EXISTS.
    Returns an open connection.
    """
    logger = get_logger()
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    cursor = conn.cursor()

    # ── 1. datasets ───────────────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            dataset_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            name            TEXT    NOT NULL UNIQUE,
            format          TEXT    NOT NULL,
            local_path      TEXT    NOT NULL,
            total_records   INTEGER DEFAULT 0,
            registered_at   TEXT    NOT NULL
        )
    """)

    # ── 2. records ────────────────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS records (
            record_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id      INTEGER NOT NULL REFERENCES datasets(dataset_id),
            record_name     TEXT    NOT NULL,
            signal_file     TEXT    NOT NULL,
            annotation_file TEXT,
            file_size_kb    REAL,
            registered_at   TEXT    NOT NULL,
            UNIQUE(dataset_id, record_name)
        )
    """)

    # ── 3. record_metadata ────────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS record_metadata (
            meta_id             INTEGER PRIMARY KEY AUTOINCREMENT,
            record_id           INTEGER NOT NULL UNIQUE
                                    REFERENCES records(record_id),
            original_fs         REAL,
            target_fs           REAL,
            n_samples           INTEGER,
            duration_hr         REAL,
            channels            TEXT,
            ecg_channel_idx     INTEGER,
            ann_labels          TEXT,
            n_annotations       INTEGER,
            recording_start     TEXT,
            extracted_at        TEXT NOT NULL
        )
    """)

    # ── 4. processing_status ─────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS processing_status (
            status_id             INTEGER PRIMARY KEY AUTOINCREMENT,
            record_id             INTEGER NOT NULL UNIQUE
                                      REFERENCES records(record_id),

            -- Phase completion flags: 0=pending, 1=done, 2=failed
            phase0_done           INTEGER DEFAULT 0,
            phase1_done           INTEGER DEFAULT 0,
            phase2_done           INTEGER DEFAULT 0,
            phase3_done           INTEGER DEFAULT 0,
            phase4_done           INTEGER DEFAULT 0,
            phase5_done           INTEGER DEFAULT 0,
            phase6_done           INTEGER DEFAULT 0,

            -- Timestamps
            phase0_at             TEXT,
            phase1_at             TEXT,
            phase2_at             TEXT,
            phase3_at             TEXT,
            phase4_at             TEXT,
            phase5_at             TEXT,
            phase6_at             TEXT,

            -- Output directory paths
            phase0_output         TEXT,
            phase1_output         TEXT,
            phase2_output         TEXT,
            phase3_output         TEXT,
            phase4_output         TEXT,
            phase5_output         TEXT,
            phase6_output         TEXT,

            -- Phase 2 result summary
            phase2_method_used    TEXT,    -- top R-peak detection method
            phase2_mean_hr        REAL,    -- mean heart rate (bpm)
            phase2_valid_epochs   INTEGER, -- epochs with valid RR intervals

            -- Phase 3 result summary (NEW)
            phase3_n_epochs       INTEGER, -- total epochs processed
            phase3_n_features     INTEGER, -- total feature columns extracted
            phase3_t1_cols        INTEGER, -- T1 feature count
            phase3_t2_cols        INTEGER, -- T2 feature count
            phase3_t3_cols        INTEGER, -- T3 feature count
            phase3_sqi_flagged    INTEGER, -- epochs flagged by SQI gate

            -- Phase 4 result summary (sleep stage classification)
            phase4_n_epochs       INTEGER, -- total epochs classified
            phase4_wake_min       REAL,    -- Wake duration (minutes, smoothed)
            phase4_n1_min         REAL,    -- N1 duration (minutes, smoothed)
            phase4_n2_min         REAL,    -- N2 duration (minutes, smoothed)
            phase4_n3_min         REAL,    -- N3 duration (minutes, smoothed)
            phase4_rem_min        REAL,    -- REM duration (minutes, smoothed)

            -- Phase 5 result summary (PSG gold-standard labels)
            phase5_n_psg_epochs   INTEGER, -- total PSG epochs aligned
            phase5_wake_min       REAL,    -- PSG Wake duration (minutes)
            phase5_n1_min         REAL,    -- PSG N1 duration (minutes)
            phase5_n2_min         REAL,    -- PSG N2 duration (minutes)
            phase5_n3_min         REAL,    -- PSG N3 duration (minutes)
            phase5_rem_min        REAL,    -- PSG REM duration (minutes)

            -- Phase 6 result summary (Hypnogram Comparison & Evaluation)
            phase6_overall_acc          REAL,    -- overall accuracy (ECG vs PSG)
            phase6_cohen_kappa          REAL,    -- Cohen's kappa (ECG vs PSG)
            phase6_kappa_interpretation TEXT,    -- kappa text label
            phase6_n_epochs             INTEGER, -- valid epochs scored
            phase6_per_stage_f1         TEXT,    -- JSON {stage: f1, ...}

            -- Error tracking
            error_msg             TEXT,
            updated_at            TEXT NOT NULL
        )
    """)

    conn.commit()
    logger.info(f"Database created / verified  →  {db_path}")
    logger.info("Tables: datasets | records | record_metadata | processing_status")
    logger.info("Phases tracked: 0 (scan) | 1 (preprocess) | 2 (R-peak) | "
                "3 (HRV features) | 4 (sleep staging) | 5 (PSG labels) | "
                "6 (evaluation: ECG vs PSG)")

    # ── ADD COLUMNS to existing DB if upgrading ───────────────────────────────
    _add_column_if_missing(conn, "processing_status",
                            "phase2_method_used",  "TEXT")
    _add_column_if_missing(conn, "processing_status",
                            "phase2_mean_hr",      "REAL")
    _add_column_if_missing(conn, "processing_status",
                            "phase2_valid_epochs", "INTEGER")
    # Phase 3 result columns
    _add_column_if_missing(conn, "processing_status",
                            "phase3_n_epochs",     "INTEGER")
    _add_column_if_missing(conn, "processing_status",
                            "phase3_n_features",   "INTEGER")
    _add_column_if_missing(conn, "processing_status",
                            "phase3_t1_cols",      "INTEGER")
    _add_column_if_missing(conn, "processing_status",
                            "phase3_t2_cols",      "INTEGER")
    _add_column_if_missing(conn, "processing_status",
                            "phase3_t3_cols",      "INTEGER")
    _add_column_if_missing(conn, "processing_status",
                            "phase3_sqi_flagged",  "INTEGER")
    # Phase 4 flag, timestamp, output path
    _add_column_if_missing(conn, "processing_status",
                            "phase4_done",         "INTEGER")
    _add_column_if_missing(conn, "processing_status",
                            "phase4_at",           "TEXT")
    _add_column_if_missing(conn, "processing_status",
                            "phase4_output",       "TEXT")
    # Phase 4 result columns
    _add_column_if_missing(conn, "processing_status",
                            "phase4_n_epochs",     "INTEGER")
    _add_column_if_missing(conn, "processing_status",
                            "phase4_wake_min",     "REAL")
    _add_column_if_missing(conn, "processing_status",
                            "phase4_n1_min",       "REAL")
    _add_column_if_missing(conn, "processing_status",
                            "phase4_n2_min",       "REAL")
    _add_column_if_missing(conn, "processing_status",
                            "phase4_n3_min",       "REAL")
    _add_column_if_missing(conn, "processing_status",
                            "phase4_rem_min",      "REAL")
    # Phase 5 flag, timestamp, output path
    _add_column_if_missing(conn, "processing_status",
                            "phase5_done",         "INTEGER")
    _add_column_if_missing(conn, "processing_status",
                            "phase5_at",           "TEXT")
    _add_column_if_missing(conn, "processing_status",
                            "phase5_output",       "TEXT")
    # Phase 5 result columns (PSG gold-standard)
    _add_column_if_missing(conn, "processing_status",
                            "phase5_n_psg_epochs", "INTEGER")
    _add_column_if_missing(conn, "processing_status",
                            "phase5_wake_min",     "REAL")
    _add_column_if_missing(conn, "processing_status",
                            "phase5_n1_min",       "REAL")
    _add_column_if_missing(conn, "processing_status",
                            "phase5_n2_min",       "REAL")
    _add_column_if_missing(conn, "processing_status",
                            "phase5_n3_min",       "REAL")
    _add_column_if_missing(conn, "processing_status",
                            "phase5_rem_min",      "REAL")

    # Phase 6 flag, timestamp, output path
    _add_column_if_missing(conn, "processing_status",
                            "phase6_done",         "INTEGER")
    _add_column_if_missing(conn, "processing_status",
                            "phase6_at",           "TEXT")
    _add_column_if_missing(conn, "processing_status",
                            "phase6_output",       "TEXT")
    # Phase 6 result columns (Hypnogram Comparison & Evaluation)
    _add_column_if_missing(conn, "processing_status",
                            "phase6_overall_acc",          "REAL")
    _add_column_if_missing(conn, "processing_status",
                            "phase6_cohen_kappa",          "REAL")
    _add_column_if_missing(conn, "processing_status",
                            "phase6_kappa_interpretation", "TEXT")
    _add_column_if_missing(conn, "processing_status",
                            "phase6_n_epochs",             "INTEGER")
    _add_column_if_missing(conn, "processing_status",
                            "phase6_per_stage_f1",         "TEXT")
    return conn


def _add_column_if_missing(conn: sqlite3.Connection,
                             table: str, column: str, col_type: str) -> None:
    """
    Add a column to an existing table only if it doesn't already exist.
    Safe to call on both new and already-created databases.
    """
    cursor = conn.execute(f"PRAGMA table_info({table})")
    existing = [row[1] for row in cursor.fetchall()]
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        conn.commit()
        logging.getLogger("create_database").info(
            f"  Added column '{column}' to {table}"
        )


if __name__ == "__main__":
    conn = create_database()
    conn.close()
    print("Database created successfully.")
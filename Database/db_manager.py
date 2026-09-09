"""
=============================================================================
db_manager.py
Central helper for all SQLite read / write operations.
Used by every phase module — import this instead of touching sqlite3 directly.
=============================================================================
"""

import sqlite3
import json
import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

# Resolve DB path relative to this file (database/ folder)
_HERE   = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_HERE, "Sleepanalysis.db")


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Return a WAL-mode connection with row_factory for dict-like rows."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


# ─────────────────────────────────────────────────────────────────────────────
#  DATASETS
# ─────────────────────────────────────────────────────────────────────────────

def register_dataset(name: str, fmt: str, local_path: str,
                     db_path: str = DB_PATH) -> int:
    """
    Insert or retrieve a dataset row.
    Returns dataset_id.
    """
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "SELECT dataset_id FROM datasets WHERE name = ?", (name,)
        )
        row = cur.fetchone()
        if row:
            return row["dataset_id"]

        cur = conn.execute(
            """INSERT INTO datasets (name, format, local_path, registered_at)
               VALUES (?, ?, ?, ?)""",
            (name, fmt, local_path, _now()),
        )
        conn.commit()
        return cur.lastrowid


def update_dataset_record_count(dataset_id: int, count: int,
                                db_path: str = DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE datasets SET total_records = ? WHERE dataset_id = ?",
            (count, dataset_id),
        )
        conn.commit()


def get_dataset_by_name(name: str, db_path: str = DB_PATH) -> Optional[Dict]:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "SELECT * FROM datasets WHERE name = ?", (name,)
        )
        row = cur.fetchone()
        return dict(row) if row else None


# ─────────────────────────────────────────────────────────────────────────────
#  RECORDS
# ─────────────────────────────────────────────────────────────────────────────

def register_record(dataset_id: int, record_name: str,
                    signal_file: str, annotation_file: Optional[str] = None,
                    db_path: str = DB_PATH) -> int:
    """
    Insert a record row (skip if already exists).
    Returns record_id.
    """
    file_size_kb = None
    if os.path.exists(signal_file):
        file_size_kb = round(os.path.getsize(signal_file) / 1024, 2)

    with get_connection(db_path) as conn:
        cur = conn.execute(
            """SELECT record_id FROM records
               WHERE dataset_id = ? AND record_name = ?""",
            (dataset_id, record_name),
        )
        row = cur.fetchone()
        if row:
            return row["record_id"]

        cur = conn.execute(
            """INSERT INTO records
               (dataset_id, record_name, signal_file, annotation_file,
                file_size_kb, registered_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (dataset_id, record_name, signal_file,
             annotation_file, file_size_kb, _now()),
        )
        conn.commit()

        # Auto-create processing_status row
        conn.execute(
            """INSERT OR IGNORE INTO processing_status
               (record_id, updated_at) VALUES (?, ?)""",
            (cur.lastrowid, _now()),
        )
        conn.commit()
        return cur.lastrowid


def get_record(record_name: str, dataset_name: str,
               db_path: str = DB_PATH) -> Optional[Dict]:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """SELECT r.*, d.name AS dataset_name
               FROM records r
               JOIN datasets d ON d.dataset_id = r.dataset_id
               WHERE r.record_name = ? AND d.name = ?""",
            (record_name, dataset_name),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def get_all_records(dataset_name: str,
                    db_path: str = DB_PATH) -> List[Dict]:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """SELECT r.*
               FROM records r
               JOIN datasets d ON d.dataset_id = r.dataset_id
               WHERE d.name = ?
               ORDER BY r.record_name""",
            (dataset_name,),
        )
        return [dict(row) for row in cur.fetchall()]


# ─────────────────────────────────────────────────────────────────────────────
#  RECORD METADATA
# ─────────────────────────────────────────────────────────────────────────────

def upsert_metadata(record_id: int, meta: Dict[str, Any],
                    db_path: str = DB_PATH) -> None:
    """
    Insert or update record_metadata for a given record_id.
    `meta` keys match column names; lists are JSON-serialised.
    """
    channels     = json.dumps(meta.get("channels", []))
    ann_labels   = json.dumps(meta.get("ann_labels", []))

    with get_connection(db_path) as conn:
        conn.execute(
            """INSERT INTO record_metadata
               (record_id, original_fs, target_fs, n_samples, duration_hr,
                channels, ecg_channel_idx, ann_labels, n_annotations,
                recording_start, extracted_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(record_id) DO UPDATE SET
                 original_fs     = excluded.original_fs,
                 target_fs       = excluded.target_fs,
                 n_samples       = excluded.n_samples,
                 duration_hr     = excluded.duration_hr,
                 channels        = excluded.channels,
                 ecg_channel_idx = excluded.ecg_channel_idx,
                 ann_labels      = excluded.ann_labels,
                 n_annotations   = excluded.n_annotations,
                 recording_start = excluded.recording_start,
                 extracted_at    = excluded.extracted_at
            """,
            (
                record_id,
                meta.get("original_fs"),
                meta.get("target_fs"),
                meta.get("n_samples"),
                meta.get("duration_hr"),
                channels,
                meta.get("ecg_channel_idx"),
                ann_labels,
                meta.get("n_annotations"),
                meta.get("recording_start"),
                _now(),
            ),
        )
        conn.commit()


def get_metadata(record_id: int, db_path: str = DB_PATH) -> Optional[Dict]:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "SELECT * FROM record_metadata WHERE record_id = ?", (record_id,)
        )
        row = cur.fetchone()
        if not row:
            return None
        d = dict(row)
        d["channels"]   = json.loads(d["channels"])   if d["channels"]   else []
        d["ann_labels"] = json.loads(d["ann_labels"]) if d["ann_labels"] else []
        return d


# ─────────────────────────────────────────────────────────────────────────────
#  PROCESSING STATUS
# ─────────────────────────────────────────────────────────────────────────────

def mark_phase_done(record_id: int, phase: int, output_dir: str = "",
                    db_path: str = DB_PATH) -> None:
    """Mark phaseN as done (1) and record its output path + timestamp."""
    col_done = f"phase{phase}_done"
    col_at   = f"phase{phase}_at"
    col_out  = f"phase{phase}_output"
    with get_connection(db_path) as conn:
        conn.execute(
            f"""UPDATE processing_status
                SET {col_done}=1, {col_at}=?, {col_out}=?, updated_at=?
                WHERE record_id=?""",
            (_now(), output_dir, _now(), record_id),
        )
        conn.commit()


def mark_phase_failed(record_id: int, phase: int, error: str,
                      db_path: str = DB_PATH) -> None:
    col_done = f"phase{phase}_done"
    with get_connection(db_path) as conn:
        conn.execute(
            f"""UPDATE processing_status
                SET {col_done}=2, error_msg=?, updated_at=?
                WHERE record_id=?""",
            (error, _now(), record_id),
        )
        conn.commit()


def get_pending_records(phase: int, dataset_name: str,
                        db_path: str = DB_PATH) -> List[Dict]:
    """Return all records where phaseN is NOT done (status 0 or 2)."""
    col = f"phase{phase}_done"
    with get_connection(db_path) as conn:
        cur = conn.execute(
            f"""SELECT r.record_id, r.record_name, r.signal_file,
                       r.annotation_file, ps.{col} AS phase_status
                FROM records r
                JOIN datasets d ON d.dataset_id = r.dataset_id
                JOIN processing_status ps ON ps.record_id = r.record_id
                WHERE d.name = ? AND ps.{col} != 1
                ORDER BY r.record_name""",
            (dataset_name,),
        )
        return [dict(row) for row in cur.fetchall()]


def get_processing_status(record_id: int,
                           db_path: str = DB_PATH) -> Optional[Dict]:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "SELECT * FROM processing_status WHERE record_id = ?", (record_id,)
        )
        row = cur.fetchone()
        return dict(row) if row else None


def summary_report(db_path: str = DB_PATH) -> None:
    """Print a quick pipeline completion summary to stdout."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """SELECT d.name AS dataset,
                      COUNT(*) AS total,
                      SUM(ps.phase0_done=1) AS p0,
                      SUM(ps.phase1_done=1) AS p1,
                      SUM(ps.phase2_done=1) AS p2,
                      SUM(ps.phase3_done=1) AS p3
               FROM processing_status ps
               JOIN records r ON r.record_id = ps.record_id
               JOIN datasets d ON d.dataset_id = r.dataset_id
               GROUP BY d.name"""
        ).fetchall()

    print("\n" + "="*60)
    print("  PIPELINE STATUS SUMMARY")
    print("="*60)
    print(f"  {'Dataset':<10} {'Total':>6} {'Ph0':>5} {'Ph1':>5} {'Ph2':>5} {'Ph3':>5}")
    print("-"*60)
    for row in rows:
        print(
            f"  {row['dataset']:<10} {row['total']:>6} "
            f"{row['p0']:>5} {row['p1']:>5} "
            f"{row['p2']:>5} {row['p3']:>5}"
        )
    print("="*60 + "\n")


if __name__ == "__main__":
    summary_report()

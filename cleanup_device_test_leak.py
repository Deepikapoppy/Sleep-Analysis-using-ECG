"""
=============================================================================
cleanup_device_test_leak.py
One-off cleanup: removes the "device" test dataset that leaked into the
PRODUCTION Sleepanalysis.db (from the old run_phase0_test.py bug, now fixed
— every p0_*.py module explicitly passes CONFIG["db_path"] so this can't
happen again).

Run this from database_sp/ (where Database/ lives), i.e. place this file
directly in database_sp/, next to the Database/ folder itself:
    cd database_sp
    python cleanup_device_test_leak.py

SAFE BY DESIGN:
  1. Makes a timestamped backup copy of Sleepanalysis.db before touching
     anything.
  2. Prints exactly what it's about to delete and asks for confirmation —
     nothing is deleted until you type "yes".
  3. Deletes child rows before parent rows (processing_status ->
     record_metadata -> records -> datasets), respecting your foreign keys.
=============================================================================
"""

import os
import shutil
import sqlite3
from datetime import datetime

from Database.create_database import DB_PATH   # Sleepanalysis.db, production

DATASET_NAME_TO_REMOVE = "device"


def main():
    if not os.path.exists(DB_PATH):
        print(f"No database found at {DB_PATH} — nothing to do.")
        return

    backup_path = DB_PATH.replace(
        ".db", f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
    )
    shutil.copy2(DB_PATH, backup_path)
    print(f"Backup written -> {backup_path}\n")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")

    row = conn.execute(
        "SELECT * FROM datasets WHERE name = ?", (DATASET_NAME_TO_REMOVE,)
    ).fetchone()
    if row is None:
        print(f"No dataset named '{DATASET_NAME_TO_REMOVE}' found in {DB_PATH}. "
              f"Nothing to do.")
        conn.close()
        return

    dataset_id = row["dataset_id"]
    print(f"Found dataset_id={dataset_id}  name='{row['name']}'  "
          f"local_path='{row['local_path']}'")

    records = conn.execute(
        "SELECT record_id, record_name FROM records WHERE dataset_id = ?",
        (dataset_id,),
    ).fetchall()

    print(f"\nThis will permanently delete {len(records)} record(s) and their "
          f"metadata/status rows, plus the dataset row itself:")
    for r in records:
        print(f"    record_id={r['record_id']:<4} {r['record_name']}")

    confirm = input("\nType 'yes' to proceed, anything else to abort: ").strip()
    if confirm.lower() != "yes":
        print("Aborted — nothing deleted. Backup file above is safe to remove "
              "manually if you didn't need it.")
        conn.close()
        return

    record_ids = [r["record_id"] for r in records]
    placeholders = ",".join("?" * len(record_ids)) if record_ids else None

    if record_ids:
        conn.execute(
            f"DELETE FROM processing_status WHERE record_id IN ({placeholders})",
            record_ids,
        )
        conn.execute(
            f"DELETE FROM record_metadata WHERE record_id IN ({placeholders})",
            record_ids,
        )
        conn.execute(
            f"DELETE FROM records WHERE record_id IN ({placeholders})",
            record_ids,
        )
    conn.execute("DELETE FROM datasets WHERE dataset_id = ?", (dataset_id,))
    conn.commit()

    conn.execute("VACUUM;")
    conn.close()

    print(f"\nDone. Removed dataset '{DATASET_NAME_TO_REMOVE}' "
          f"({len(records)} record(s)) from {DB_PATH}.")
    print(f"Backup kept at {backup_path} — delete it once you've confirmed "
          f"everything looks right.")


if __name__ == "__main__":
    main()

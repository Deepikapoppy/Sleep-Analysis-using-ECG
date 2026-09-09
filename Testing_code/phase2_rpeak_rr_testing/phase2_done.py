# phase2_rpeak_rr_testing/phase2_done.py
# Run this as a one-off script to reset phase2_done=0 for all sessions in
# the ISOLATED TEST DB (never touches your production DB.db).
#
# Usage: python phase2_done.py   (run from project root, or adjust TEST_DB_PATH)

import sqlite3

TEST_DB_PATH = "test_pipeline.db"

conn = sqlite3.connect(TEST_DB_PATH)
conn.execute("UPDATE processing_status SET phase2_done = 0")
conn.commit()
conn.close()
print(f"phase2_done reset to 0 for all sessions in {TEST_DB_PATH}")

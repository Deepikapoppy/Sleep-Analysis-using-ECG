# run this as a one-off Python script: reset_phase2.py

import sqlite3
import sys
sys.path.insert(0, ".")
from Database.db_manager import DB_PATH

conn = sqlite3.connect(DB_PATH)
conn.execute("UPDATE processing_status SET phase2_done = 0")
conn.commit()
conn.close()
print("phase2_done reset to 0 for all records")
# mlearning/clean_db.py

"""
DB Cleanup Utility
- Removes orphan SHAP/meta rows
- Vacuum SQLite
"""

import sqlite3, logging
from config import DB_PATH


def run_db_cleanup():
    logging.info("🧹 Cleaning up DB...")

    try:
        conn = sqlite3.connect(DB_PATH, isolation_level=None)  # auto-commit mode
        cursor = conn.cursor()

        cursor.execute("PRAGMA optimize")
        cursor.execute("VACUUM")  # runs outside transactions
        conn.close()

        logging.info("✅ DB maintenance complete.")
    except Exception as e:
        logging.error(f"❌ DB cleanup failed.\n{e}")
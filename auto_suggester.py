# mlearning/auto_suggester.py

"""
auto_suggester.py

Purpose:
- Fetch top N combos from `meta` table
- Suggests them via logging or return
- Used to report best strategies
"""

import sqlite3
import pandas as pd
import logging
from config import DB_PATH

def suggest_top_combos(top_n=10):
    logging.info("🔍 Recommending best combos from DB...")

    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT combo, target, accuracy, shap_quality
        FROM meta
        WHERE kept = 1
        ORDER BY accuracy DESC, shap_quality DESC
        LIMIT ?
    """
    df = pd.read_sql(query, conn, params=(top_n,))
    conn.close()

    if df.empty:
        logging.warning("⚠️ No top combos found above threshold.")
        return

    for target in df["target"].unique():
        subset = df[df["target"] == target]
        logging.info(f"\n🔹 Top {len(subset)} strategies for {target}:\n{subset.to_string(index=False)}")

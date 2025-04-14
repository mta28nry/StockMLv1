# mlearning/self_improver.py

"""
self_improver.py

Purpose:
- Analyzes `meta` table performance
- Promotes strong combos to future training
- Flags weak combos as pruned
- Updates `combo_history` table with status
"""

import logging
import sqlite3

from config import DB_PATH


def run_self_improver(min_acc=0.52, min_shap=0.15, promote_threshold=3):
    logging.info("🧠 Self-Improver started...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Fetch all combo performance
    cursor.execute("""
        SELECT combo, target, COUNT(*), AVG(accuracy), AVG(shap_quality)
        FROM meta
        WHERE kept = 1
        GROUP BY combo, target
    """)
    rows = cursor.fetchall()

    updates = 0
    for combo, target, count, avg_acc, avg_shap in rows:
        # Promote if good and frequent
        if avg_acc >= min_acc and avg_shap >= min_shap and count >= promote_threshold:
            cursor.execute("""
                UPDATE combo_history
                SET promoted = 1
                WHERE combo = ? AND target = ?
            """, (combo, target))
            logging.info(f"🚀 Promoted combo: {combo} | {target}")
            updates += 1

        # Prune if poor performance
        if avg_acc < min_acc or avg_shap < min_shap:
            cursor.execute("""
                UPDATE combo_history
                SET pruned = 1
                WHERE combo = ? AND target = ?
            """, (combo, target))
            logging.info(f"🪓 Pruned combo: {combo} | {target}")
            updates += 1

    conn.commit()
    conn.close()
    if updates == 0:
        logging.info("✅ No updates necessary. All combos are consistent.")
    else:
        logging.info(f"✅ Self-improvement finished. {updates} combos updated.")

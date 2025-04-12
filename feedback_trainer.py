# mlearning/feedback_trainer.py

"""
Module: feedback_trainer.py
Purpose:
- Evaluate trained combos against actual returns
- Log success rate and average return into SQLite DB
- Supports full-loop scoring feedback
"""

import sqlite3
import pandas as pd
import logging
from config import DB_PATH, RETURN_COLUMNS, OUTPUT_DIR
import os
from datetime import datetime

def score_predictions(pred_df, combo_name, target_col, threshold=0.01):
    if pred_df.empty or target_col not in pred_df:
        logging.warning(f"⚠️ No data to score for {combo_name} → {target_col}")
        return

    pred_df = pred_df.dropna(subset=[target_col])
    pred_df["success"] = (pred_df["prediction"] == 1) & (pred_df[target_col] > threshold)

    total_signals = pred_df["prediction"].sum()
    success_signals = pred_df["success"].sum()
    success_rate = (success_signals / total_signals) if total_signals else 0
    avg_return = pred_df.loc[pred_df["prediction"] == 1, target_col].mean()

    # Insert into DB
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback_scores (
            id INTEGER PRIMARY KEY,
            combo TEXT,
            target TEXT,
            total_signals INTEGER,
            success_rate REAL,
            avg_return REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        INSERT INTO feedback_scores (combo, target, total_signals, success_rate, avg_return)
        VALUES (?, ?, ?, ?, ?)
    """, (combo_name, target_col, int(total_signals), float(success_rate), float(avg_return)))
    conn.commit()
    conn.close()

    logging.info(f"📈 Feedback scored: {combo_name} | {target_col} | success_rate={success_rate:.2%}, avg_return={avg_return:.4f}")

    return {
        "combo": combo_name,
        "target": target_col,
        "signals": total_signals,
        "success_rate": success_rate,
        "avg_return": avg_return
    }


def run_feedback_scoring():
    logging.info("📊 Running feedback scoring over journals...")
    folder = os.path.join(OUTPUT_DIR, "journals")
    if not os.path.exists(folder):
        logging.warning("⚠️ No journal folder found.")
        return

    results = []
    for root, _, files in os.walk(folder):
        for file in files:
            if not file.endswith(".csv"): continue
            path = os.path.join(root, file)

            try:
                df = pd.read_csv(path)
                name = file.replace(".csv", "")
                parts = name.split("_")
                combo = "_".join(parts[:-2])
                target = "_".join(parts[-2:])

                summary = score_predictions(df, combo, target)
                if summary: results.append(summary)

            except Exception as e:
                logging.error(f"❌ Failed to process journal: {file} | {e}")

    if results:
        df_out = pd.DataFrame(results)
        out_path = os.path.join(OUTPUT_DIR, f"feedback_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df_out.to_csv(out_path, index=False)
        logging.info(f"✅ Feedback summary saved to: {out_path}")
    else:
        logging.warning("⚠️ No feedback scores computed.")

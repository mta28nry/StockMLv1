# mlearning/daily_journal.py

import logging
import os
from datetime import datetime

import pandas as pd

from config import OUTPUT_DIR


def save_daily_journal(df: pd.DataFrame, combo_name: str, target: str):
    """Saves summarized predictions per day into a timestamped journal folder."""
    if "Datetime" not in df.columns:
        logging.warning(f"⚠️ Missing 'Datetime' column. Cannot journal {combo_name}")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    journal_dir = os.path.join(OUTPUT_DIR, "journals", timestamp)
    os.makedirs(journal_dir, exist_ok=True)
    journal_file = os.path.join(journal_dir, f"{combo_name}_{target}.csv")

    try:
        df["key_0"] = pd.to_datetime(df["Datetime"]).dt.date

        if "predicted_signal" not in df.columns or "confidence" not in df.columns:
            logging.warning(f"⚠️ Missing columns for journal ({combo_name}). Writing placeholder.")
            df_empty = pd.DataFrame(columns=["key_0", "signals_triggered", "avg_confidence", "avg_predicted_return", "strategy", "target"])
            df_empty.to_csv(journal_file, index=False)
            return

        summary = (
            df.groupby("key_0")
              .agg({
                  "predicted_signal": "sum",
                  "confidence": "mean",
                  target: "mean"
              })
              .rename(columns={
                  "predicted_signal": "signals_triggered",
                  "confidence": "avg_confidence",
                  target: "avg_predicted_return"
              })
              .reset_index()
        )

        summary["strategy"] = combo_name
        summary["target"] = target
        summary.to_csv(journal_file, index=False)
        logging.info(f"📘 Journal saved: {journal_file}")

    except Exception as e:
        logging.error(f"❌ Failed to save journal for {combo_name}: {e}")

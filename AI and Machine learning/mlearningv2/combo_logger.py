# mlearning/combo_logger.py
import logging
import os

import pandas as pd

from config import OUTPUT_DIR


def log_top_combos(importance_df: pd.DataFrame, model_name: str, target: str, top_n: int = 6):
    """
    Save top N feature combos with importances for this model + target.
    """
    path = os.path.join(OUTPUT_DIR, f"top_combos_{model_name}_{target}.csv")
    df = importance_df.head(top_n).copy()
    df["combo"] = " + ".join(df["feature"])
    df.to_csv(path, index=False)
    logging.info(f"Top combo logged: {path}")

# mlearning/labels.py

"""
Adds target return labels based on config definitions.
Automatically skips existing columns.
"""

import pandas as pd
import logging
from config import RETURN_COLUMNS

def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("🔁 Generating target return columns...")

    horizon_map = {
        "return_5m": 1,
        "return_15m": 3,
        "return_30m": 6,
        "return_60m": 12,
        "return_240m": 48,
        "return_1440m": 288
    }

    for col in RETURN_COLUMNS:
        if col in df.columns:
            logging.info(f"⚠️ {col} already exists. Skipping generation.")
            continue

        period = horizon_map.get(col)
        if not period:
            logging.warning(f"⚠️ No mapping for {col}.")
            continue

        df[col] = df["Close"].pct_change(periods=period).shift(-period)
        logging.info(f"✅ Created label: {col} (shift={period})")

    return df

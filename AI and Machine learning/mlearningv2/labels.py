# mlearning/labels.py

"""
Generates target return labels based on defined horizons.
Clearly handles existing columns and adds additional volatility-based labels.
"""

import logging

import numpy as np
import pandas as pd

from config import RETURN_COLUMNS, create_logger

labels = create_logger("labels", log_to_file=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Clearly defined return horizons (periods based on 5-min intervals)
HORIZON_MAP = {
    "return_5m": 1,
    "return_15m": 3,
    "return_30m": 6,
    "return_60m": 12,
    "return_240m": 48,
    "return_1440m": 288
    }


def build_return_labels(df: pd.DataFrame, return_columns: list) -> pd.DataFrame:
    """
    Generate specified return labels based on horizons.
    """
    for col in return_columns:
        if col in df.columns:
            labels.info(f"⚠️ Column '{col}' already exists. Skipping.")
            continue

        period = HORIZON_MAP.get(col)
        if period is None:
            labels.warning(f"❌ No horizon mapping found for '{col}'. Skipped.")
            continue

        df[col] = df["Close"].pct_change(periods=period).shift(-period)
        labels.info(f"✅ Generated label '{col}' with horizon of {period} periods.")

    return df


def build_directional_labels(df: pd.DataFrame, return_columns: list, threshold: float = 0.001) -> pd.DataFrame:
    """
    Generate directional labels based on thresholds for binary classification tasks.
    """
    for col in return_columns:
        direction_col = f"{col}_direction"
        if direction_col in df.columns:
            labels.info(f"⚠️ Column '{direction_col}' already exists. Skipping.")
            continue

        df[direction_col] = np.where(
            df[col] > threshold, 1,
            np.where(df[col] < -threshold, -1, 0)
            )
        labels.info(f"✅ Generated directional label '{direction_col}' using threshold {threshold}.")

    return df


def build_volatility_labels(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Generate labels representing future volatility for risk assessment.
    """
    vol_col = f"volatility_{window}bars"
    if vol_col in df.columns:
        labels.info(f"⚠️ Column '{vol_col}' already exists. Skipping volatility calculation.")
        return df

    df[vol_col] = df["Close"].rolling(window=window, min_periods=1).std().shift(-window)
    labels.info(f"✅ Generated volatility label '{vol_col}' over {window}-bar window.")

    return df


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main orchestrator to generate all labels clearly.
    """
    labels.info("🚩 Starting comprehensive label generation...")

    df = build_return_labels(df, RETURN_COLUMNS)
    df = build_directional_labels(df, RETURN_COLUMNS)
    df = build_volatility_labels(df, window=20)

    labels.info("✅ All labels generated successfully.")
    return df

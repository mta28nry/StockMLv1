# mlearning/recommender.py

"""
recommender.py

Purpose:
- Load trained model and generate predictions on full dataset
- Used to simulate strategy behavior in practice
- Adds prediction and confidence columns
"""

import logging
import os

import joblib
import pandas as pd


def recommend_trades(model_path, df: pd.DataFrame, features: list):
    if not os.path.exists(model_path):
        logging.error(f"❌ Model not found: {model_path}")
        return df

    model = joblib.load(model_path)
    df = df.copy()

    try:
        df["prediction"] = model.predict(df[features])
        if hasattr(model, "predict_proba"):
            df["confidence"] = model.predict_proba(df[features])[:, 1]
        else:
            df["confidence"] = 0.5  # fallback
    except Exception as e:
        logging.error(f"❌ Failed to generate predictions: {e}")
        df["prediction"] = 0
        df["confidence"] = 0.0

    logging.info(f"✅ Predictions generated: {len(df)} rows")
    return df

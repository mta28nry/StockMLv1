# mlearning/config.py
"""
Module: config.py
Purpose:
- Define central paths and directories
- Set model options and time horizons
- Tag features by category (used by SHAP/evolution)
"""

import os

# --- Base Directory ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# --- Data Paths ---
DATA_PATH = os.path.join(BASE_DIR, "historical data_5min_ml_ready.csv")
LOG_DIR = os.path.join(BASE_DIR, "Logs")
MODEL_DIR = os.path.join(BASE_DIR, "Models")
OUTPUT_DIR = os.path.join(BASE_DIR, "Output")
DB_PATH = os.path.join(BASE_DIR, "strategy_meta.db")

# --- Auto-create required folders ---
for path in [LOG_DIR, MODEL_DIR, OUTPUT_DIR, os.path.dirname(DB_PATH)]:
    os.makedirs(path, exist_ok=True)

# --- Model Types ---
CLASSIFIERS = ["RandomForest", "XGBoost", "LogisticRegression"]
REGRESSORS = ["RandomForest", "XGBoost"]

# --- Targets (Multi-Horizon Returns) ---
RETURN_COLUMNS = ["return_5m", "return_15m", "return_30m", "return_60m", "return_240m", "return_1440m"]

# --- Strategy Features (Used in stacking) ---
STRATEGY_FEATURES = [
    "Close < EMA_8", "Close < EMA_21", "Close < orbLow", "Close < premarketLow",
    "Close < VWAP", "RSI < 30", "MACD < MACD_signal"
]

# --- Feature Categories for SHAP Logging + Mutation ---
FEATURE_TYPES = {
    # Time-based
    "hour_sin": "time", "hour_cos": "time", "session": "time",
    "minute_bin": "time", "is_opening_bar": "time",

    # Price/level based
    "distance_orbLow": "level", "distance_orbHigh": "level",
    "dist_to_prevClose": "level", "below_orbLow": "level",
    "above_orbHigh": "level", "below_prevLow": "level",

    # Indicators
    "rsi_low": "indicator", "rsi_high": "indicator",
    "macd_cross_up": "indicator", "macd_cross_down": "indicator",
    "ema8_cross_ema21": "indicator", "ema21_cross_ema90": "indicator",

    # Trend
    "ema8_gt_ema21": "trend", "ema21_gt_ema90": "trend",

    # Volume
    "vol_spike": "volume", "volume_change": "volume",

    # Candlestick structure
    "body": "structure", "body_pct": "structure",
    "wick": "structure", "tail": "structure",
    "wick_to_body": "structure", "tail_to_body": "structure"
}

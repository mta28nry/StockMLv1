"""
config.py — Centralized Environment, Defaults, and Logging Configuration

Purpose:
- Auto-create directories and default paths
- Provide structured logger creation
- Fallback default config if YAML is not used
- Define supported models and feature categories
"""

import logging
import os

# ---------------------------
# 📁 Directory and Path Setup
# ---------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")
MODEL_DIR = os.path.join(BASE_DIR, "Models")
OUTPUT_DIR = os.path.join(BASE_DIR, "Output")
LOG_DIR = os.path.join(BASE_DIR, "Logs")
DB_PATH = os.path.join(BASE_DIR, "strategy_meta.db")
DATA_PATH = os.path.join(DATA_DIR, "")

# Ensure essential folders exist
for path in [DATA_DIR, MODEL_DIR, OUTPUT_DIR, LOG_DIR, os.path.dirname(DB_PATH)]:
    os.makedirs(path, exist_ok=True)

# ---------------------------
# 🪵 Centralized Logging with Prefix and Dual Output
# ---------------------------
import traceback

def create_logger(name: str, level=logging.INFO, log_to_file=True, log_to_console=False) -> logging.Logger:
    """
    Create a prefixed logger that outputs to both console and unified file log.
    All logs include source (module/phase) prefix.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Prevent duplicate handler setup

    logger.setLevel(level)
    prefix_fmt = logging.Formatter(f"%(asctime)s | %(levelname)s | {name} | %(message)s")


    # Optional Console Output
    if log_to_console:
        stream = logging.StreamHandler()
        stream.setFormatter(prefix_fmt)
        logger.addHandler(stream)

    # Always write to file if enabled
    if log_to_file:
        log_path = os.path.join(LOG_DIR, f"debug_logs.log")
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(prefix_fmt)
        logger.addHandler(file_handler)

    return logger

# Logger for standard modules
info_logger     = create_logger("INFO", logging.INFO)
debug_logger    = create_logger("DEBUG", logging.DEBUG)
error_logger    = create_logger("ERROR", logging.ERROR)

# Error log (separate file with only errors)
if not error_logger.handlers or all(type(h) != logging.FileHandler or h.baseFilename != os.path.join(LOG_DIR, "ERRORS.log")
                                    for h in error_logger.handlers):
    error_file = logging.FileHandler(os.path.join(LOG_DIR, "ERRORS.log"), encoding="utf-8")
    error_file.setLevel(logging.ERROR)
    error_file.setFormatter(logging.Formatter("%(asctime)s | ERROR | %(name)s | %(message)s"))
    error_logger.addHandler(error_file)





# ---------------------------
# 📦 Default Configuration Fallback
# ---------------------------
def get_default_config():
    """
    Default fallback configuration for full ML pipeline (if no YAML provided).
    Ensures all training, evaluation, optimization, and logging settings are available.
    """
    return {
        "task": "train",
        "symbol": "ALL",  # Can be used for symbol-specific runs later
        "data_path": str(DATA_PATH),
        "db_path": str(DB_PATH),
        "log_dir": str(LOG_DIR),
        "output_dir": str(OUTPUT_DIR),
        "model_dir": str(MODEL_DIR),

        "models": {
            "classifiers": ["RandomForest", "XGBoost", "LogisticRegression", "LightGBM"],
            "regressors": ["RandomForest", "XGBoost", "LightGBM"]
        },

        "training": {
            "epochs": 20,
            "batch_size": 32,
            "learning_rate": 0.001,
            "use_gpu": True,
            "early_stopping": 20,
            "verbose": True,
            "optimizer": "adam",
            "shuffle": True,
            "validation_split": 0.2,
            "random_state": 42
        },

        "features": {
            "range": "return_5m",
            "feature_list": [
                "above_orbHigh", "below_orbLow", "above_ema8", "below_ema8",
                "above_ema21", "below_ema21", "above_premarketHigh", "below_premarketLow",
                "above_prevHigh", "below_prevLow", "vol_spike", "ema8_gt_ema21",
                "Close < EMA_8", "Close < EMA_21", "RSI < 30", "MACD < MACD_signal"
            ]
        },

        "hyperparameter_optimization": {
            "enabled": False,
            "method": "grid_search",
            "n_iterations": 25,
            "scoring": "accuracy",
            "cv_folds": 5,
            "hyperparameters": {
                "learning_rate": [0.001, 0.01, 0.1],
                "batch_size": [32, 64],
                "epochs": [10, 20, 50]
            }
        },

        "evaluate": {
            "enabled": True,
            "metric": "accuracy",
            "top_n": 10,
            "save_results": True,
            "save_as_csv": True,
            "csv_path": os.path.join(LOG_DIR, "evaluation_results.csv")
        },

        "logging": {
            "log_level": "INFO",
            "log_file": os.path.join(LOG_DIR, "training.log"),
            "error_log_file": os.path.join(LOG_DIR, "errors.log"),
            "log_to_console": True,
            "log_to_file": True,
            "log_format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        },

        "reporting": {
            "enabled": True,
            "export_yaml": True,
            "readme_path": os.path.join(OUTPUT_DIR, "run_summary.md"),
            "summary_plot_path": os.path.join(OUTPUT_DIR, "summary.png")
        },

        "visualization": {
            "enabled": True,
            "output_path": os.path.join(OUTPUT_DIR, "visuals"),
            "show_plots": False,
            "save_plots": True
        }
    }

# ---------------------------
# ✅ Constant Model + Feature Lists
# ---------------------------
CLASSIFIERS = ["RandomForest", "XGBoost", "LogisticRegression", "LightGBM"]
REGRESSORS = ["RandomForest", "XGBoost", "LightGBM"]
RETURN_COLUMNS = ["return_5m", "return_15m", "return_30m", "return_60m", "return_240m", "return_1440m"]

STRATEGY_FEATURES = [
    "Close < EMA_8", "Close < EMA_21", "Close < orbLow", "Close < premarketLow",
    "Close < VWAP", "RSI < 30", "MACD < MACD_signal"
]

FEATURE_TYPES = {
    # Time
    "hour_sin": "time", "hour_cos": "time", "session": "time",
    "minute_bin": "time", "is_opening_bar": "time",

    # Price levels
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

    # Candle structure
    "body": "structure", "body_pct": "structure",
    "wick": "structure", "tail": "structure",
    "wick_to_body": "structure", "tail_to_body": "structure"
}

# -------------------------------
# 🧠 Hardware Diagnostics Utilities
# -------------------------------


def get_gpu_info():
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if not gpus:
            return False, []
        info = [f"{gpu.name} | Mem: {gpu.memoryTotal}MB | Load: {gpu.load*100:.1f}%" for gpu in gpus]
        return True, info
    except Exception as e:
        error_logger.error(f"❌ GPU detection failed: {e}")
        return False, []

def get_cpu_info():
    import platform
    import psutil
    return {
        "cpu_name": platform.processor(),
        "cpu_cores": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 1024**3, 2)
    }



def log_traceback(logger: logging.Logger, prefix: str, exc: Exception):
    """
    Logs a formatted traceback with optional prefix
    """
    tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    for line in tb_lines:
        logger.error(f"{prefix} | {line.strip()}")




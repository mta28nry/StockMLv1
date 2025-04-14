"""
master.py — Self-Improving ML Engine Orchestrator (Sequential Execution)
Executes all ML phases with diagnostics, GPU/CPU validation, SHAP eval, combo optimization, and logging.
"""

import os
import uuid
from datetime import datetime

import pandas as pd

from auto_suggester import suggest_top_combos
from autotune_self import autotune
from clean_db import run_db_cleanup
from combo_autogen import generate_shap_combos
from combo_config import load_combo_rules
from combo_mutation import evolve_combos
from config import (
    BASE_DIR, RETURN_COLUMNS, create_logger, error_logger, get_cpu_info, get_default_config, get_gpu_info, log_traceback
    )
from features import build_features
from feedback_trainer import run_feedback_scoring
from labels import build_labels
from model_trainer import train_each_indicator_all_models_full
from parallel_trainer import run_parallel_training
from report_generator import generate_report, generate_run_readme
from self_improver import run_self_improver
from strategy_meta import init_db, init_self_learning_tables, init_tuning_table
from visualize import generate_dashboard_summary
from walkforward import run_walkforward_backtest


# ---------------------- 📂 Config Path ----------------------
#CONFIG_PATH = os.path.join(BASE_DIR, "configs", "")

# ---------------------- 🧠 Runtime Metadata ----------------------
master = create_logger("master", log_to_file=True)
RUN_ID = str(uuid.uuid4())[:8]
START_TIME = datetime.now()
errors = []

# ---------------------- 🧠 Hardware Info ----------------------
GPU_AVAILABLE, GPU_INFO = get_gpu_info()
CPU_INFO = get_cpu_info()

# ---------------------- ⚙️ Config Load ----------------------

CONFIG = get_default_config()
master.info("⚙️ Using built-in default configuration.")



master.info("⚙️ CONFIGURATION SUMMARY:")
for k, v in CONFIG.items():
    if isinstance(v, dict):
        master.info(f"  {k}:")
        for sk, sv in v.items():
            master.info(f"     {sk}: {sv}")
    else:
        master.info(f"  {k}: {v}")
# def load_config_or_default(path):
#     try:
#         with open(path, 'r') as file:
#             config = yaml.safe_load(file)
#         master.info(f"📄 Loaded config from: {path}")
#         return config
#     except Exception as e:
#         error_logger.error(f"⚠️ Failed loading config: {e}")
#         return get_default_config()
#
# CONFIG = load_config_or_default(CONFIG_PATH)
#
# master.info("⚙️ CONFIGURATION SUMMARY:")
# for k, v in CONFIG.items():
#     if isinstance(v, dict):
#         master.info(f"  {k}:")
#         for sk, sv in v.items():
#             master.info(f"     {sk}: {sv}")
#     else:
#         master.info(f"  {k}: {v}")

# ---------------------- ✅ Preflight Check ----------------------
def run_preflight_check():
    master.info("🔎 PHASE -1: PREFLIGHT CHECK")

    # 🧠 config or strategy_meta.py
    import sqlite3
    from config import DB_PATH

    def add_symbol_column_if_missing():
        """
        Adds 'symbol' column to all key tables if missing.
        """
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        tables = ["meta", "shap_features", "indicator_scores"]

        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [col[1] for col in cursor.fetchall()]
            if "symbol" not in columns:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN symbol TEXT")
                print(f"✅ Added 'symbol' column to '{table}' table.")

        conn.commit()
        conn.close()

    #
    # if not os.path.exists(CONFIG_PATH):
    #     error_logger.error(f"❌ YAML Config not found at: {CONFIG_PATH}")
    #     raise FileNotFoundError(f"YAML config missing: {CONFIG_PATH}")

    try:
        init_db()
        init_self_learning_tables()
        init_tuning_table()
        master.info("✅ Database initialized and verified.")
    except Exception as e:
        log_traceback(error_logger, "DB INIT", e)
        raise

    master.info("🧪 SYSTEM SUMMARY:")
    master.info(f"🎯 Run ID: {RUN_ID}")
    master.info(f"🧠 CPU: {CPU_INFO['cpu_name']} | Cores: {CPU_INFO['cpu_cores']} | RAM: {CPU_INFO['ram_gb']} GB")
    master.info(f"🎮 GPU Available: {GPU_AVAILABLE}")
    if GPU_AVAILABLE:
        for g in GPU_INFO:
            master.info(f"   → {g}")
    master.info("✅ Preflight Check Complete")

# ---------------------- 🔁 Phase Functions ----------------------
def run_indicator_eval(symbol, path):
    try:
        master.info(f"[PHASE 0] Indicator Evaluation → {symbol}")
        df = pd.read_csv(path)
        df = build_features(df)
        df = build_labels(df)
        df.dropna(inplace=True)

        train_each_indicator_all_models_full(
            df=df,
            config=CONFIG,
            symbol=symbol,
            export_csv=True,
            csv_path=os.path.join(CONFIG["log_dir"], f"solo_scores_{symbol}.csv")
        )

        master.info("✅ PHASE 0 COMPLETE")
    except Exception as e:
        log_traceback(master, "PHASE 0", e)
        errors.append(f"{symbol}-PHASE0")


def run_baseline():
    try:
        master.info("🔻 PHASE 1: BASELINE TRAINING")
        df = pd.read_csv(CONFIG["data_path"])
        df = build_features(df)
        df = build_labels(df)
        df.dropna(inplace=True)
        combos = list(load_combo_rules().values())
        if not combos:
            master.warning("⚠️ No strategy combos loaded.")
            return
        for target in RETURN_COLUMNS:
            run_parallel_training(df.copy(), combos, target, regression=False, prune=True, journal=True, show_plot=False, n_jobs=os.cpu_count())
        master.info("✅ PHASE 1 COMPLETE")
    except Exception as e:
        log_traceback(error_logger, "PHASE 1", e)
        errors.append("Baseline Training")

def run_shap_evaluation():
    try:
        master.info("🔻 PHASE 2: SHAP EVALUATION")
        generate_shap_combos()
        master.info("✅ PHASE 2 COMPLETE")
    except Exception as e:
        log_traceback(error_logger, "PHASE 2", e)
        errors.append("SHAP Evaluation")

def run_autotuning():
    try:
        master.info("🔻 PHASE 3: AUTO-TUNING")
        autotune()
        master.info("✅ PHASE 3 COMPLETE")
    except Exception as e:
        log_traceback(error_logger, "PHASE 3", e)
        errors.append("Auto-Tuning")

def run_feedback():
    try:
        master.info("🔻 PHASE 4: FEEDBACK SCORING")
        run_feedback_scoring()
        master.info("✅ PHASE 4 COMPLETE")
    except Exception as e:
        log_traceback(error_logger, "PHASE 4", e)
        errors.append("Feedback Scoring")

def run_combo_autogen():
    try:
        master.info("🔻 PHASE 5: SHAP COMBO AUTOGEN")
        generate_shap_combos()
        master.info("✅ PHASE 5 COMPLETE")
    except Exception as e:
        log_traceback(error_logger, "PHASE 5", e)
        errors.append("SHAP Combo Autogen")

def run_evolution():
    try:
        master.info("🔻 PHASE 6: STRATEGY EVOLUTION")
        evolve_combos()
        master.info("✅ PHASE 6 COMPLETE")
    except Exception as e:
        log_traceback(error_logger, "PHASE 6", e)
        errors.append("Strategy Evolution")

def run_walkforward():
    try:
        master.info("🔻 PHASE 7: WALKFORWARD BACKTEST")
        run_walkforward_backtest()
        master.info("✅ PHASE 7 COMPLETE")
    except Exception as e:
        log_traceback(error_logger, "PHASE 7", e)
        errors.append("Walkforward Backtest")

def run_visuals():
    try:
        master.info("🔻 PHASE 8: VISUALIZATION")
        generate_dashboard_summary()
        master.info("✅ PHASE 8 COMPLETE")
    except Exception as e:
        log_traceback(error_logger, "PHASE 8", e)
        errors.append("Visualization")

def run_suggestions():
    try:
        master.info("🔻 PHASE 9: STRATEGY SUGGESTIONS")
        suggest_top_combos(top_n=10)
        master.info("✅ PHASE 9 COMPLETE")
    except Exception as e:
        log_traceback(error_logger, "PHASE 9", e)
        errors.append("Strategy Suggestions")

def run_reporting():
    try:
        master.info("🔻 PHASE 10: REPORTING")
        summary = generate_report()
        generate_run_readme(summary)
        master.info("✅ PHASE 10 COMPLETE")
    except Exception as e:
        log_traceback(error_logger, "PHASE 10", e)
        errors.append("Reporting")

def run_maintenance():
    try:
        master.info("🔻 PHASE 11: MAINTENANCE")
        run_db_cleanup()
        master.info("✅ PHASE 11 COMPLETE")
    except Exception as e:
        log_traceback(error_logger, "PHASE 11", e)
        errors.append("Maintenance")



import glob
import re

# ---------------------- 🧠 Dataset Scanner ----------------------

def find_all_datasets(data_root: str):
    """
    Locate all *_ml_ready.csv files under subfolders of data_root.
    Returns: List of (symbol, timeframe, full_path)
    """
    csvs = glob.glob(os.path.join(data_root, "**", "*_ml_ready.csv"), recursive=True)
    found = []
    for path in csvs:
        filename = os.path.basename(path)
        folder = os.path.basename(os.path.dirname(path))

        # Try symbol from filename prefix (e.g., "NVDA_5min_ml_ready.csv")
        match = re.match(r"([A-Z]+)[_-](\d{1,2}(min|hr|h|m))", filename, re.IGNORECASE)
        if match:
            symbol = match.group(1).upper()
            tf = match.group(2).lower()
        else:
            # fallback: use folder
            symbol = folder.upper()
            tf = "unknown"

        found.append((symbol, tf, path))
    return found


# ---------------------- 🚀 MAIN ----------------------

def main():
    datasets = find_all_datasets(os.path.join(BASE_DIR, "Data"))
    master.info(f"🔍 Found {len(datasets)} dataset(s)")

    for symbol, tf, path in datasets:
        master.info(f"\n📂 NEW RUN → {symbol} | TF: {tf} | {path}")

        try:
            CONFIG["symbol"] = symbol
            CONFIG["timeframe"] = tf
            CONFIG["data_path"] = path

            run_preflight_check()

            for phase in [
                lambda: run_indicator_eval(symbol, path),
                run_baseline,
                run_shap_evaluation,
                run_autotuning,
                run_feedback,
                run_combo_autogen,
                run_self_improver,
                run_suggestions,
                run_walkforward,
                run_evolution,
                run_reporting,
                run_visuals,
                run_maintenance
                ]:
                phase()


        except Exception as e:
            master.exception(f"❌ Failure: {symbol}-{tf}: {e}")
            errors.append(f"{symbol}-{tf}")

    master.info("✅ ALL DATASETS COMPLETE")
    if errors:
        master.warning("⚠️ GLOBAL ERRORS:")
        for err in errors:
            master.warning(f" • {err}")

# ---------------------- ⏯️ Entry ----------------------
if __name__ == "__main__":
    main()
"""
master.py — Self-Improving ML Engine Orchestrator
Enhancements:
- Phase-Level Error Aggregation
- Unique Pipeline Run ID Logging
- Parallel Phase Execution Support
"""

import uuid
import time
import pandas as pd
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from config import DATA_PATH, RETURN_COLUMNS
from features import build_features
from labels import build_labels
from combo_config import load_combo_rules
from parallel_trainer import run_parallel_training
from strategy_meta import (
    init_db,
    init_tuning_table,
    init_self_learning_tables
)
from auto_suggester import suggest_top_combos
from autotune_self import autotune
from combo_mutation import evolve_combos
from combo_autogen import generate_shap_combos
from self_improver import run_self_improver
from walkforward import run_walkforward_backtest
from feedback_trainer import run_feedback_scoring
from report_generator import generate_report, generate_run_readme
from visualize import generate_dashboard_summary
from clean_db import run_db_cleanup
from model_trainer import train_each_indicator_all_models
from log_config import get_logger

logger = get_logger("master")

# -----------------------
# 🔁 Phase Definitions
# -----------------------

def run_indicator_eval():
    logger.info("🔻 PHASE 0: INDICATOR EVALUATION")
    df = pd.read_csv(DATA_PATH)
    df = build_features(df)
    df = build_labels(df)
    df.dropna(inplace=True)
    train_each_indicator_all_models(df, target_col="return_15m", task="regression")
    logger.info("✅ PHASE 0 COMPLETE")

def run_baseline():
    logger.info("🔻 PHASE 1: BASELINE COMBO TRAINING")
    df = pd.read_csv(DATA_PATH)
    df = build_features(df)
    df = build_labels(df)
    df.dropna(inplace=True)

    combos = list(load_combo_rules().values())
    if not combos:
        logger.warning("⚠️ No strategy combos found.")
        return

    for target in RETURN_COLUMNS:
        logger.info(f"📈 Training for {target} using {len(combos)} combos")
        run_parallel_training(
            df.copy(), combos, target,
            regression=False, prune=True,
            journal=True, show_plot=False,
            n_jobs=4
        )
    logger.info("✅ PHASE 1 COMPLETE")

def run_autotuning():
    logger.info("🔻 PHASE 2: AUTO-TUNING")
    autotune()
    logger.info("✅ PHASE 2 COMPLETE")

def run_feedback():
    logger.info("🔻 PHASE 3: FEEDBACK SCORING")
    run_feedback_scoring()
    logger.info("✅ PHASE 3 COMPLETE")

def run_combo_autogen():
    logger.info("🔻 PHASE 4: SHAP COMBO AUTOGEN")
    generate_shap_combos()
    logger.info("✅ PHASE 4 COMPLETE")

def run_evolution():
    logger.info("🔻 PHASE 5: STRATEGY EVOLUTION")
    evolve_combos()
    logger.info("✅ PHASE 5 COMPLETE")

def run_walkforward():
    logger.info("🔻 PHASE 6: WALKFORWARD BACKTEST")
    run_walkforward_backtest()
    logger.info("✅ PHASE 6 COMPLETE")

def run_visuals():
    logger.info("🔻 PHASE 7: PERFORMANCE VISUALIZATION")
    generate_dashboard_summary()
    logger.info("✅ PHASE 7 COMPLETE")

def run_suggestions():
    logger.info("🔻 PHASE 8: STRATEGY SUGGESTIONS")
    suggest_top_combos(top_n=10)
    logger.info("✅ PHASE 8 COMPLETE")

def run_reporting():
    logger.info("🔻 PHASE 9: REPORTING")
    summary = generate_report()
    generate_run_readme(summary)
    logger.info("✅ PHASE 9 COMPLETE")

def run_maintenance():
    logger.info("🔻 PHASE 10: DB MAINTENANCE")
    run_db_cleanup()
    logger.info("✅ PHASE 10 COMPLETE")

# -----------------------
# 🧠 Main Execution
# -----------------------

def main():
    run_id = f"RUN-{uuid.uuid4().hex[:8].upper()}"
    logger.info(f"🧠 INITIATING SELF-LEARNING ML ENGINE — {run_id}")
    error_phases = []

    # Init DBs
    try:
        logger.info("🔧 Initializing metadata tables...")
        init_db()
        init_self_learning_tables()
        init_tuning_table()
    except Exception as e:
        logger.exception("❌ DB Initialization Failed")
        return

    # Define all phases in order
    phase_tasks = [
        ("Indicator Eval", run_indicator_eval),
        ("Baseline Training", run_baseline),
        ("Auto Tuning", run_autotuning),
        ("Feedback Scoring", run_feedback),
        ("SHAP Combo Autogen", run_combo_autogen),
        ("Self Improver", run_self_improver),
        ("Suggestions", run_suggestions),
        ("Walkforward", run_walkforward),
        ("Evolution", run_evolution),
        ("Reporting", run_reporting),
        ("Visualization", run_visuals),
        ("Maintenance", run_maintenance)
    ]

    # Run each phase and track errors
    start = time.time()
    for phase_name, phase_fn in phase_tasks:
        try:
            logger.info(f"🔹 START: {phase_name}")
            phase_fn()
        except Exception as e:
            error_phases.append((phase_name, str(e)))
            logger.error(f"❌ PHASE FAILED: {phase_name}")
            logger.error(traceback.format_exc())

    duration = time.time() - start

    # Final summary
    logger.info(f"🧾 RUN COMPLETE: {run_id}")
    logger.info(f"⏱️ Total Duration: {duration:.2f}s")
    if error_phases:
        logger.warning(f"⚠️ {len(error_phases)} phase(s) failed:")
        for phase, err in error_phases:
            logger.warning(f"   ❌ {phase}: {err}")
    else:
        logger.info("🎉 ALL PHASES SUCCESSFUL.")

# -----------------------
if __name__ == "__main__":
    main()

# mlearning/master.py

"""
Master Pipeline Controller (Self-Improving ML Engine)

Phases:
1. Baseline Model Training
2. Auto-Tuning
3. Feedback Scoring
4. SHAP Combo Auto-Generation
5. Combo Evolution
6. Walkforward Backtesting
7. SHAP/Score Visualization
8. Top Combo Suggestions
9. Report Summary
10. DB Cleanup + Maintenance
"""

import logging
import pandas as pd
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

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
)

def run_baseline():
    logging.info("🔻 PHASE 1: BASELINE TRAINING")
    try:
        logging.info("📊 Loading dataset...")
        df = pd.read_csv(DATA_PATH)
        df = build_features(df)
        df = build_labels(df)
        df.dropna(inplace=True)

        logging.info(f"✅ Dataset ready: {df.shape[0]} rows, {df.shape[1]} columns")
        combos = list(load_combo_rules().values())
        if not combos:
            logging.warning("⚠️ No strategy combos loaded. Check YAML.")
            return

        logging.info(f"✅ Loaded {len(combos)} combos")

        for target in RETURN_COLUMNS:
            logging.info(f"📈 Training combos for target: {target}")
            run_parallel_training(
                df.copy(), combos, target,
                regression=False,
                prune=True,
                journal=True,
                show_plot=False,
                n_jobs=4
            )

        logging.info("✅ PHASE 1 COMPLETE: Baseline training finished.")

    except Exception as e:
        logging.exception(f"❌ Baseline training failed: {e}")


def run_autotuning():
    logging.info("🔻 PHASE 2: AUTO-TUNING")
    try:
        autotune()
        logging.info("✅ PHASE 2 COMPLETE: Auto-Tuning done.")
    except Exception as e:
        logging.exception("❌ Auto-Tuning failed.")


def run_feedback():
    logging.info("🔻 PHASE 3: FEEDBACK SCORING")
    try:
        run_feedback_scoring()
        logging.info("✅ PHASE 3 COMPLETE: Feedback scoring complete.")
    except Exception as e:
        logging.exception("❌ Feedback scoring failed.")


def run_combo_autogen():
    logging.info("🔻 PHASE 4: SHAP COMBO AUTOGEN")
    try:
        generate_shap_combos()
        logging.info("✅ PHASE 4 COMPLETE: SHAP combo generation done.")
    except Exception as e:
        logging.exception("❌ SHAP combo generation failed.")


def run_evolution():
    logging.info("🔻 PHASE 5: STRATEGY EVOLUTION")
    try:
        evolve_combos()
        logging.info("✅ PHASE 5 COMPLETE: Combo evolution complete.")
    except Exception as e:
        logging.exception("❌ Evolution phase failed.")


def run_walkforward():
    logging.info("🔻 PHASE 6: WALKFORWARD BACKTESTING")
    try:
        run_walkforward_backtest()
        logging.info("✅ PHASE 6 COMPLETE: Walkforward evaluation complete.")
    except Exception as e:
        logging.exception("❌ Walkforward phase failed.")


def run_visuals():
    logging.info("🔻 PHASE 7: PERFORMANCE VISUALIZATION")
    try:
        generate_dashboard_summary()
        logging.info("✅ PHASE 7 COMPLETE: Visuals saved.")
    except Exception as e:
        logging.exception("❌ Visualization generation failed.")


def run_suggestions():
    logging.info("🔻 PHASE 8: STRATEGY SUGGESTIONS")
    try:
        suggest_top_combos(top_n=10)
        logging.info("✅ PHASE 8 COMPLETE: Suggestions retrieved.")
    except Exception as e:
        logging.exception("❌ Suggestion phase failed.")


def run_reporting():
    logging.info("🔻 PHASE 9: REPORT GENERATION")
    try:
        generate_report()
        logging.info("✅ PHASE 9 COMPLETE: Report generated.")
    except Exception as e:
        logging.exception("❌ Report generation failed.")


def run_maintenance():
    logging.info("🔻 PHASE 10: DB MAINTENANCE")
    try:
        run_db_cleanup()
        logging.info("✅ PHASE 10 COMPLETE: Cleanup finished.")
    except Exception as e:
        logging.exception("❌ DB cleanup failed.")


def main():
    logging.info("🛠 SYSTEM STARTUP — Initializing metadata tables...")
    init_db()
    init_self_learning_tables()
    init_tuning_table()
    logging.info("✅ DBs initialized.")

    logging.info("🚀 STARTING FULL ML PIPELINE...")
    run_baseline()
    run_autotuning()
    run_feedback()
    run_combo_autogen()
    run_self_improver()
    run_suggestions()
    run_walkforward()
    run_evolution()
    run_reporting()
    summary_meta_dict = generate_report()
    generate_run_readme(summary_meta_dict)
    run_visuals()
    run_maintenance()
    logging.info("🏁 ALL STAGES COMPLETE — Self-Learning Pipeline Finished.")


if __name__ == "__main__":
    main()

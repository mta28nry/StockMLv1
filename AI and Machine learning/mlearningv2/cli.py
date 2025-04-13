# mlearning/cli.py
import argparse
import logging
import pandas as pd
import itertools
import os
from config import *
from features import build_features
from labels import build_labels
from model_trainer import train_and_save
from evaluate import explain_with_shap, plot_heatmap
from combo_logger import log_top_combos
from walkforward import walk_forward_validation
from strategies import evaluate_strategy
from recommender import recommend_trades
from combo_config import load_combo_rules
from regression_utils import train_regression
from daily_journal import save_daily_journal
from boruta_prune import prune_combo

logging.basicConfig(level=logging.INFO)


def generate_combos(features, max_len=4):
    """Generate all feature condition combinations up to max_len"""
    all_combos = []
    for r in range(2, max_len + 1):
        all_combos += list(itertools.combinations(features, r))
    return all_combos


def run_combo_training(regression=False, prune=False, journal=False, use_config=False, show_plot=False):
    df = pd.read_csv(DATA_PATH)
    df = build_features(df)
    df = build_labels(df)
    df = df.dropna()

    combos = []
    if use_config:
        rules = load_combo_rules()
        combos = [v for v in rules.values()]
    else:
        all_bools = [c for c in df.columns if df[c].dtype == bool]
        combos = generate_combos(all_bools, max_len=4)

    for ret_col in RETURN_COLUMNS:
        for combo in combos:
            combo_name = "_".join(combo)
            logging.info(f"🧠 Training on combo: {combo_name} → Target: {ret_col}")

            signal = evaluate_strategy(df, list(combo))
            df["combo_signal"] = signal

            if signal.sum() < 10:
                logging.info("❌ Skipping: insufficient triggered samples")
                continue

            X = df[list(combo)].copy()
            if regression:
                y = df[ret_col].copy()
                model_path, model = train_regression(X, y, combo_name, ret_col)
            else:
                y = (df[ret_col] > 0).astype(int)
                train_and_save(X, y, "XGBoost", task="classification", suffix=f"{combo_name}_{ret_col}")
                model_path = os.path.join(MODEL_DIR, f"XGBoost_classification_{combo_name}_{ret_col}.pkl")
                model = recommend_trades(model_path, X, list(combo))

            shap_df = explain_with_shap(model, X, "XGBoost", suffix=f"{combo_name}_{ret_col}", show=show_plot)

            if prune and prune_combo(shap_df, combo_name=combo_name, target=ret_col):
                continue

            log_top_combos(shap_df, "XGBoost", f"{combo_name}_{ret_col}", top_n=3)

            if journal:
                df_preds = recommend_trades(model_path, df, list(combo))
                df_preds[ret_col] = df[ret_col]
                save_daily_journal(df_preds, combo_name, ret_col)


def cli_main():
    parser = argparse.ArgumentParser(description="Run ML pipeline for trade strategy discovery")
    parser.add_argument("--train", action="store_true", help="Train classification models")
    parser.add_argument("--combo", action="store_true", help="Train all strategy combinations")
    parser.add_argument("--walk", action="store_true", help="Run walk-forward validation")
    parser.add_argument("--recommend", action="store_true", help="Run prediction with a model")

    # New flags
    parser.add_argument("--regression", action="store_true", help="Enable regression training mode")
    parser.add_argument("--journal", action="store_true", help="Log daily strategy predictions")
    parser.add_argument("--prune", action="store_true", help="Prune combos with low SHAP importance")
    parser.add_argument("--config", action="store_true", help="Use combo_rules.yaml for training combos")
    parser.add_argument("--visual", action="store_true", help="Enable SHAP plot display during training")

    args = parser.parse_args()

    if args.combo:
        run_combo_training(
            regression=args.regression,
            prune=args.prune,
            journal=args.journal,
            use_config=args.config,
            show_plot=args.visual
        )

    elif args.train:
        logging.info("Basic classification training is deprecated. Use --combo.")
    elif args.walk:
        logging.info("Running walk-forward...")
        from sklearn.ensemble import RandomForestClassifier
        df = pd.read_csv(DATA_PATH)
        df = build_features(df)
        df = build_labels(df)
        df = df.dropna()
        features = ["below_ema8", "below_orb", "rsi_low"]
        y = (df["return_15m"] > 0).astype(int)
        results = walk_forward_validation(RandomForestClassifier(), df, features, y)
        results.to_csv(f"{OUTPUT_DIR}/walkforward_results.csv", index=False)
    elif args.recommend:
        logging.info("Running recommender...")
        df = pd.read_csv(DATA_PATH)
        df = build_features(df)
        model_path = f"{MODEL_DIR}/XGBoost_classification_below_ema8_below_orb_return_15m.pkl"
        result_df = recommend_trades(model_path, df, ["below_ema8", "below_orb"])
        result_df.to_csv(f"{OUTPUT_DIR}/recommended_trades.csv", index=False)
    else:
        logging.error("No valid option selected. Use --help for options.")


if __name__ == "__main__":
    cli_main()

from joblib import Parallel, delayed
import logging
import gc
import os
import joblib

from config import MODEL_DIR
from model_trainer import train_and_save
from regression_utils import train_regression
from evaluate import explain_with_shap
from combo_logger import log_top_combos
from boruta_prune import prune_combo
from recommender import recommend_trades
from daily_journal import save_daily_journal
from strategy_meta import log_top_shap_features


def process_combo(df, combo, ret_col, regression, prune, journal, show_plot):
    combo_name = "_".join(combo)
    logging.info(f"[🧪 Training] {combo_name} → {ret_col}")

    model = None
    try:
        # ✅ Signal filter
        signal = df[combo[0]].astype(bool)
        for cond in combo[1:]:
            signal &= df[cond].astype(bool)
        df["combo_signal"] = signal.astype(int)

        if signal.sum() < 10:
            logging.warning(f"⚠️ Skipping {combo_name}: not enough valid signals ({signal.sum()})")
            return

        # ✅ Prepare data
        X = df[list(combo)].copy()
        y = df[ret_col] if regression else (df[ret_col] > 0).astype(int)

        os.makedirs(MODEL_DIR, exist_ok=True)

        # ✅ Train model
        if regression:
            model_path, model = train_regression(X, y, combo_name, ret_col)
        else:
            train_and_save(X, y, model_type="XGBoost", task="classification", suffix=f"{combo_name}_{ret_col}")
            model_path = os.path.join(MODEL_DIR, f"XGBoost_classification_{combo_name}_{ret_col}.pkl")

            try:
                model = joblib.load(model_path)
            except Exception as e:
                logging.error(f"❌ Failed to load trained model: {model_path} | {e}")
                return

        # ✅ SHAP Explanation
        shap_df = explain_with_shap(model, X, "XGBoost", suffix=f"{combo_name}_{ret_col}", show=show_plot)

        log_top_shap_features(combo_name, ret_col, shap_df, top_n=5)

        # ✅ Optional pruning
        if prune and prune_combo(shap_df, combo_name=combo_name, target=ret_col):
            logging.info(f"🧹 Pruned weak combo: {combo_name}")
            return

        # ✅ Log top features
        log_top_combos(shap_df, "XGBoost", f"{combo_name}_{ret_col}", top_n=3)

        # ✅ Optional journal
        if journal:
            try:
                df_preds = recommend_trades(model_path, df, list(combo))
                df_preds[ret_col] = df[ret_col]
                save_daily_journal(df_preds, combo_name, ret_col)
            except Exception as e:
                logging.warning(f"⚠️ Journal failed for {combo_name}: {e}")

    except Exception as e:
        logging.error(f"❌ Failed training combo {combo_name}: {e}")

    finally:
        # ✅ Safe memory cleanup
        for var in ['model', 'X', 'y', 'shap_df']:
            if var in locals():
                del locals()[var]
        gc.collect()


def run_parallel_training(df, combos, ret_col, regression=False, prune=False, journal=False, show_plot=False, n_jobs=4):
    Parallel(n_jobs=n_jobs)(
        delayed(process_combo)(df.copy(), combo, ret_col, regression, prune, journal, show_plot)
        for combo in combos
    )


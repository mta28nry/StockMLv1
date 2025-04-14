# mlearning/autotune_self.py

"""
Module: autotune_self.py
Purpose:
- Auto-tunes top-performing combos using XGBoost + SHAP
- Logs improvements, stores tuned models, updates SQLite
"""

import logging
import os
import sqlite3

import joblib
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from config import DATA_PATH, DB_PATH, MODEL_DIR
from evaluate import explain_with_shap
from features import build_features
from labels import build_labels
from strategy_meta import log_meta, log_tuning_result

logging.basicConfig(level=logging.INFO)

# --- Auto-tuning Core ---

def auto_tune_xgb(X, y, cv=3, n_iter=20):
    """Performs hyperparameter tuning on XGBoost model using RandomizedSearchCV."""
    try:
        model = XGBClassifier(eval_metric='logloss', use_label_encoder=False, verbosity=0)
        param_grid = {
            "n_estimators": [50, 100, 150],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
        }

        search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=n_iter,
                                    cv=cv, scoring='accuracy', verbose=0, n_jobs=-1)
        search.fit(X, y)

        logging.info(f"✅ Best tuned params: {search.best_params_}")
        return search.best_estimator_, search.best_params_
    except Exception as e:
        logging.error(f"❌ XGBoost tuning failed: {e}")
        return None, None

# --- Fetch Best Combos from meta ---

def fetch_top_combos(min_acc=0.6, min_shap=0.01, max_attempts=2):
    """Returns combos with high accuracy & SHAP not tuned more than max_attempts."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT combo, target, MAX(accuracy) as best_acc
        FROM meta
        WHERE accuracy >= ? AND shap_quality >= ?
        GROUP BY combo, target
    """, (min_acc, min_shap))
    top_combos = cursor.fetchall()

    filtered = []
    for combo, target, _ in top_combos:
        cursor.execute("SELECT COUNT(*) FROM meta WHERE combo=? AND target=?", (combo, target))
        attempts = cursor.fetchone()[0]
        if attempts < max_attempts:
            filtered.append((combo, target))
    conn.close()
    return filtered

# --- Main Tuning Pipeline ---

def autotune():
    logging.info("🚀 Starting autotune_self.py...")

    df = pd.read_csv(DATA_PATH)
    df = build_features(df)
    df = build_labels(df)
    df = df.dropna()

    combos = fetch_top_combos()
    logging.info(f"🔍 Found {len(combos)} combos eligible for tuning.")

    for combo, target in combos:
        feature_list = combo.split("_")
        if not all(f in df.columns for f in feature_list):
            logging.warning(f"❌ Skipping {combo} (missing features)")
            continue

        X = df[feature_list]
        y = (df[target] > 0).astype(int)

        # Load baseline accuracy & SHAP
        baseline_score, baseline_shap = 0, 0
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT MAX(accuracy), MAX(shap_quality)
                FROM meta WHERE combo=? AND target=? AND kept=1
            """, (combo, target))
            result = cursor.fetchone()
            if result:
                baseline_score = result[0] or 0
                baseline_shap = result[1] or 0
            conn.close()
        except Exception as e:
            logging.warning(f"⚠️ Failed to load baseline for {combo}: {e}")

        try:
            best_model, best_params = auto_tune_xgb(X, y)
            if best_model is None:
                continue

            score = best_model.score(X, y)
            shap_df = explain_with_shap(best_model, X, "XGBoost", suffix=f"TUNED_{combo}_{target}", show=False)
            shap_score = shap_df["importance"].mean()

            # Log tuning result
            log_tuning_result(combo, target, baseline_score, score, baseline_shap, shap_score, best_params)

            if score > baseline_score + 0.01 and shap_score > baseline_shap + 0.005:
                os.makedirs(MODEL_DIR, exist_ok=True)
                model_path = os.path.join(MODEL_DIR, f"TUNED_XGB_{combo}_{target}.pkl")
                joblib.dump(best_model, model_path, compress=3)

                log_meta(combo, target, score, shap_score, kept=True)
                logging.info(f"✅ Tuning improved {combo} | {target} | score {baseline_score:.3f} → {score:.3f}")
            else:
                log_meta(combo, target, score, shap_score, kept=False)
                logging.info(f"⚠️ Tuning did not significantly improve {combo} | score {score:.3f}")
        except Exception as e:
            logging.error(f"❌ Tuning failed for {combo}: {e}")

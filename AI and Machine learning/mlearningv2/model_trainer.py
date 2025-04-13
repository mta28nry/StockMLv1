"""
model_trainer.py — Unified ML training module (solo + combo)
"""

import os
import hashlib
import joblib
import logging
import sqlite3
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

from config import MODEL_DIR, DB_PATH
from evaluate import explain_with_shap
from strategy_meta import log_meta
from log_config import get_logger

# ---------------------------
# Logging
# ---------------------------
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "Logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger = get_logger("model_trainer")

# ---------------------------
# Environment
# ---------------------------
def is_gpu_available():
    try:
        import pynvml
        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetCount() > 0
    except Exception:
        return False

USE_GPU = is_gpu_available()
THREADS = os.cpu_count() or 4
logger.info(f"🖥️ GPU Available: {USE_GPU} | Threads: {THREADS}")

# ---------------------------
# Model Factory
# ---------------------------
def get_model(name, task):
    def safe_lightgbm(device_type):
        try:
            logger.info(f"🔧 Initializing LightGBM ({device_type})")
            return {
                "classification": LGBMClassifier(device=device_type, gpu_use_dp=True, n_jobs=THREADS),
                "regression": LGBMRegressor(device=device_type, gpu_use_dp=True, n_jobs=THREADS)
            }[task]
        except Exception as e:
            logger.warning(f"⚠️ LightGBM GPU failed: {e} — Using CPU")
            return {
                "classification": LGBMClassifier(device="cpu", n_jobs=THREADS),
                "regression": LGBMRegressor(device="cpu", n_jobs=THREADS)
            }[task]

    if task == "classification":
        return {
            "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=THREADS),
            "LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=THREADS),
            "XGBoost": XGBClassifier(
                eval_metric='logloss', use_label_encoder=False,
                tree_method="gpu_hist" if USE_GPU else "hist",
                predictor="gpu_predictor" if USE_GPU else "auto", n_jobs=THREADS
            ),
            "LightGBM": safe_lightgbm("gpu" if USE_GPU else "cpu")
        }[name]

    if task == "regression":
        return {
            "RandomForest": RandomForestRegressor(n_estimators=100, n_jobs=THREADS),
            "XGBoost": XGBRegressor(
                tree_method="gpu_hist" if USE_GPU else "hist",
                predictor="gpu_predictor" if USE_GPU else "auto", n_jobs=THREADS
            ),
            "LightGBM": safe_lightgbm("gpu" if USE_GPU else "cpu")
        }[name]

    raise ValueError(f"❌ Unsupported model: {name} ({task})")

# ---------------------------
# Helpers
# ---------------------------
def hash_features(feature_list):
    return hashlib.md5("_".join(sorted(feature_list)).encode()).hexdigest()

def already_trained(combo, target, model_type, feature_hash, min_acc=0.01):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.execute("""
                SELECT accuracy, feature_hash FROM meta
                WHERE combo=? AND target=? AND model_type=? AND kept=1
                ORDER BY date_trained DESC LIMIT 1
            """, (combo, target, model_type))
            row = cur.fetchone()
            return row and row[0] >= min_acc and row[1] == feature_hash
    except Exception as e:
        logger.warning(f"⚠️ Failed meta lookup: {e}")
        return False

def log_indicator_score(feature, model, target, acc, shap_avg, feature_hash):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS indicator_scores (
                    feature TEXT, model TEXT, target TEXT,
                    accuracy REAL, shap_score REAL,
                    feature_hash TEXT, date_trained TEXT
                )
            """)
            conn.execute("""
                INSERT INTO indicator_scores VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (feature, model, target, acc, shap_avg, feature_hash, datetime.now().isoformat()))
    except Exception as e:
        logger.warning(f"⚠️ Failed indicator logging: {feature}, {model}: {e}")

def log_shap_to_db(feature, model, target, shap_df):
    try:
        if shap_df.empty:
            return
        shap_df["combo"] = feature
        shap_df["target"] = target
        shap_df["model_type"] = model
        shap_df["rank"] = shap_df["importance"].rank(ascending=False)
        with sqlite3.connect(DB_PATH) as conn:
            shap_df.to_sql("shap_features", conn, if_exists="append", index=False)
    except Exception as e:
        logger.warning(f"⚠️ Failed SHAP logging for {feature}/{model}: {e}")

# ---------------------------
# Core Training Logic
# ---------------------------
def train_and_save(X, y, model_type="XGBoost", task="classification", suffix="model",
                   feature_list=None, fit_params=None, early_stopping_rounds=20, eval_metric=None):

    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        if feature_list:
            X = X[feature_list]

        feature_hash = hash_features(feature_list or X.columns.tolist())
        target = suffix.split("_")[-1]
        combo_name = suffix.replace(f"_{target}", "")

        if already_trained(combo_name, target, model_type, feature_hash):
            logger.info(f"⏭️ {model_type} {combo_name} skipped (cached)")
            return None

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2,
            stratify=y if task == "classification" else None,
            random_state=42
        )

        model = get_model(model_type, task)
        fit_args = {"eval_set": [(X_val, y_val)], "verbose": False}
        if "XGBoost" in model_type or "LightGBM" in model_type:
            fit_args["early_stopping_rounds"] = early_stopping_rounds
            if eval_metric: fit_args["eval_metric"] = eval_metric
        if fit_params: fit_args.update(fit_params)

        model.fit(X_train, y_train, **fit_args)
        score_val = model.score(X_val, y_val)

        logger.info(f"📊 {model_type} | {combo_name} | Val: {score_val:.4f}")
        model_path = os.path.join(MODEL_DIR, f"{model_type}_{task}_{suffix}.pkl")
        joblib.dump(model, model_path, compress=3)

        shap_df = explain_with_shap(model, X_val, model_type, suffix=suffix, show=False)
        shap_score = shap_df["importance"].mean() if not shap_df.empty else 0

        log_meta(combo=combo_name, target=target, accuracy=score_val,
                 shap_quality=shap_score, kept=True,
                 model_type=model_type, feature_hash=feature_hash)

        if feature_list and len(feature_list) == 1:
            log_indicator_score(feature_list[0], model_type, target, score_val, shap_score, feature_hash)
            log_shap_to_db(feature_list[0], model_type, target, shap_df)

        preds = pd.DataFrame(index=X_val.index)
        preds["prediction"] = model.predict(X_val)
        if hasattr(model, "predict_proba"):
            preds["confidence"] = model.predict_proba(X_val)[:, 1]
        preds["predicted_signal"] = (preds["prediction"] == 1).astype(int)
        return preds

    except Exception as e:
        logger.error(f"❌ {model_type} failed: {suffix} — {e}")
        return None

# ---------------------------
# Solo Feature Evaluator
# ---------------------------
def train_each_indicator_all_models_full(
    df,
    target_col="return_15m",
    task="regression",
    models=("RandomForest", "XGBoost", "LightGBM", "LogisticRegression"),
    export_csv=True,
    csv_path="Logs/solo_indicator_scores.csv"
):
    logger.info("🚀 SOLO FEATURE TRAINING: All Models")

    exclude = ["Open", "High", "Low", "Close", "Volume", target_col]
    features = [f for f in df.columns if f not in exclude and df[f].dtype in [np.float64, np.int64]]

    results, shap_logs = [], {}

    for feature in features:
        for model_type in models:
            if model_type == "LogisticRegression" and task != "classification":
                continue

            suffix = f"{model_type}_{feature}_{target_col}"
            preds = train_and_save(
                X=df[[feature]].copy(),
                y=df[target_col],
                model_type=model_type,
                task=task,
                suffix=suffix,
                feature_list=[feature]
            )
            if preds is not None:
                acc = preds["predicted_signal"].mean() if "predicted_signal" in preds else None
                shap_df = explain_with_shap(get_model(model_type, task), df[[feature]], model_type, suffix=suffix, show=False)
                shap_score = shap_df["importance"].mean() if not shap_df.empty else 0
                results.append({
                    "feature": feature, "model": model_type,
                    "target": target_col, "accuracy": acc, "shap_score": shap_score
                })
                shap_logs.setdefault(model_type, []).append(shap_df)

    df_result = pd.DataFrame(results)
    if export_csv:
        df_result.to_csv(csv_path, index=False)
        logger.info(f"✅ Solo model results → {csv_path}")

    save_heatmaps(df_result, shap_logs)
    export_top_combos(df_result)
    return df_result

# ---------------------------
# Visual Outputs
# ---------------------------
def save_heatmaps(results_df, shap_logs):
    try:
        acc_map = results_df.pivot(index="feature", columns="model", values="accuracy")
        plt.figure(figsize=(14, 8))
        sns.heatmap(acc_map, annot=True, fmt=".3f", cmap="viridis")
        plt.title("Solo Indicator Accuracy")
        plt.tight_layout()
        plt.savefig("Logs/solo_accuracy_heatmap.png")
        plt.close()
    except Exception as e:
        logger.warning(f"⚠️ Failed accuracy heatmap: {e}")

    for model_type, shap_list in shap_logs.items():
        try:
            df = pd.concat(shap_list)
            top_df = df.groupby("feature")["importance"].mean().sort_values(ascending=False).head(25)
            top_df.plot(kind="barh", title=f"SHAP: {model_type}", figsize=(10, 6))
            plt.tight_layout()
            plt.savefig(f"Logs/shap_summary_{model_type}.png")
            plt.close()
        except Exception as e:
            logger.warning(f"⚠️ Failed SHAP summary: {e}")

def export_top_combos(df, top_n=20, combo_size=3):
    try:
        top_feats = df.sort_values("accuracy", ascending=False)["feature"].unique()[:top_n]
        combos = { "_".join(c): list(c) for c in combinations(top_feats, combo_size) }
        with open("autogen_combos_from_solo.yaml", "w") as f:
            yaml.dump(combos, f)
        logger.info("✅ Exported solo combos to autogen_combos_from_solo.yaml")
    except Exception as e:
        logger.warning(f"⚠️ Combo export failed: {e}")

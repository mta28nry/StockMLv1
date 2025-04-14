"""
model_trainer.py — ML Training Engine (SHAP Ready)
Purpose:
- Trains, saves, and evaluates models (XGBoost, LightGBM, RF, LR)
- Logs model performance
- SHAP evaluation now deferred to post-phase
"""

import hashlib
import os
import sqlite3
from datetime import datetime
from itertools import combinations

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from config import DB_PATH, MODEL_DIR, create_logger, error_logger
from strategy_meta import log_meta

# ---------------------------
# 🔧 Logger
# ---------------------------
model_trainer = create_logger("Model_Trainer", log_to_file=True)


# ---------------------------
# ⚙️ Model Factory
# ---------------------------
def get_model(name, task, use_gpu=True, threads=os.cpu_count() or 8):
    if name == "LightGBM":
        device = "gpu" if use_gpu else "cpu"
        try:
            return LGBMClassifier(
                device=device, gpu_use_dp=True, n_jobs=threads, verbosity=-1
                ) if task == "classification" else \
                LGBMRegressor(device=device, gpu_use_dp=True, n_jobs=threads, verbosity=-1)
        except:
            return LGBMClassifier(n_jobs=threads, verbosity=-1) if task == "classification" else \
                LGBMRegressor(n_jobs=threads, verbosity=-1)
    elif name == "XGBoost":
        return XGBClassifier(
            use_label_encoder=False,
            tree_method="gpu_hist" if use_gpu else "hist",
            predictor="gpu_predictor" if use_gpu else "auto",
            n_jobs=threads,
            verbosity=0
            ) if task == "classification" else XGBRegressor(
            tree_method="gpu_hist" if use_gpu else "hist",
            predictor="gpu_predictor" if use_gpu else "auto",
            n_jobs=threads,
            verbosity=0
            )
    elif name == "RandomForest":
        return RandomForestClassifier(n_estimators=100, n_jobs=threads) if task == "classification" else \
            RandomForestRegressor(n_estimators=100, n_jobs=threads)
    elif name == "LogisticRegression":
        return LogisticRegression(max_iter=1000, n_jobs=threads)
    raise ValueError(f"❌ Unsupported model: {name} ({task})")


# ---------------------------
# 🔑 Utility Functions
# ---------------------------
def hash_features(features):
    return hashlib.md5("_".join(sorted(features)).encode()).hexdigest()


def already_trained(combo, target, model_type, feature_hash):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute(
                """
                SELECT accuracy, feature_hash FROM meta
                WHERE combo=? AND target=? AND model_type=? AND kept=1
                ORDER BY date_trained DESC LIMIT 1
            """, (combo, target, model_type)
                ).fetchone()
            return row and row[1] == feature_hash
    except Exception as e:
        model_trainer.warning(f"⚠️ Already-trained check failed: {e}")
        return False


def safe_model_save(model, model_type, suffix):
    folder = os.path.join(MODEL_DIR, model_type)
    os.makedirs(folder, exist_ok=True)

    clean = suffix.replace("return_", "").replace("_", "-")
    path = os.path.join(folder, f"{model_type}_{clean}.pkl")  # Ensure same format for saving/loading
    joblib.dump(model, path, compress=3)
    model_trainer.info(f"✅ Model saved: {path}")



def log_indicator_score(feature, model, target, acc, feature_hash, symbol="UNKNOWN"):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS indicator_scores (
                    feature TEXT, model TEXT, target TEXT,
                    accuracy REAL, shap_score REAL,
                    feature_hash TEXT, date_trained TEXT,
                    symbol TEXT
                )
            """
                )
            conn.execute(
                """
                INSERT INTO indicator_scores (feature, model, target, accuracy, shap_score, feature_hash, date_trained, symbol)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (feature, model, target, acc, feature_hash, datetime.now().isoformat(), symbol)
                )
    except Exception as e:
        error_logger.error(f"[log_indicator_score] ❌ Failed: {e}")


# ---------------------------
# 🧠 Core Training Function
# ---------------------------

import shap
def train_and_save(
    X, y, model_type="XGBoost", task="classification", suffix="model", symbol="UNKNOWN",
    feature_list=None, fit_params=None, eval_metric=None, use_gpu=False, threads=os.cpu_count()
):
    try:
        if feature_list:
            X = X[feature_list]

        feature_hash = hash_features(feature_list or X.columns.tolist())
        target = suffix.split("_")[-1]
        combo = suffix.replace(f"_{target}", "")

        if already_trained(combo, target, model_type, feature_hash):
            model_trainer.info(f"⏭️ Skipping {model_type} {combo} (already trained)")
            return None

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2,
            stratify=y if task == "classification" else None,
            random_state=42
        )

        model = get_model(model_type, task, use_gpu=use_gpu, threads=threads)

        fit_args = {}
        accepted = model.fit.__code__.co_varnames
        if "eval_set" in accepted:
            fit_args["eval_set"] = [(X_val, y_val)]
        if eval_metric and "eval_metric" in accepted:
            fit_args["eval_metric"] = eval_metric
        if "early_stopping_rounds" in accepted:
            fit_args["early_stopping_rounds"] = 20
        if "verbose" in accepted:
            fit_args["verbose"] = False
        if fit_params:
            for k, v in fit_params.items():
                if k in accepted:
                    fit_args[k] = v

        model.fit(X_train, y_train, **fit_args)
        score = model.score(X_val, y_val)
        model_trainer.info(f"📊 {model_type} | {combo} | Score: {score:.4f}")
        safe_model_save(model, model_type, suffix)

        # SHAP calculation
        try:
            explainer = shap.Explainer(model, X_val)
            shap_values = explainer(X_val)
            shap_mean = abs(shap_values.values).mean(axis=0)
            shap_score = shap_mean.mean()
        except Exception as shap_err:
            model_trainer.warning(f"⚠️ SHAP failed for {combo}: {shap_err}")
            shap_score = 0.0

        # Log meta
        try:
            log_meta(
                combo=combo,
                target=target,
                accuracy=score,
                shap_score=shap_score,
                kept=True,
                model_type=model_type,
                feature_hash=feature_hash,
                symbol=symbol,
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            model_trainer.error(f"❌ log_meta failed for {combo}: {e}")

        return model

    except Exception as e:
        model_trainer.error(f"❌ {model_type} | {suffix} -> {e}")
        return None

def train_each_indicator_all_models_full(
    df: pd.DataFrame,
    config: dict,
    symbol: str = "UNKNOWN",
    horizons=None,
    export_csv=True,
    csv_path="Logs/solo_indicator_scores_all_tf.csv"
) -> pd.DataFrame:
    """
    Trains each indicator model using all classifiers/regressors.
    Logs performance + shap and saves optional CSV.
    """
    model_trainer.info(f"🚀 Solo Model Training Started for: {symbol}")

    results = []
    task_type = config.get("task", "regression")
    model_types = config["models"]["classifiers"] if task_type == "classification" else config["models"]["regressors"]
    horizons = horizons or config.get("features", {}).get("targets", config["RETURN_COLUMNS"])

    exclude_cols = ["Open", "High", "Low", "Close", "Volume"]
    features = [
        col for col in df.columns
        if col not in exclude_cols and not col.startswith("return_") and df[col].dtype in [float, int]
    ]

    for target in horizons:
        if target not in df.columns:
            model_trainer.warning(f"⚠️ Target missing in data: {target}")
            continue

        for feature in features:
            for model_type in model_types:
                if model_type == "LogisticRegression" and task_type != "classification":
                    continue

                suffix = f"{model_type}_{feature}_{target}"
                model_trainer.info(f"🧪 Training {model_type} on {feature} → {target}")

                model = train_and_save(
                    X=df[[feature]],
                    y=df[target],
                    model_type=model_type,
                    task=task_type,
                    suffix=suffix,
                    feature_list=[feature],
                    use_gpu=config["training"].get("use_gpu", False),
                    threads=os.cpu_count(),
                    symbol=symbol
                )

                if model:
                    try:
                        acc = model.score(df[[feature]], df[target])
                        results.append({
                            "feature": feature,
                            "model": model_type,
                            "target": target,
                            "accuracy": acc,
                            "symbol": symbol
                        })
                    except Exception as e:
                        model_trainer.error(f"⚠️ Scoring failed for {suffix}: {e}")

    result_df = pd.DataFrame(results)
    if export_csv and not result_df.empty:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        result_df.to_csv(csv_path, index=False)
        model_trainer.info(f"📤 Scores saved to: {csv_path}")

    return result_df




# ---------------------------
# 🔬 SHAP Summary Heatmaps
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
        model_trainer.error(f"⚠️ Accuracy heatmap failed: {e}")

    for model_type, shap_list in shap_logs.items():
        try:
            df = pd.concat(shap_list)
            top_df = df.groupby("feature")["importance"].mean().sort_values(ascending=False).head(25)
            top_df.plot(kind="barh", title=f"SHAP: {model_type}", figsize=(10, 6))
            plt.tight_layout()
            plt.savefig(f"Logs/shap_summary_{model_type}.png")
            plt.close()
        except Exception as e:
            model_trainer.error(f"⚠️ SHAP summary failed: {e}")


def log_shap_to_db(feature: str, model: str, target: str, shap_df: pd.DataFrame, symbol="UNKNOWN"):
    """
    Store SHAP values into the SQLite database for later querying and meta-analysis.
    """
    try:
        if shap_df.empty: return
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS shap_features (
                    feature TEXT, importance REAL, rank REAL,
                    combo TEXT, target TEXT, model_type TEXT, symbol TEXT
                )
            """
                )
            shap_df["combo"] = feature
            shap_df["target"] = target
            shap_df["model_type"] = model
            shap_df["rank"] = shap_df["importance"].rank(ascending=False)
            shap_df["feature"] = shap_df.index
            shap_df["symbol"] = symbol
            shap_df.to_sql("shap_features", conn, if_exists="append", index=False)
    except Exception as e:
        error_logger.error(f"[log_shap_to_db] ❌ Failed: {e}")


def evaluate_shap_parallel(model_data_list):
    """
    Parallel SHAP explainability using Dask for multiple models.
    model_data_list = List of (model, X_val, model_type, suffix)
    """
    from dask.distributed import Client, LocalCluster

    client = Client(LocalCluster(n_workers=os.cpu_count(), threads_per_worker=1))
    model_trainer.info(f"⚡ SHAP parallel cluster started with {os.cpu_count()} workers")

    futures = []
    for model, X_val, model_type, suffix in model_data_list:
        futures.append(client.submit(explain_with_shap, model, X_val, model_type, suffix, show=False))

    results = []
    for fut in futures:
        try:
            shap_df = fut.result()
            results.append(shap_df)
        except Exception as e:
            model_trainer.error(f"❌ SHAP future failed: {e}")
    client.close()
    return results


# ---------------------------
# 🔁 Combo Export
# ---------------------------
def export_top_combos(results_df, top_n=20, combo_size=3):
    try:
        top_feats = results_df.sort_values("accuracy", ascending=False)["feature"].unique()[:top_n]
        combos = {"_".join(c): list(c) for c in combinations(top_feats, combo_size)}
        with open("autogen_combos_from_solo.yaml", "w") as f:
            yaml.dump(combos, f)
        model_trainer.info("✅ Exported top combos to autogen_combos_from_solo.yaml")
    except Exception as e:
        model_trainer.error(f"⚠️ Combo export failed: {e}")

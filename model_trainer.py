# mlearning/model_trainer.py

"""
Trains and evaluates classification or regression models,
saves models, logs performance, tracks SHAP, and adds prediction metadata.
"""

import os
import joblib
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRegressor
from config import MODEL_DIR
from evaluate import explain_with_shap
from strategy_meta import log_meta

logging.basicConfig(level=logging.INFO)

def get_model(name, task):
    if task == "classification":
        return {
            "RandomForest": RandomForestClassifier(n_estimators=100),
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False, verbosity=0)
        }[name]
    elif task == "regression":
        return {
            "RandomForest": RandomForestRegressor(n_estimators=100),
            "XGBoost": XGBRegressor()
        }[name]
    raise ValueError(f"Unsupported model: {name}")

def train_and_save(X, y, model_type="XGBoost", task="classification", suffix="model", feature_list=None):
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        if feature_list:
            X = X[feature_list]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2,
            stratify=y if task == "classification" else None,
            random_state=42
        )

        model = get_model(model_type, task)
        model.fit(X_train, y_train)

        score_train = model.score(X_train, y_train)
        score_val = model.score(X_val, y_val)

        logging.info(f"📊 {model_type} {task} | Train: {score_train:.4f} | Val: {score_val:.4f}")

        path = os.path.join(MODEL_DIR, f"{model_type}_{task}_{suffix}.pkl")
        joblib.dump(model, path, compress=3)
        logging.info(f"✅ Saved model: {path}")

        shap_df = explain_with_shap(model, X_val, model_type, suffix=suffix, show=False)
        shap_score = shap_df["importance"].mean() if not shap_df.empty else 0

        log_meta(combo=suffix, target=suffix.split("_")[-1], accuracy=score_val, shap_quality=shap_score, kept=True)

        # Attach prediction metadata for downstream journal scoring
        preds = pd.DataFrame(index=X_val.index)
        preds["prediction"] = model.predict(X_val)
        if hasattr(model, "predict_proba"):
            preds["confidence"] = model.predict_proba(X_val)[:, 1]
        preds["predicted_signal"] = (preds["prediction"] == 1).astype(int)
        return preds

    except Exception as e:
        logging.error(f"❌ Failed model training: {model_type}_{suffix}: {e}")
        return None

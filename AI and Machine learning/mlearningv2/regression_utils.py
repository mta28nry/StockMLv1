# mlearning/regression_utils.py
import logging
import os

import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from config import MODEL_DIR


def train_regression(X, y, combo_name, target):
    model = XGBRegressor()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    logging.info(f"[REGRESSION] R^2 for {combo_name} on {target}: {score:.4f}")

    model_path = os.path.join(MODEL_DIR, f"XGBRegressor_{combo_name}_{target}.pkl")
    joblib.dump(model, model_path)
    return model_path, model

# mlearning/pipeline.py
import pandas as pd

from config import *
from evaluate import explain_with_shap, plot_heatmap
from features import build_features
from labels import build_labels
from model_trainer import train_and_save

logging.basicConfig(level=logging.INFO)


def run_pipeline():
    logging.info("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df = build_features(df)
    df = build_labels(df)

    features = [c for c in df.columns if df[c].dtype in [float, bool] and not c.startswith("return")]

    for ret_col in RETURN_COLUMNS:
        df = df.dropna(subset=[ret_col])
        X = df[features]
        y = (df[ret_col] > 0).astype(int)  # binary target: positive return = success

        for clf in CLASSIFIERS:
            train_and_save(X, y, clf, task="classification", suffix=ret_col)

model_path = os.path.join(MODEL_DIR, f"{clf}_classification_{ret_col}.pkl")
model = joblib.load(model_path)
importance_df = explain_with_shap(model, X, clf, suffix=ret_col)
plot_heatmap(importance_df, clf, suffix=ret_col)

if __name__ == "__main__":
    run_pipeline()

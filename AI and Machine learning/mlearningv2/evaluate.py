# mlearning/evaluate.py

"""
Purpose:
- SHAP explainability + fallback for trained models
- Generates SHAP bar/summary plots and heatmaps
"""

import os
import shap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.inspection import permutation_importance
from config import OUTPUT_DIR
from log_config import get_logger
logger = get_logger("model_trainer")

# --- Logging Config ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

# --- Ensure output dir exists ---
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# SHAP + Fallback Importance
# =============================================================================
def explain_with_shap(model, X, model_name: str, suffix: str = "", show: bool = False) -> pd.DataFrame:
    """
    Run SHAP explainability and fallback to permutation importance if SHAP fails.

    Returns:
        pd.DataFrame: feature importance ranked by mean absolute SHAP or permutation scores.
    """
    logging.info(f"🔍 Running SHAP for {model_name} ({suffix})...")
    try:
        X_sample = X.sample(min(len(X), 500), random_state=42)

        explainer = shap.Explainer(model.predict, X_sample)
        shap_values = explainer(X_sample)

        # --- Plot SHAP bar ---
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=show)
        if not show:
            bar_path = os.path.join(OUTPUT_DIR, f"{model_name}_{suffix}_shap_bar.png")
            plt.title(f"{model_name} Feature Importance")
            plt.savefig(bar_path)
            plt.clf()

        # --- Plot SHAP summary ---
        shap.summary_plot(shap_values, X_sample, show=show)
        if not show:
            summary_path = os.path.join(OUTPUT_DIR, f"{model_name}_{suffix}_shap_summary.png")
            plt.title(f"{model_name} SHAP Summary")
            plt.savefig(summary_path)
            plt.clf()

        # --- Extract Importance ---
        importance = shap_values.abs.mean(0).values if hasattr(shap_values, 'abs') else shap_values.values.mean(0)
        df = pd.DataFrame({
            "feature": X_sample.columns,
            "importance": importance
        }).sort_values("importance", ascending=False)
        df["rank"] = range(1, len(df) + 1)

        logging.info(f"✅ SHAP explainability complete. Top feature: {df.iloc[0]['feature']}")
        return df

    except Exception as e:
        logging.warning(f"⚠️ SHAP failed: {e}. Falling back to permutation importance.")
        try:
            result = permutation_importance(model, X, model.predict(X), n_repeats=10, random_state=42)
            df = pd.DataFrame({
                "feature": X.columns,
                "importance": result.importances_mean
            }).sort_values("importance", ascending=False)
            df["rank"] = range(1, len(df) + 1)

            logging.info("✅ Fallback importance computed.")
            return df

        except Exception as e2:
            logging.error(f"❌ Fallback also failed: {e2}")
            return pd.DataFrame(columns=["feature", "importance", "rank"])


# =============================================================================
# Heatmap
# =============================================================================
def plot_heatmap(df_importance: pd.DataFrame, model_name: str, suffix: str = ""):
    """
    Generate heatmap of feature importance from SHAP or fallback results.

    Args:
        df_importance (pd.DataFrame): DataFrame with columns [feature, importance]
    """
    logging.info("📌 Generating strategy combo heatmap...")
    try:
        df_ = df_importance.copy().set_index("feature").T
        plt.figure(figsize=(12, 1.5 + 0.25 * len(df_)))
        sns.heatmap(df_, cmap="coolwarm", annot=True, fmt=".3f")
        plt.title(f"Top Strategy Combination Importance - {model_name} ({suffix})")
        out_path = os.path.join(OUTPUT_DIR, f"{model_name}_{suffix}_heatmap.png")
        plt.savefig(out_path)
        plt.clf()
        logging.info(f"✅ Heatmap saved: {out_path}")
    except Exception as e:
        logging.error(f"❌ Heatmap generation failed: {e}")

"""
boruta_prune.py
------------------
Optional pruning of combos based on SHAP score quality.

Features:
- Removes combos with low average SHAP score
- Logs all pruning actions
"""

import logging

def prune_combo(shap_df, combo_name, target, shap_thresh=0.005):
    if shap_df is None or shap_df.empty:
        logging.warning(f"⚠️ Cannot prune {combo_name}: SHAP data missing")
        return False

    mean_shap = shap_df["importance"].mean()
    if mean_shap < shap_thresh:
        logging.warning(f"⚠️ All features in {combo_name} for {target} are weak. Pruning.")
        return True

    return False
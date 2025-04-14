# mlearning/combo_autogen.py

"""
combo_autogen.py

Purpose:
- Generate new strategy combos using SHAP + Meta DB intelligence
- Combine top features from best-performing models
- Log to autogen_combos.yaml for future training cycles
"""

import os
import sqlite3
from collections import defaultdict

import yaml

from config import DB_PATH, create_logger

combo_autogen = create_logger("combo_autogen", log_to_file=True)



# --- Constants ---
OUTFILE = "autogen_combos.yaml"
MAX_COMBO_SIZE = 4
TOP_SHAP_LIMIT = 5

# --- combo_autogen Config ---
#combo_autogen.basicConfig(level=combo_autogen.INFO, format='%(asctime)s | %(levelname)s | %(message)s')


# =============================================================================
# SHAP-Based Combo Extraction
# =============================================================================
def fetch_shap_combos(min_shap: float = 0.15) -> dict:
    """
    Query SHAP features from the database and group top-ranked features
    into auto-generated combos per target.
    """
    combo_autogen.info("🔍 Scanning SHAP features for auto-combo generation...")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT combo, target, feature
            FROM shap_features
            WHERE importance >= ?
            ORDER BY target, combo, rank ASC
        """, (min_shap,))
        rows = cursor.fetchall()
        conn.close()

        buckets = defaultdict(set)
        for combo, target, feat in rows:
            key = f"{target}"
            buckets[key].add(feat)

        combos = {}
        for target, feats in buckets.items():
            feats = list(feats)[:TOP_SHAP_LIMIT]
            if len(feats) >= 2:
                name = f"shap_{'_'.join(sorted(feats))}_{target}"
                combos[name] = feats

        combo_autogen.info(f"✅ Generated {len(combos)} SHAP combos")
        return combos

    except Exception as e:
        combo_autogen.exception("❌ Failed to fetch SHAP combos")
        return {}


# =============================================================================
# Meta Table-Based Combo Extraction
# =============================================================================
def fetch_meta_combos(min_score: float = 0.52, min_shap: float = 0.15, max_size: int = 4) -> dict:
    """
    Query the meta table to extract combos from previously successful models.
    """
    combo_autogen.info("🔍 Scanning meta table combos...")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT combo, target
            FROM meta
            WHERE accuracy >= ? AND shap_quality >= ? AND kept = 1
        """, (min_score, min_shap))

        rows = cursor.fetchall()
        conn.close()

        combos = {}
        for raw_combo, target in rows:
            feats = raw_combo.split("_")
            if 2 <= len(feats) <= max_size:
                key = f"meta_{raw_combo}_{target}"
                combos[key] = feats

        combo_autogen.info(f"✅ Retrieved {len(combos)} meta combos")
        return combos

    except Exception as e:
        combo_autogen.exception("❌ Failed to fetch meta combos")
        return {}


# =============================================================================
# Save Combo Rules to YAML
# =============================================================================
def save_autogen_combos(new_combos: dict):
    """
    Append newly generated combos to YAML file without duplication.
    """
    if not new_combos:
        combo_autogen.warning("⚠️ No new combos to save.")
        return

    existing = {}
    if os.path.exists(OUTFILE):
        try:
            with open(OUTFILE, "r") as f:
                existing = yaml.safe_load(f) or {}
        except Exception as e:
            combo_autogen.warning(f"⚠️ Failed to read existing YAML: {e}")
            existing = {}

    added = 0
    for name, feats in new_combos.items():
        if name not in existing:
            existing[name] = feats
            added += 1

    if added == 0:
        combo_autogen.warning("⚠️ No new unique combos to append.")
    else:
        try:
            with open(OUTFILE, "w") as f:
                yaml.dump(existing, f)
            combo_autogen.info(f"✅ {added} new combos appended → {OUTFILE}")
        except Exception as e:
            combo_autogen.exception(f"❌ Failed to write to {OUTFILE}")


# =============================================================================
# Main Entrypoint
# =============================================================================
def generate_shap_combos():
    """
    Entry function to fetch, merge, and save SHAP + Meta combos.
    """
    combo_autogen.info("🚀 Generating SHAP + Meta-based strategy combos...")
    shap_combos = fetch_shap_combos()
    meta_combos = fetch_meta_combos()
    all_combos = {**shap_combos, **meta_combos}
    save_autogen_combos(all_combos)
    combo_autogen.info("🏁 Combo generation complete.")


# =============================================================================
# Script Execution
# =============================================================================
if __name__ == "__main__":
    generate_shap_combos()

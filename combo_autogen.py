# mlearning/combo_autogen.py

"""
combo_autogen.py

Purpose:
- Generate new strategy combos using SHAP + Meta DB intelligence
- Combines top features from best-performing models
- Logs to autogen_combos.yaml for use in next training cycle
"""

import os
import sqlite3
import yaml
import logging
from collections import defaultdict
from config import DB_PATH

OUTFILE = "autogen_combos.yaml"
MAX_COMBO_SIZE = 4
TOP_SHAP_LIMIT = 5

def fetch_shap_combos(min_shap=0.15, top_n=3):
    logging.info("🔍 Scanning SHAP features for auto-combo generation...")
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

    # Group top shap features per target
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

    return combos


def fetch_meta_combos(min_score=0.52, min_shap=0.15, max_size=4):
    logging.info("🔍 Scanning top meta table combos...")
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

    return combos


def save_autogen_combos(new_combos):
    if not new_combos:
        logging.warning("⚠️ No new combos to save.")
        return

    existing = {}
    if os.path.exists(OUTFILE):
        with open(OUTFILE, "r") as f:
            existing = yaml.safe_load(f) or {}

    # Append only new combos
    added = 0
    for name, feats in new_combos.items():
        if name not in existing:
            existing[name] = feats
            added += 1

    if added == 0:
        logging.warning("⚠️ No new unique combos to append.")
    else:
        with open(OUTFILE, "w") as f:
            yaml.dump(existing, f)
        logging.info(f"✅ {added} new combos appended → {OUTFILE}")


def generate_shap_combos():
    """Main execution."""
    shap_combos = fetch_shap_combos()
    meta_combos = fetch_meta_combos()
    all_combos = {**shap_combos, **meta_combos}
    save_autogen_combos(all_combos)

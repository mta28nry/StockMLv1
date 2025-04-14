# mlearning/combo_mutation.py

"""
combo_mutation.py — Strategy Mutation Engine

Purpose:
- Evolve strategy combos using mutation of top performers.
- Uses SHAP values + feature type diversity (via config.FEATURE_TYPES).
- Outputs evolved combos to YAML for retraining.

Fixes:
- Removes dependency on 'feature_type' DB column.
- Uses FEATURE_TYPES dict to enrich SHAP results in-memory.
"""

import logging
import os
import random
import sqlite3

import yaml

from config import DB_PATH, FEATURE_TYPES

EVOLVED_PATH = "mlearning/evolved_combos.yaml"

# ---------------------------
# 🔍 Fetch Top Performing Combos
# ---------------------------
def fetch_top_combos(limit=10):
    """Fetch high-performing combos from meta table"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT combo FROM meta
        WHERE kept = 1 AND accuracy >= 0.65
        ORDER BY accuracy DESC
        LIMIT ?
    """, (limit,))
    combos = [row[0].split("_") for row in cursor.fetchall()]
    conn.close()
    return combos

# ---------------------------
# 📈 Fetch SHAP Features (Tagged by type)
# ---------------------------
def fetch_top_shap_features(thresh=0.01):
    """
    Fetch SHAP features from DB and tag their type from FEATURE_TYPES.
    Returns: {type: [features]}
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT feature
        FROM shap_features
        WHERE importance >= ?
        ORDER BY importance DESC
    """, (thresh,))
    all_feats = [row[0] for row in cursor.fetchall()]
    conn.close()

    typed = {}
    for feat in all_feats:
        ftype = FEATURE_TYPES.get(feat, "unknown")
        typed.setdefault(ftype, []).append(feat)

    logging.info(f"📦 SHAP feature breakdown:")
    for ftype, feats in typed.items():
        logging.info(f"  - {ftype}: {len(feats)} features")

    return typed

# ---------------------------
# 🧬 Mutation Logic
# ---------------------------
def mutate_combo(combo, feature_pool):
    """
    Randomly mutate a combo:
    - Add (if under 5)
    - Remove (if over 2)
    - Swap (random replace)
    """
    mutation_type = random.choice(["add", "remove", "swap"])
    mutated = list(combo)

    if mutation_type == "add" and len(mutated) < 5:
        options = [f for f in feature_pool if f not in mutated]
        if options:
            new_feat = random.choice(options)
            mutated.append(new_feat)
            logging.debug(f"➕ Added {new_feat} to {combo}")

    elif mutation_type == "remove" and len(mutated) > 2:
        removed = random.choice(mutated)
        mutated.remove(removed)
        logging.debug(f"➖ Removed {removed} from {combo}")

    elif mutation_type == "swap" and feature_pool:
        removed = random.choice(mutated)
        options = [f for f in feature_pool if f not in mutated]
        if options:
            replacement = random.choice(options)
            idx = mutated.index(removed)
            mutated[idx] = replacement
            logging.debug(f"🔁 Swapped {removed} → {replacement} in {combo}")

    return sorted(set(mutated))

# ---------------------------
# 🔁 Main Evolution Pipeline
# ---------------------------
def evolve_combos():
    logging.info("🧬 Starting combo mutation pipeline...")

    base_combos = fetch_top_combos()
    shap_feats = fetch_top_shap_features()

    if not base_combos or not shap_feats:
        logging.warning("⚠️ Not enough SHAP data or combos to evolve.")
        return []

    feature_pool = list(set(sum(shap_feats.values(), [])))
    all_mutants = []

    for combo in base_combos:
        for _ in range(3):  # Generate 3 mutants per base combo
            mutant = mutate_combo(combo, feature_pool)
            if len(mutant) >= 2:
                all_mutants.append(mutant)

    # Deduplicate
    seen = set()
    unique = []
    for combo in all_mutants:
        key = "_".join(sorted(combo))
        if key not in seen:
            seen.add(key)
            unique.append(combo)

    # Save
    out = {f"mutant_{i+1}": combo for i, combo in enumerate(unique)}
    os.makedirs(os.path.dirname(EVOLVED_PATH), exist_ok=True)
    with open(EVOLVED_PATH, "w") as f:
        yaml.dump(out, f)

    logging.info(f"✅ {len(unique)} evolved combos saved → {EVOLVED_PATH}")
    return unique

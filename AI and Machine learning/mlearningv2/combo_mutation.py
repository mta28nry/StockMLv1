# mlearning/combo_mutation.py
"""
Module: combo_mutation.py
Purpose: Evolve strategy combos using top performers and SHAP data
Features:
- Loads high-performing combos
- Mutates them using add/swap/remove logic
- Incorporates feature diversity via SHAP/DB
- Outputs evolved combos to YAML for future training
"""

import sqlite3
import random
import yaml
import os
import logging
from config import FEATURE_TYPES, DB_PATH

EVOLVED_PATH = "mlearning/evolved_combos.yaml"

# --- Mutation Helpers ---

def fetch_top_combos(limit=10):
    """Fetch top performing combos from meta table"""
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

def mutate_combo(combo, feature_pool):
    """Randomly mutate a single combo"""
    mutation_type = random.choice(["add", "remove", "swap"])
    mutated = list(combo)

    if mutation_type == "add" and len(mutated) < 5:
        options = [f for f in feature_pool if f not in mutated]
        if options:
            mutated.append(random.choice(options))

    elif mutation_type == "remove" and len(mutated) > 2:
        mutated.remove(random.choice(mutated))

    elif mutation_type == "swap":
        if mutated and len(mutated) >= 1:
            old = random.choice(mutated)
            swap_from = [f for f in feature_pool if f not in mutated]
            if swap_from:
                mutated[mutated.index(old)] = random.choice(swap_from)

    return sorted(set(mutated))

def fetch_top_shap_features(thresh=0.01):
    """Fetch SHAP-derived features by type"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT feature, feature_type
        FROM shap_features
        WHERE importance >= ?
    """, (thresh,))
    typed = {}
    for feat, ftype in cursor.fetchall():
        typed.setdefault(ftype, []).append(feat)
    conn.close()
    return typed

# --- Evolution Engine ---

def evolve_combos():
    logging.info("🧬 Starting strategy combo mutation...")
    base_combos = fetch_top_combos()
    shap_feats = fetch_top_shap_features()

    if not shap_feats or not base_combos:
        logging.warning("⚠️ Not enough data to evolve strategies.")
        return []

    all_mutants = []
    feature_pool = list(set(sum(shap_feats.values(), [])))

    for combo in base_combos:
        for _ in range(3):  # 3 mutations per combo
            mutant = mutate_combo(combo, feature_pool)
            if len(mutant) >= 2:
                all_mutants.append(mutant)

    # Deduplicate mutants
    seen = set()
    unique = []
    for combo in all_mutants:
        key = "_".join(combo)
        if key not in seen:
            seen.add(key)
            unique.append(combo)

    out = {f"mutant_{i+1}": combo for i, combo in enumerate(unique)}
    os.makedirs(os.path.dirname(EVOLVED_PATH), exist_ok=True)
    with open(EVOLVED_PATH, "w") as f:
        yaml.dump(out, f)

    logging.info(f"✅ {len(unique)} evolved combos saved → {EVOLVED_PATH}")
    return unique

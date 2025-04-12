# mlearning/combo_config.py

import yaml
import os
import logging

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "combo_rules.yaml")

def load_combo_rules():
    """Load combo_rules.yaml and merge with evolved_combos.yaml (if exists)."""
    with open(CONFIG_PATH, "r") as f:
        static_combos = yaml.safe_load(f)

    try:
        with open("evolved_combos.yaml", "r") as f:
            evolved_combos = yaml.safe_load(f)
        static_combos.update(evolved_combos)
        logging.info(f"🧬 Loaded {len(evolved_combos)} evolved combos.")
    except FileNotFoundError:
        logging.warning("⚠️ No evolved_combos.yaml found. Only using static combos.")

    return static_combos

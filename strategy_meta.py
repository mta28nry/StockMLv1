# mlearning/strategy_meta.py

import os
import sqlite3
import logging
import json
from config import DB_PATH


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS meta (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            combo TEXT,
            target TEXT,
            accuracy REAL,
            shap_quality REAL,
            kept INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    logging.info("✅ meta table ensured in DB")

def log_combo_history(combo, target, accuracy, shap, uses=1, pruned=0):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO combo_history (combo, target, avg_accuracy, avg_shap, uses, pruned)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (combo, target, accuracy, shap, uses, pruned))
    conn.commit()
    conn.close()
    logging.info(f"📈 History updated for {combo} → acc={accuracy:.3f}, shap={shap:.3f}")


def log_meta(combo, target, accuracy, shap_quality, kept):
    """Log combo result to the meta table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO meta (combo, target, accuracy, shap_quality, kept)
        VALUES (?, ?, ?, ?, ?)
    """, (combo, target, accuracy, shap_quality, int(kept)))
    conn.commit()
    conn.close()
    logging.info(f"Meta logged: {combo} | {target} | score={accuracy:.4f} | shap={shap_quality:.4f} | kept={kept}")

def init_tuning_table():
    """Create the tuning results table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tuning_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            combo TEXT,
            target TEXT,
            score_before REAL,
            score_after REAL,
            shap_before REAL,
            shap_after REAL,
            best_params TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def log_tuning_result(combo, target, before_score, after_score, before_shap, after_shap, best_params):
    """Log the results of model tuning (before vs after)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO tuning_results (
            combo, target, score_before, score_after, shap_before, shap_after, best_params
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (combo, target, before_score, after_score, before_shap, after_shap, json.dumps(best_params)))
    conn.commit()
    conn.close()
    logging.info(f"📊 Logged tuning: {combo} | score {before_score:.3f}→{after_score:.3f} | shap {before_shap:.4f}→{after_shap:.4f}")

def log_top_shap_features(combo, target, shap_df, top_n=5):
    """Log the top SHAP features into the shap_features table."""
    if shap_df.empty:
        return
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    for rank, row in enumerate(shap_df.head(top_n).itertuples(), start=1):
        cursor.execute("""
            INSERT INTO shap_features (combo, target, feature, importance, rank)
            VALUES (?, ?, ?, ?, ?)
        """, (combo, target, row.feature, float(row.importance), rank))
    conn.commit()
    conn.close()
    logging.info(f"📥 SHAP features logged: {combo} → {target}")

def init_self_learning_tables():
    """Initialize SHAP tracking + combo history tables."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS shap_features (
            id INTEGER PRIMARY KEY,
            combo TEXT,
            target TEXT,
            feature TEXT,
            importance REAL,
            feature_type TEXT,
            rank INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS combo_history (
            id INTEGER PRIMARY KEY,
            combo TEXT,
            target TEXT,
            source TEXT,
            avg_accuracy REAL,
            avg_shap REAL,
            uses INTEGER,
            pruned INTEGER DEFAULT 0,
            promoted INTEGER DEFAULT 0,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )

    """)
    conn.commit()
    conn.close()

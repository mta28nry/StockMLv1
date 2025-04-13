import os
import sqlite3
import logging
import json
from datetime import datetime
from config import DB_PATH
from log_config import get_logger

logger = get_logger("strategy_meta")

# -------------------------------
# ✅ Connection Utility
# -------------------------------
def get_conn():
    return sqlite3.connect(DB_PATH)

# -------------------------------
# ✅ Initialize Meta DB
# -------------------------------
def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS meta (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                combo TEXT,
                target TEXT,
                accuracy REAL,
                shap_quality REAL,
                kept INTEGER,
                model_type TEXT,
                feature_hash TEXT,
                date_trained TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_meta_combo ON meta(combo)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_meta_target ON meta(target)")
        conn.commit()
    logger.info("✅ meta table ensured")

# -------------------------------
# ✅ Log Model Metadata
# -------------------------------
def log_meta(combo, target, accuracy, shap_quality, kept, model_type=None, feature_hash=None):
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO meta (
                combo, target, accuracy, shap_quality,
                kept, model_type, feature_hash, date_trained
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            combo, target, accuracy, shap_quality,
            int(kept), model_type, feature_hash,
            datetime.now().isoformat()
        ))
        conn.commit()
    logger.info(f"📊 Meta logged: {combo} | {target} | acc={accuracy:.4f} | shap={shap_quality:.4f} | kept={kept}")

# -------------------------------
# ✅ Initialize Tuning Table
# -------------------------------
def init_tuning_table():
    with get_conn() as conn:
        conn.execute("""
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
    logger.info("✅ tuning_results table ensured")

# -------------------------------
# ✅ Log Tuning Outcome
# -------------------------------
def log_tuning_result(combo, target, before_score, after_score, before_shap, after_shap, best_params):
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO tuning_results (
                combo, target, score_before, score_after,
                shap_before, shap_after, best_params
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            combo, target,
            before_score, after_score,
            before_shap, after_shap,
            json.dumps(best_params)
        ))
        conn.commit()
    logger.info(f"🔧 Tuning: {combo} | score {before_score:.3f}→{after_score:.3f} | shap {before_shap:.4f}→{after_shap:.4f}")

# -------------------------------
# ✅ SHAP Feature Logger
# -------------------------------
def log_top_shap_features(combo, target, shap_df, top_n=5):
    if shap_df.empty:
        return
    with get_conn() as conn:
        for rank, row in enumerate(shap_df.head(top_n).itertuples(), start=1):
            conn.execute("""
                INSERT INTO shap_features (
                    combo, target, feature, importance, rank, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                combo, target, row.feature,
                float(row.importance), rank,
                datetime.now().isoformat()
            ))
        conn.commit()
    logger.info(f"📥 SHAP features logged: {combo} → {target}")

# -------------------------------
# ✅ Combo History Logger
# -------------------------------
def log_combo_history(combo, target, accuracy, shap, uses=1, pruned=0):
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO combo_history (
                combo, target, avg_accuracy, avg_shap, uses, pruned
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (combo, target, accuracy, shap, uses, pruned))
        conn.commit()
    logger.info(f"📈 History logged: {combo} acc={accuracy:.3f}, shap={shap:.3f}")

# -------------------------------
# ✅ Self-Learning Table Init
# -------------------------------
def init_self_learning_tables():
    with get_conn() as conn:
        conn.execute("""
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

        conn.execute("""
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
    logger.info("✅ self-learning tables ensured")

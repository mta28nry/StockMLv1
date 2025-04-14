import json
import sqlite3
from datetime import datetime

from config import DB_PATH, create_logger, error_logger

strategy_meta = create_logger("strategy_meta", log_to_file=True)


# -------------------------------
# ✅ Connection Utility
# -------------------------------
def get_conn():
    return sqlite3.connect(DB_PATH)


# -------------------------------
# ✅ Meta Table Init
# -------------------------------
def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS meta (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                combo TEXT,
                target TEXT,
                accuracy REAL,
                shap_score REAL,
                kept INTEGER,
                model_type TEXT,
                feature_hash TEXT,
                symbol TEXT,
                timestamp TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_meta_combo ON meta(combo)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_meta_target ON meta(target)")
        conn.commit()
    strategy_meta.info("✅ meta table ensured")


# -------------------------------
# ✅ Log Model Metadata
# -------------------------------
def log_meta(combo, target, accuracy, shap_score, kept, model_type, feature_hash, symbol="UNKNOWN", timestamp=None):
    try:
        timestamp = timestamp or datetime.now().isoformat()
        with get_conn() as conn:
            conn.execute("""
                INSERT INTO meta (combo, target, accuracy, shap_score, kept, model_type, feature_hash, symbol, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (combo, target, accuracy, shap_score, int(kept), model_type, feature_hash, symbol, timestamp))
    except Exception as e:
        error_logger.error(f"❌ log_meta failed for {combo}_{target}: {e}")


# -------------------------------
# ✅ Tuning Table Init
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
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    strategy_meta.info("✅ tuning_results table ensured")


# -------------------------------
# ✅ Log Tuning Results
# -------------------------------
def log_tuning_result(combo, target, before_score, after_score, before_shap, after_shap, best_params):
    try:
        with get_conn() as conn:
            conn.execute("""
                INSERT INTO tuning_results (
                    combo, target, score_before, score_after,
                    shap_before, shap_after, best_params
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                combo, target, before_score, after_score,
                before_shap, after_shap, json.dumps(best_params)
            ))
        strategy_meta.info(f"🔧 Tuning: {combo} | acc {before_score:.3f}→{after_score:.3f} | shap {before_shap:.4f}→{after_shap:.4f}")
    except Exception as e:
        error_logger.error(f"❌ log_tuning_result failed: {e}")


# -------------------------------
# ✅ SHAP Features Table Logger
# -------------------------------
def log_top_shap_features(combo, target, shap_df, top_n=5, symbol="UNKNOWN"):
    if shap_df.empty:
        return
    try:
        with get_conn() as conn:
            for rank, row in enumerate(shap_df.head(top_n).itertuples(), start=1):
                conn.execute("""
                    INSERT INTO shap_features (
                        combo, target, feature, importance, rank, timestamp, symbol
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    combo, target, row.feature,
                    float(row.importance), rank,
                    datetime.now().isoformat(),
                    symbol
                ))
        strategy_meta.info(f"📥 SHAP features logged: {combo} → {target}")
    except Exception as e:
        error_logger.error(f"❌ log_top_shap_features failed: {e}")


# -------------------------------
# ✅ Combo History Logger
# -------------------------------
def log_combo_history(combo, target, accuracy, shap_score, uses=1, pruned=0):
    try:
        with get_conn() as conn:
            conn.execute("""
                INSERT INTO combo_history (
                    combo, target, avg_accuracy, avg_shap, uses, pruned
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (combo, target, accuracy, shap_score, uses, pruned))
        strategy_meta.info(f"📈 History logged: {combo} acc={accuracy:.3f}, shap={shap_score:.3f}")
    except Exception as e:
        error_logger.error(f"❌ log_combo_history failed: {e}")


# -------------------------------
# ✅ Initialize Self-Learning Tables
# -------------------------------
def init_self_learning_tables():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS shap_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                combo TEXT,
                target TEXT,
                feature TEXT,
                importance REAL,
                feature_type TEXT,
                rank INTEGER,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS combo_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                combo TEXT,
                target TEXT,
                source TEXT,
                avg_accuracy REAL,
                avg_shap REAL,
                uses INTEGER,
                pruned INTEGER DEFAULT 0,
                promoted INTEGER DEFAULT 0,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    strategy_meta.info("✅ self-learning tables ensured")

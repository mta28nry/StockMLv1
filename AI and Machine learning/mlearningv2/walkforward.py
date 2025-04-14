# mlearning/walkforward.py

"""
Walkforward Backtester
- Evaluates top combos using rolling walk-forward validation
- Logs out-of-sample scores
"""
import sqlite3

import pandas as pd

from config import DATA_PATH, DB_PATH, create_logger

walkforward = create_logger("walkforward", log_to_file=True )


def run_walkforward_backtest():
    walkforward.info("🧪 Running walk-forward backtesting...")
    df = pd.read_csv(DATA_PATH)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT combo, target FROM meta
        WHERE kept = 1 ORDER BY accuracy DESC LIMIT 10
    """)
    combos = cursor.fetchall()
    conn.close()

    for combo, target in combos:
        features = combo.split("_")
        if not all(f in df.columns for f in features + [target]):
            continue

        df = df.dropna(subset=features + [target])
        scores = []

        chunks = 5
        size = len(df) // chunks
        for i in range(chunks - 1):
            train = df.iloc[i * size:(i + 1) * size]
            test = df.iloc[(i + 1) * size:(i + 2) * size]

            X_train = train[features]
            y_train = (train[target] > 0).astype(int)
            X_test = test[features]
            y_test = (test[target] > 0).astype(int)

            from xgboost import XGBClassifier
            model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
            model.fit(X_train, y_train)
            acc = model.score(X_test, y_test)
            scores.append(acc)

        avg = sum(scores) / len(scores)
        walkforward.info(f"🔁 Walkforward: {combo}_{target} → avg score={avg:.4f}")

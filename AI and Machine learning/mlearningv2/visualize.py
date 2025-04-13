# mlearning/visualize.py

"""
Visualization Generator
- SHAP trends
- Accuracy growth per combo
"""

import sqlite3, matplotlib.pyplot as plt
import pandas as pd
from config import DB_PATH, OUTPUT_DIR
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # supports most emojis



def generate_dashboard_summary():
    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql("SELECT * FROM meta", conn)
    if df.empty:
        return

    df["ts"] = pd.to_datetime(df["timestamp"])
    df.sort_values("ts", inplace=True)

    combos = df["combo"].unique()
    for combo in combos:
        sub = df[df["combo"] == combo]
        plt.plot(sub["ts"], sub["accuracy"], label=f"{combo[:30]}")
    plt.title("📈 Accuracy Over Time")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "accuracy_trend.png")
    plt.savefig(path)
    plt.clf()

    for combo in combos:
        sub = df[df["combo"] == combo]
        plt.plot(sub["ts"], sub["shap_quality"], label=f"{combo[:30]}")
    plt.title("🔥 SHAP Quality Over Time")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "shap_trend.png")
    plt.savefig(path)
    plt.clf()

# mlearning/visualize_all_combos.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from features import build_features
from labels import build_labels
from config import DATA_PATH, OUTPUT_DIR
from combo_config import load_combo_rules
from log_config import get_logger
logger = get_logger("model_trainer")

CHART_DIR = os.path.join(OUTPUT_DIR, "strategy_charts")
os.makedirs(CHART_DIR, exist_ok=True)


def generate_visual(df, features, combo_name, target="return_5m", lookback=300):
    try:
        df["signal"] = df[features].all(axis=1).astype(int)
        df_look = df.iloc[-lookback:].copy()
        df_look.index = pd.DatetimeIndex(df_look["Datetime"])

        # Overlays
        apds = [
            mpf.make_addplot(df_look["EMA_8"], color="blue"),
            mpf.make_addplot(df_look["EMA_21"], color="orange"),
            mpf.make_addplot(df_look["orbHigh"], color="green"),
            mpf.make_addplot(df_look["orbLow"], color="red"),
            mpf.make_addplot(df_look["Close"][df_look["signal"] == 1], type='scatter',
                             markersize=100, marker='^', color='lime', panel=0)
        ]

        fig_name = f"{combo_name}_{target}.png"
        fig_path = os.path.join(CHART_DIR, fig_name)

        mpf.plot(
            df_look,
            type="candle",
            style="charles",
            title=f"{combo_name} → {target}",
            ylabel="Price",
            addplot=apds,
            volume=True,
            figratio=(16, 9),
            figscale=1.2,
            savefig=dict(fname=fig_path, dpi=150)
        )

        print(f"✅ Saved chart: {fig_path}")
    except Exception as e:
        print(f"❌ Failed to generate chart for {combo_name}: {e}")


def visualize_all_combos(target="return_5m", lookback=300):
    print("📊 Loading dataset and features...")
    df = pd.read_csv(DATA_PATH)
    df = build_features(df)
    df = build_labels(df)
    df.dropna(inplace=True)

    combos = load_combo_rules()
    print(f"✅ Loaded {len(combos)} combos")

    for combo_name, features in combos.items():
        if not all(f in df.columns for f in features):
            print(f"⚠️ Skipping {combo_name} (missing features)")
            continue
        generate_visual(df.copy(), features, combo_name, target=target, lookback=lookback)


if __name__ == "__main__":
    visualize_all_combos()

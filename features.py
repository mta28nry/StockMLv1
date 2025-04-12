# mlearning/features.py
import pandas as pd
import numpy as np
import logging

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Building robust and symmetric features...")

    # --- Core Binary Strategy Flags ---
    df["below_ema8"] = df["Close"] < df["EMA_8"]
    df["below_ema21"] = df["Close"] < df["EMA_21"]
    df["below_ema200"] = df["Close"] < df["EMA_200"]
    df["above_ema8"] = df["Close"] > df["EMA_8"]
    df["above_ema21"] = df["Close"] > df["EMA_21"]
    df["above_ema200"] = df["Close"] > df["EMA_200"]
    df["below_vwap"] = df["Close"] < df["VWAP"]
    df["above_vwap"] = df["Close"] > df["VWAP"]
    df["below_orbLow"] = df["Close"] < df["orbLow"]
    df["above_orbLow"] = df["Close"] > df["orbLow"]
    df["below_orbHigh"] = df["Close"] < df["orbHigh"]
    df["above_orbHigh"] = df["Close"] > df["orbHigh"]
    df["below_prevLow"] = df["Close"] < df["premarketLow"]
    df["above_prevHigh"] = df["Close"] > df["premarketHigh"]
    df["rsi_low"] = df["RSI"] < 30
    df["rsi_high"] = df["RSI"] > 70
    df["macd_cross_down"] = df["MACD"] < df["MACD_signal"]
    df["macd_cross_up"] = df["MACD"] > df["MACD_signal"]

    # --- Price Positioning ---
    df["distance_vwap"] = (df["Close"] - df["VWAP"]) / df["Close"]
    df["distance_orbLow"] = (df["Close"] - df["orbLow"]) / df["Close"]
    df["distance_orbHigh"] = (df["orbHigh"] - df["Close"]) / df["Close"]
    df["distance_ema8"] = (df["Close"] - df["EMA_8"]) / df["Close"]
    df["distance_ema21"] = (df["Close"] - df["EMA_21"]) / df["Close"]

    # --- Lag Features ---
    for i in range(1, 4):
        df[f"close_lag_{i}"] = df["Close"].shift(i)
        df[f"open_lag_{i}"] = df["Open"].shift(i)
        df[f"return_lag_{i}"] = df["return_15m"].shift(i)
        df[f"rsi_lag_{i}"] = df["RSI"].shift(i)
        df[f"macd_lag_{i}"] = df["MACD"].shift(i)
        df[f"volume_lag_{i}"] = df["Volume"].shift(i)

    # --- Volatility ---
    df["true_range"] = df["High"] - df["Low"]
    df["rolling_vol_5"] = df["true_range"].rolling(5).std()
    df["vol_spike"] = df["Volume"] > df["Volume"].rolling(10).mean() * 1.5

    # --- Time Features ---
    df["session"] = pd.cut(df["hour"], bins=[0, 6, 9, 12, 14], labels=["premarket", "open", "mid", "close"])
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["minute_bin"] = (df["minute"] // 5).astype(int)
    df["is_opening_bar"] = ((df["hour"] == 6) & (df["minute"] == 30)).astype(int)

    # --- Candle Anatomy ---
    df["body"] = abs(df["Close"] - df["Open"])
    df["wick"] = df["High"] - df[["Open", "Close"]].max(axis=1)
    df["tail"] = df[["Open", "Close"]].min(axis=1) - df["Low"]
    df["body_pct"] = df["body"] / (df["High"] - df["Low"] + 1e-5)
    df["wick_to_body"] = df["wick"] / (df["body"] + 1e-5)
    df["tail_to_body"] = df["tail"] / (df["body"] + 1e-5)

    # --- Cross Confirmations ---
    df["ema8_cross_ema21"] = (df["EMA_8"] > df["EMA_21"]).astype(int)
    df["ema21_cross_ema90"] = (df["EMA_21"] > df["EMA_90"]).astype(int)

    # --- Candle Patterns ---
    df["bullish_engulfing"] = ((df["Open"] < df["Close"]) & (df["Open"] < df["close_lag_1"]) & (df["Close"] > df["open_lag_1"])).astype(int)
    df["bearish_engulfing"] = ((df["Open"] > df["Close"]) & (df["Open"] > df["close_lag_1"]) & (df["Close"] < df["open_lag_1"])).astype(int)

    # --- Session Trends ---
    df["session_return"] = df.groupby("session", observed=False)["Close"].transform(lambda x: x.pct_change())
    df["session_volatility"] = df.groupby("session", observed=False)["Close"].transform(lambda x: x.rolling(5).std())

    # --- Momentum & Breakout Signals ---
    df["momentum_3"] = df["Close"] - df["Close"].shift(3)
    df["ema_slope"] = df["EMA_21"] - df["EMA_21"].shift(3)
    df["narrow_range"] = df["true_range"] < df["true_range"].rolling(5).mean()

    df["below_orb"] = df["Close"] < ((df["orbHigh"] + df["orbLow"]) / 2)
    df["above_orb"] = df["Close"] > ((df["orbHigh"] + df["orbLow"]) / 2)

    logging.info(f"✅ Feature generation complete. Total: {len(df.columns)}")
    return df
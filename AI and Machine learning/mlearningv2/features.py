# mlearning/features.py
import pandas as pd
import numpy as np
import logging
from log_config import get_logger
logger = get_logger("model_trainer")

# Ensure logger captures all relevant details
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def build_core_strategy_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Generate binary core trading strategy flags."""
    emas = [8, 21, 200]
    for ema in emas:
        df[f'below_ema{ema}'] = df['Close'] < df[f'EMA_{ema}']
        df[f'above_ema{ema}'] = df['Close'] > df[f'EMA_{ema}']

    df['below_vwap'] = df['Close'] < df['VWAP']
    df['above_vwap'] = df['Close'] > df['VWAP']
    df['below_orbLow'] = df['Close'] < df['orbLow']
    df['above_orbHigh'] = df['Close'] > df['orbHigh']
    df['below_prevLow'] = df['Close'] < df['premarketLow']
    df['above_prevHigh'] = df['Close'] > df['premarketHigh']
    df['rsi_low'] = df['RSI'] < 30
    df['rsi_high'] = df['RSI'] > 70
    df['macd_cross_down'] = df['MACD'] < df['MACD_signal']
    df['macd_cross_up'] = df['MACD'] > df['MACD_signal']

    return df


def build_price_positioning(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate relative distance from key levels."""
    df["distance_vwap"] = (df["Close"] - df["VWAP"]) / df["Close"]
    df["distance_orbLow"] = (df["Close"] - df["orbLow"]) / df["Close"]
    df["distance_orbHigh"] = (df["orbHigh"] - df["Close"]) / df["Close"]
    df["distance_ema8"] = (df["Close"] - df["EMA_8"]) / df["Close"]
    df["distance_ema21"] = (df["Close"] - df["EMA_21"]) / df["Close"]

    return df


def build_lag_features(df: pd.DataFrame, lags: int = 3) -> pd.DataFrame:
    """Generate lagged features to capture short-term price memory."""
    lag_cols = ["Close", "Open", "return_15m", "RSI", "MACD", "Volume"]
    for col in lag_cols:
        for lag in range(1, lags + 1):
            df[f"{col.lower()}_lag_{lag}"] = df[col].shift(lag)

    return df


def build_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate volatility-related features."""
    df["true_range"] = df["High"] - df["Low"]
    df["rolling_vol_5"] = df["true_range"].rolling(window=5, min_periods=1).std()
    df["vol_spike"] = df["Volume"] > df["Volume"].rolling(10, min_periods=1).mean() * 1.5

    return df


def build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features encoding session timing and cyclicality."""
    df["session"] = pd.cut(
        df["hour"], bins=[0, 6, 9, 12, 14, 24],
        labels=["after_hours", "premarket", "open", "mid", "close"],
        right=False, include_lowest=True
        )
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["minute_bin"] = (df["minute"] // 5).astype(int)
    df["is_opening_bar"] = ((df["hour"] == 6) & (df["minute"] == 30)).astype(int)

    return df


def build_candle_anatomy(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze candle stick patterns and ratios."""
    df["body"] = abs(df["Close"] - df["Open"])
    candle_range = df["High"] - df["Low"] + 1e-9  # Avoid division by zero
    df["wick"] = df["High"] - df[["Open", "Close"]].max(axis=1)
    df["tail"] = df[["Open", "Close"]].min(axis=1) - df["Low"]

    df["body_pct"] = df["body"] / candle_range
    df["wick_to_body"] = df["wick"] / (df["body"] + 1e-9)
    df["tail_to_body"] = df["tail"] / (df["body"] + 1e-9)

    return df


def build_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Detect bullish and bearish engulfing candle patterns."""
    df["bullish_engulfing"] = ((df["Close"] > df["Open"]) &
                               (df["Open"] < df["close_lag_1"]) &
                               (df["Close"] > df["open_lag_1"])).astype(int)

    df["bearish_engulfing"] = ((df["Close"] < df["Open"]) &
                               (df["Open"] > df["close_lag_1"]) &
                               (df["Close"] < df["open_lag_1"])).astype(int)

    return df


def build_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate momentum and breakout indicators."""
    df["momentum_3"] = df["Close"] - df["Close"].shift(3)
    df["ema_slope"] = df["EMA_21"] - df["EMA_21"].shift(3)
    df["narrow_range"] = df["true_range"] < df["true_range"].rolling(5, min_periods=1).mean()
    df["below_orb"] = df["Close"] < ((df["orbHigh"] + df["orbLow"]) / 2)
    df["above_orb"] = df["Close"] > ((df["orbHigh"] + df["orbLow"]) / 2)

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Orchestrate the feature building process clearly."""
    logging.info("🚩 Starting robust feature generation...")

    df = (df.pipe(build_core_strategy_flags)
          .pipe(build_price_positioning)
          .pipe(build_lag_features)
          .pipe(build_volatility_features)
          .pipe(build_time_features)
          .pipe(build_candle_anatomy)
          .pipe(build_candle_patterns)
          .pipe(build_momentum_features))

    # Session-specific trend metrics
    df["session_return"] = df.groupby("session", observed=False)["Close"].transform(lambda x: x.pct_change())
    df["session_volatility"] = df.groupby("session", observed=False)["Close"].transform(
        lambda x: x.rolling(5, min_periods=1).std()
        )

    df.dropna(inplace=True)

    logging.info(f"✅ Feature generation complete: {df.shape[1]} features created, {df.shape[0]} rows ready.")
    return df

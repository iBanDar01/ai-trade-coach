from __future__ import annotations
import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def bollinger(series: pd.Series, period: int = 20, k: float = 2.0):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    return mid, mid + k * std, mid - k * std


def macd(series: pd.Series):
    fast = ema(series, 12)
    slow = ema(series, 26)
    line = fast - slow
    signal = ema(line, 9)
    hist = line - signal
    return line, signal, hist


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = ema(df.close, 9)
    df["ema_slow"] = ema(df.close, 21)
    df["rsi"] = rsi(df.close, 14)
    df["macd"], df["macd_signal"], df["macd_hist"] = macd(df.close)
    df["bb_mid"], df["bb_upper"], df["bb_lower"] = bollinger(df.close)
    df["support"] = df.low.rolling(20).min()
    df["resistance"] = df.high.rolling(20).max()
    return df


def score_signal(df: pd.DataFrame) -> dict:
    """Rule-based read of the last candle's indicators.

    IMPORTANT: this is a transparent heuristic (EMA trend + MACD cross +
    RSI zone + Bollinger position + support/resistance proximity), not a
    trained/validated predictive model. On the very short durations this
    app targets (30s-5m) it has no demonstrated statistical edge -- treat
    the reasons list as "what the indicators currently show", not proof
    of a future outcome. WAIT is the honest default when scores are close.
    """
    if len(df) < 30:
        return {"side": "WAIT", "confidence": 0, "reasons": ["Need at least 30 candles of data"]}

    d = enrich(df)
    x = d.iloc[-1]
    p = d.iloc[-2]
    long_score = 0
    short_score = 0
    reasons: list[str] = []

    if x.ema_fast > x.ema_slow:
        long_score += 20
        reasons.append("Trend: fast EMA above slow EMA (bullish)")
    elif x.ema_fast < x.ema_slow:
        short_score += 20
        reasons.append("Trend: fast EMA below slow EMA (bearish)")

    if x.macd_hist > 0 and p.macd_hist <= 0:
        long_score += 25
        reasons.append("Momentum: MACD bullish crossover")
    elif x.macd_hist < 0 and p.macd_hist >= 0:
        short_score += 25
        reasons.append("Momentum: MACD bearish crossover")
    else:
        if x.macd_hist > 0:
            long_score += 10
        if x.macd_hist < 0:
            short_score += 10

    if 50 <= x.rsi <= 68:
        long_score += 15
        reasons.append("RSI supports upside")
    if 32 <= x.rsi < 50:
        short_score += 15
        reasons.append("RSI supports downside")
    if x.rsi < 30:
        long_score += 10
        reasons.append("RSI oversold")
    if x.rsi > 70:
        short_score += 10
        reasons.append("RSI overbought")

    if pd.notna(x.bb_lower) and x.close <= x.bb_lower * 1.005:
        long_score += 15
        reasons.append("Price near lower Bollinger band")
    if pd.notna(x.bb_upper) and x.close >= x.bb_upper * 0.995:
        short_score += 15
        reasons.append("Price near upper Bollinger band")
    if pd.notna(x.support) and x.close <= x.support * 1.01:
        long_score += 10
        reasons.append("Price near recent support")
    if pd.notna(x.resistance) and x.close >= x.resistance * 0.99:
        short_score += 10
        reasons.append("Price near recent resistance")

    momentum = x.close - p.close
    if momentum > 0:
        long_score += 10
    elif momentum < 0:
        short_score += 10

    volatility = float(d.close.tail(20).pct_change().std() or 0)

    if long_score == short_score or max(long_score, short_score) < 45:
        side = "WAIT"
        confidence = int(max(long_score, short_score))
    elif long_score > short_score:
        side = "BUY"
        confidence = int(min(95, long_score))
    else:
        side = "SELL"
        confidence = int(min(95, short_score))

    return {
        "side": side,
        "confidence": confidence,
        "long_score": int(long_score),
        "short_score": int(short_score),
        "price": round(float(x.close), 5),
        "rsi": round(float(x.rsi), 2),
        "volatility": round(volatility, 5),
        "reasons": reasons[-6:],
    }

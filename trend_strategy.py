from __future__ import annotations

import numpy as np
import pandas as pd


def intraday_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """5분봉 대장주 추세매매용 VWAP·돌파·눌림 지표."""
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)

    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(set(d.columns)):
        return pd.DataFrame()

    d = d.dropna().copy()
    if len(d) < 25:
        return pd.DataFrame()

    close = d["Close"].astype(float)
    high = d["High"].astype(float)
    low = d["Low"].astype(float)
    open_ = d["Open"].astype(float)
    volume = d["Volume"].astype(float)

    typical = (high + low + close) / 3.0
    d["VWAP"] = (typical * volume).cumsum() / volume.cumsum().replace(0, np.nan)
    d["MA5"] = close.rolling(5).mean()
    d["MA20"] = close.rolling(20).mean()
    d["VOL_MA20"] = volume.rolling(20).mean()
    d["VOL_RATIO20"] = volume / d["VOL_MA20"].replace(0, np.nan)

    d["PREV_12_HIGH"] = high.shift(1).rolling(12).max()
    d["DAY_HIGH"] = high.cummax()
    d["BREAKOUT"] = close > d["PREV_12_HIGH"]

    prev_close = close.shift(1)
    prev_vwap = d["VWAP"].shift(1)
    d["PULLBACK_RECLAIM"] = (
        (prev_close <= prev_vwap * 1.005)
        & (close > d["VWAP"])
        & (close > open_)
        & (d["MA5"] > d["MA20"])
    )

    d["FROM_DAY_HIGH_PCT"] = (close / d["DAY_HIGH"] - 1.0) * 100.0
    return d.dropna()


def score_leader_trend(df: pd.DataFrame):
    """대장주 추세점수 0~100. 특정 트레이더 복제가 아니라 규칙화한 별도 전략."""
    d = intraday_trend_indicators(df)
    if d.empty:
        return None

    x = d.iloc[-1]
    score = 0
    reasons = []

    if float(x["Close"]) > float(x["VWAP"]):
        score += 20
        reasons.append("VWAP 위")

    if float(x["MA5"]) > float(x["MA20"]):
        score += 15
        reasons.append("5봉선 > 20봉선")

    vr = float(x["VOL_RATIO20"])
    if vr >= 2.0:
        score += 20
        reasons.append("거래량 2배 이상")
    elif vr >= 1.5:
        score += 15
        reasons.append("거래량 1.5배 이상")
    elif vr >= 1.2:
        score += 8
        reasons.append("거래량 증가")

    if bool(x["BREAKOUT"]):
        score += 20
        reasons.append("최근 1시간 고가 돌파")

    if bool(x["PULLBACK_RECLAIM"]):
        score += 15
        reasons.append("VWAP 눌림 후 재상승")

    from_high = float(x["FROM_DAY_HIGH_PCT"])
    if from_high >= -0.5:
        score += 10
        reasons.append("당일 고가 0.5% 이내")
    elif from_high >= -1.5:
        score += 5
        reasons.append("당일 고가 1.5% 이내")

    vwap_gap = (float(x["Close"]) / float(x["VWAP"]) - 1.0) * 100.0
    if vwap_gap >= 5.0:
        score -= 15
        reasons.append("VWAP 대비 과열 감점")
    elif vwap_gap >= 3.0:
        score -= 7
        reasons.append("VWAP 대비 단기 과열 감점")

    score = max(0, min(100, int(round(score))))
    if score >= 70 and float(x["Close"]) > float(x["VWAP"]):
        signal = "🟢 추세매수 후보"
    elif score >= 50:
        signal = "🟡 추세관망"
    else:
        signal = "⚪ 추세약함"

    return {
        "추세점수": score,
        "추세판정": signal,
        "VWAP": round(float(x["VWAP"]), 2),
        "VWAP괴리율": round(vwap_gap, 2),
        "거래량배수": round(vr, 2),
        "당일고가거리": round(from_high, 2),
        "돌파": bool(x["BREAKOUT"]),
        "눌림재상승": bool(x["PULLBACK_RECLAIM"]),
        "추세이유": ", ".join(reasons),
    }

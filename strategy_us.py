from __future__ import annotations

from datetime import datetime
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")

# KIS 해외주식 주문에 쓰는 거래소 코드.
# 기본 유니버스는 여기서 명시해 잘못된 NASD 고정 주문을 방지한다.
EXCHANGE_MAP = {
    "AAPL": "NASD", "MSFT": "NASD", "NVDA": "NASD", "AMZN": "NASD",
    "META": "NASD", "TSLA": "NASD", "AMD": "NASD", "GOOGL": "NASD",
    "AVGO": "NASD", "NFLX": "NASD", "PLTR": "NASD", "MU": "NASD",
    "INTC": "NASD", "SMCI": "NASD", "ARM": "NASD", "QCOM": "NASD",
    "AMAT": "NASD", "MRVL": "NASD", "CRWD": "NASD",
    "COIN": "NASD", "HOOD": "NASD", "SOFI": "NASD", "MSTR": "NASD",
    "RBLX": "NYSE", "UBER": "NYSE", "PANW": "NASD", "TSM": "NYSE",
    "LLY": "NYSE", "JPM": "NYSE", "BAC": "NYSE",
}


def _safe_float(v, default: float = 0.0) -> float:
    try:
        x = float(v)
        if np.isfinite(x):
            return x
    except Exception:
        pass
    return float(default)


def _pct_change_from(close: pd.Series, bars: int) -> float:
    if close is None or len(close) < 2:
        return 0.0
    idx = max(0, len(close) - 1 - int(bars))
    base = _safe_float(close.iloc[idx])
    last = _safe_float(close.iloc[-1])
    if base <= 0:
        return 0.0
    return (last / base - 1.0) * 100.0


def _score_frame(symbol: str, frame: pd.DataFrame) -> dict | None:
    """1분봉에서 '오늘 많이 오른 종목'보다 '지금 강해지는 종목'을 점수화한다."""
    if frame is None or frame.empty:
        return None

    d = frame.copy()
    rename = {str(c).lower(): c for c in d.columns}
    required = ["open", "high", "low", "close", "volume"]
    if not all(x in rename for x in required):
        return None

    d = d[[rename[x] for x in required]].copy()
    d.columns = ["Open", "High", "Low", "Close", "Volume"]
    for c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["Close", "High", "Low", "Open"])
    if len(d) < 6:
        return None

    close = d["Close"].astype(float)
    high = d["High"].astype(float)
    volume = d["Volume"].fillna(0).astype(float)

    last = _safe_float(close.iloc[-1])
    if last <= 0:
        return None

    ret5 = _pct_change_from(close, 5)
    ret10 = _pct_change_from(close, 10)
    ret20 = _pct_change_from(close, 20)

    first_open = _safe_float(d["Open"].iloc[0], last)
    day_ret = (last / first_open - 1.0) * 100.0 if first_open > 0 else 0.0

    session_high = _safe_float(high.max(), last)
    high_distance = (last / session_high - 1.0) * 100.0 if session_high > 0 else 0.0

    prev_high = _safe_float(high.iloc[:-1].tail(20).max(), last)
    breakout = bool(prev_high > 0 and last >= prev_high * 0.998)

    ema5 = _safe_float(close.ewm(span=5, adjust=False).mean().iloc[-1], last)
    ema13 = _safe_float(close.ewm(span=13, adjust=False).mean().iloc[-1], last)
    trend_up = bool(last >= ema5 and ema5 >= ema13)

    recent_vol = _safe_float(volume.tail(min(5, len(volume))).mean(), 0.0)
    prior = volume.iloc[:-5].tail(20) if len(volume) > 5 else pd.Series(dtype=float)
    prior_vol = _safe_float(prior.mean(), 0.0) if len(prior) else 0.0
    if prior_vol > 0:
        vol_ratio = recent_vol / prior_vol
    else:
        vol_ratio = 1.0 if recent_vol > 0 else 0.0

    # 최근 모멘텀에 가장 큰 비중을 둔다.
    score = 0.0
    score += max(0.0, min(25.0, ret5 / 2.0 * 25.0))
    score += max(0.0, min(20.0, ret10 / 4.0 * 20.0))
    score += max(0.0, min(10.0, ret20 / 6.0 * 10.0))
    score += max(0.0, min(20.0, (vol_ratio - 1.0) / 2.0 * 20.0))
    score += 15.0 if breakout else 0.0
    score += 10.0 if trend_up else 0.0

    # 당일 고점에서 멀어지면 강하게 감점한다.
    if high_distance >= -0.8:
        score += 10.0
    elif high_distance < -3.0:
        score -= 20.0
    elif high_distance < -2.0:
        score -= 10.0

    # 이미 5분 사이 과도하게 튄 종목은 뒤늦은 추격 위험 감점.
    if ret5 > 4.0:
        score -= 12.0
    if day_ret < -1.0:
        score -= 10.0

    score = round(max(0.0, min(100.0, score)), 1)

    momentum_weak = bool(
        ret5 < -0.20
        or high_distance < -3.0
        or ((not trend_up) and ret10 < 0.0)
    )

    buy_candidate = bool(
        score >= 58.0
        and ret5 >= 0.15
        and ret10 >= 0.20
        and (vol_ratio >= 1.15 or breakout)
        and high_distance >= -2.5
        and trend_up
        and not momentum_weak
    )

    signal = "🟢 매수 후보" if buy_candidate else ("🔴 모멘텀 약화" if momentum_weak else "🟡 관망")

    return {
        "종목코드": str(symbol).upper(),
        "종목명": str(symbol).upper(),
        "거래소": EXCHANGE_MAP.get(str(symbol).upper(), "NASD"),
        "현재가": round(last, 4),
        "최근5분수익률": round(ret5, 3),
        "최근10분수익률": round(ret10, 3),
        "최근20분수익률": round(ret20, 3),
        "당일수익률": round(day_ret, 3),
        "거래량배수": round(vol_ratio, 2),
        "고점대비": round(high_distance, 3),
        "돌파": breakout,
        "단기추세": "상승" if trend_up else "비상승",
        "모멘텀약화": momentum_weak,
        "종합점수": score,
        "주도주점수": score,
        "판정": signal,
        "스캔시각": datetime.now(ET).isoformat(timespec="seconds"),
    }


def _extract_symbol_frame(batch: pd.DataFrame, symbol: str, universe_size: int) -> pd.DataFrame | None:
    if batch is None or batch.empty:
        return None
    if not isinstance(batch.columns, pd.MultiIndex):
        return batch.copy() if universe_size == 1 else None

    level0 = set(str(x) for x in batch.columns.get_level_values(0))
    level1 = set(str(x) for x in batch.columns.get_level_values(1))

    if symbol in level0:
        try:
            return batch[symbol].copy()
        except Exception:
            pass
    if symbol in level1:
        try:
            return batch.xs(symbol, axis=1, level=1).copy()
        except Exception:
            pass
    return None


def build_us_top5(universe: Iterable[str]) -> pd.DataFrame:
    """
    미국 정규장 1분봉을 한 번에 받아 최근 5/10/20분 모멘텀과 거래량 가속을 평가한다.

    - 후보 탐색 데이터: yfinance 배치 1분봉
    - 실제 주문가격/잔고: auto_engine에서 KIS API를 다시 확인
    - 스캔 실패 시 예외를 올려 Worker가 stale 후보를 비우도록 한다.
    """
    symbols = []
    seen = set()
    for raw in universe or []:
        s = str(raw).strip().upper()
        if s and s not in seen:
            seen.add(s)
            symbols.append(s)

    if not symbols:
        return pd.DataFrame()

    try:
        import yfinance as yf
    except Exception as e:
        raise RuntimeError(f"yfinance import 실패: {e}") from e

    try:
        batch = yf.download(
            tickers=symbols,
            period="1d",
            interval="1m",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
            prepost=False,
            timeout=12,
        )
    except TypeError:
        # 구버전 yfinance의 timeout/group_by 호환 보완
        batch = yf.download(
            tickers=symbols,
            period="1d",
            interval="1m",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
            prepost=False,
        )
    except Exception as e:
        raise RuntimeError(f"미국 1분봉 배치 조회 실패: {type(e).__name__}: {e}") from e

    if batch is None or batch.empty:
        raise RuntimeError("미국 1분봉 배치 결과가 비어 있음")

    rows = []
    for symbol in symbols:
        frame = _extract_symbol_frame(batch, symbol, len(symbols))
        scored = _score_frame(symbol, frame) if frame is not None else None
        if scored:
            rows.append(scored)

    if not rows:
        raise RuntimeError("분석 가능한 미국 1분봉 종목이 없음")

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["종합점수", "최근5분수익률", "거래량배수"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    df["순위"] = [f"{i + 1}위" for i in range(len(df))]

    # 매수 후보를 우선 배치하되 관망 종목도 비교용으로 남긴다.
    green = df[df["판정"].astype(str).str.contains("매수 후보", na=False)]
    nongreen = df[~df.index.isin(green.index)]
    df = pd.concat([green, nongreen], ignore_index=True)
    df["순위"] = [f"{i + 1}위" for i in range(len(df))]

    return df.head(5).reset_index(drop=True)

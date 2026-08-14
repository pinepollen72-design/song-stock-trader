from __future__ import annotations

import os
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from trader_core import discover_domestic_candidates

KST = ZoneInfo("Asia/Seoul")


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


def _normalize_intraday(raw: dict) -> pd.DataFrame:
    rows = (raw or {}).get("output2", []) or []
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    need = {
        "stck_cntg_hour": "Time",
        "stck_oprc": "Open",
        "stck_hgpr": "High",
        "stck_lwpr": "Low",
        "stck_prpr": "Close",
        "cntg_vol": "Volume",
    }
    if not all(k in df.columns for k in need):
        return pd.DataFrame()

    d = df[list(need.keys())].rename(columns=need).copy()
    d["Time"] = d["Time"].astype(str).str.replace(":", "", regex=False).str.zfill(6)
    for c in ("Open", "High", "Low", "Close", "Volume"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["Open", "High", "Low", "Close"])
    d = d[d["Close"] > 0]
    d = d.sort_values("Time").drop_duplicates(subset=["Time"], keep="last")

    # 현재 분은 아직 완성되지 않은 봉일 수 있어 거래량 점수 왜곡을 막기 위해 제외한다.
    # 다만 데이터가 너무 적으면 그대로 둔다.
    if len(d) >= 8:
        now_hhmm = datetime.now(KST).strftime("%H%M")
        if str(d.iloc[-1]["Time"])[:4] == now_hhmm:
            d = d.iloc[:-1].copy()

    return d.reset_index(drop=True)


def _score_candidate(base_row: pd.Series, intraday: pd.DataFrame) -> dict | None:
    if intraday is None or intraday.empty or len(intraday) < 8:
        return None

    d = intraday.copy()
    close = d["Close"].astype(float)
    high = d["High"].astype(float)
    volume = d["Volume"].fillna(0).astype(float)

    last = _safe_float(close.iloc[-1])
    if last <= 0:
        return None

    ret3 = _pct_change_from(close, 3)
    ret5 = _pct_change_from(close, 5)
    ret10 = _pct_change_from(close, 10)
    ret20 = _pct_change_from(close, 20)

    day_ret = _safe_float(base_row.get("등락률", 0.0))
    lead_score = _safe_float(base_row.get("주도주점수", 0.0))

    recent_vol = _safe_float(volume.tail(min(3, len(volume))).mean(), 0.0)
    prior = volume.iloc[:-3].tail(12) if len(volume) > 3 else pd.Series(dtype=float)
    prior_vol = _safe_float(prior.mean(), 0.0) if len(prior) else 0.0
    vol_ratio = recent_vol / prior_vol if prior_vol > 0 else (1.0 if recent_vol > 0 else 0.0)

    ema5 = _safe_float(close.ewm(span=5, adjust=False).mean().iloc[-1], last)
    ema13 = _safe_float(close.ewm(span=13, adjust=False).mean().iloc[-1], last)
    trend_up = bool(last >= ema5 and ema5 >= ema13)

    prior_high = _safe_float(high.iloc[:-1].tail(20).max(), last)
    breakout = bool(prior_high > 0 and last >= prior_high * 0.998)

    recent_high = _safe_float(high.tail(30).max(), last)
    high_distance = (last / recent_high - 1.0) * 100.0 if recent_high > 0 else 0.0

    up3 = int((close.diff().tail(3) > 0).sum())

    # 누적 강도는 후보 선별용 보조점수로만 쓰고, 실제 진입점수는 최근 모멘텀에 집중한다.
    score = min(12.0, max(0.0, lead_score) * 0.12)
    score += max(0.0, min(18.0, ret3 / 1.2 * 18.0))
    score += max(0.0, min(18.0, ret5 / 2.0 * 18.0))
    score += max(0.0, min(12.0, ret10 / 3.0 * 12.0))
    score += max(0.0, min(18.0, (vol_ratio - 1.0) / 1.5 * 18.0))
    score += 10.0 if breakout else 0.0
    score += 8.0 if trend_up else 0.0
    score += 4.0 if up3 >= 2 else 0.0

    if high_distance >= -0.8:
        score += 8.0
    elif high_distance < -3.0:
        score -= 18.0
    elif high_distance < -2.0:
        score -= 10.0

    # 이미 너무 급하게 튄 뒤 따라붙는 상황은 추격 위험으로 감점한다.
    if ret3 > 2.8:
        score -= 10.0
    if ret5 > 4.5:
        score -= 10.0

    # 오전 급등 종목이 오후까지 누적점수만 높게 남는 문제를 차단한다.
    stale_leader = bool(
        day_ret >= 5.0
        and ret10 <= 0.10
        and vol_ratio < 1.05
    )

    momentum_weak = bool(
        ret3 < -0.25
        or ret5 < -0.40
        or high_distance < -3.0
        or ((not trend_up) and ret10 <= 0.0)
        or stale_leader
    )

    score = round(max(0.0, min(100.0, score)), 1)

    buy_candidate = bool(
        score >= 58.0
        and ret3 >= 0.03
        and ret5 >= 0.08
        and ret10 >= 0.0
        and (vol_ratio >= 1.12 or breakout)
        and high_distance >= -2.2
        and trend_up
        and not momentum_weak
    )

    signal = (
        "🟢 매수 후보"
        if buy_candidate
        else "🔴 모멘텀 약화"
        if momentum_weak
        else "🟡 관망"
    )

    return {
        "종목코드": str(base_row.get("종목코드", "")).zfill(6),
        "종목명": str(base_row.get("종목명", "")),
        "현재가": round(last, 2),
        "등락률": round(day_ret, 3),
        "누적거래량": _safe_float(base_row.get("누적거래량", 0.0)),
        "거래대금": _safe_float(base_row.get("거래대금", 0.0)),
        "최근3분수익률": round(ret3, 3),
        "최근5분수익률": round(ret5, 3),
        "최근10분수익률": round(ret10, 3),
        "최근20분수익률": round(ret20, 3),
        "거래량배수": round(vol_ratio, 2),
        "최근30분고점대비": round(high_distance, 3),
        # auto_engine의 공통 필드명과 호환
        "고점대비": round(high_distance, 3),
        "돌파": breakout,
        "단기추세": "상승" if trend_up else "비상승",
        "모멘텀약화": momentum_weak,
        "후행강세": stale_leader,
        "주도주점수": round(lead_score, 1),
        "종합점수": score,
        "판정": signal,
        "스캔시각": datetime.now(KST).isoformat(timespec="seconds"),
    }


def build_kr_top5(client) -> pd.DataFrame:
    """
    국내 단타용 TOP5.

    1) KIS 거래량/거래대금 순위로 넓은 후보군을 빠르게 만든다.
    2) 상위 후보만 KIS 당일 1분봉으로 확인한다.
    3) 최근 3/5/10/20분 가격 가속 + 최근 거래량 증가 + 돌파/추세를 중심으로 점수화한다.
    4) 오전에만 강했고 최근 모멘텀이 죽은 종목은 '후행강세'로 감점/차단한다.

    실제 주문 직전 현재가와 잔고는 auto_engine이 KIS에서 다시 확인한다.
    """
    pool_size = max(10, int(os.getenv("KR_CANDIDATE_POOL", "30")))
    env_default = "8" if str(getattr(client, "env", "demo")).lower() == "demo" else "12"
    scan_count = max(5, int(os.getenv("KR_MOMENTUM_SCAN_COUNT", env_default)))

    base = discover_domestic_candidates(client, top_n=pool_size)
    if base is None or base.empty:
        raise RuntimeError("국내 거래량/거래대금 후보가 없음")

    # 누적강도 상위권 중 실제 1분 모멘텀을 검사한다.
    base = base.sort_values(
        [c for c in ("주도주점수", "거래대금", "누적거래량") if c in base.columns],
        ascending=False,
    ).head(scan_count)

    rows = []
    errors = []

    for _, base_row in base.iterrows():
        symbol = str(base_row.get("종목코드", "")).zfill(6)
        if len(symbol) != 6 or not symbol.isdigit():
            continue
        try:
            raw = client.domestic_intraday_minutes(symbol)
            intraday = _normalize_intraday(raw)
            scored = _score_candidate(base_row, intraday)
            if scored:
                rows.append(scored)
            else:
                errors.append(f"{symbol}: 분봉 부족/형식 이상")
        except Exception as e:
            errors.append(f"{symbol}: {type(e).__name__}: {e}")

    if not rows:
        detail = " / ".join(errors[:5])
        raise RuntimeError(f"분석 가능한 국내 단기 모멘텀 종목이 없음{': ' + detail if detail else ''}")

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["종합점수", "최근3분수익률", "거래량배수", "주도주점수"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    # 매수 후보를 위로 올리되, 비교/진단용 관망·약화 종목도 TOP5에는 남긴다.
    green = df[df["판정"].astype(str).str.contains("매수 후보", na=False)]
    others = df[~df.index.isin(green.index)]
    df = pd.concat([green, others], ignore_index=True)
    df["순위"] = [f"{i + 1}위" for i in range(len(df))]

    return df.head(5).reset_index(drop=True)

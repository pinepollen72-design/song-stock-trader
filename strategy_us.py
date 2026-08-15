from __future__ import annotations

from datetime import datetime
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")

# 기술주 중심 미국 유니버스의 시장 상대강도 기준.
# 별도 API 호출 없이 기존 yfinance 배치에 QQQ 1종목만 같이 넣는다.
BENCHMARK_SYMBOL = "QQQ"

# KIS 해외주식 주문에 쓰는 거래소 코드.
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


def _prepare_frame(frame: pd.DataFrame | None) -> pd.DataFrame | None:
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

    d["Volume"] = d["Volume"].fillna(0.0)
    return d


def _benchmark_metrics(frame: pd.DataFrame | None) -> dict:
    d = _prepare_frame(frame)
    if d is None:
        return {
            "day_ret": 0.0,
            "ret5": 0.0,
            "ret10": 0.0,
            "available": False,
        }

    close = d["Close"].astype(float)
    last = _safe_float(close.iloc[-1])
    first_open = _safe_float(d["Open"].iloc[0], last)

    return {
        "day_ret": (last / first_open - 1.0) * 100.0 if first_open > 0 else 0.0,
        "ret5": _pct_change_from(close, 5),
        "ret10": _pct_change_from(close, 10),
        "available": True,
    }


def _score_frame(
    symbol: str,
    frame: pd.DataFrame,
    benchmark: dict | None = None,
) -> dict | None:
    """
    미국 단타 대장주 점수 V3.

    핵심 원칙
    1) '급락 후 순간 반등'보다 '당일 강한 종목'을 우선
    2) QQQ 대비 상대강도를 반영
    3) 당일 고가 근접도 + VWAP 위치를 대장주 자격에 반영
    4) 5/10/20분 모멘텀과 거래량 가속은 진입 타이밍에 사용
    """
    d = _prepare_frame(frame)
    if d is None:
        return None

    close = d["Close"].astype(float)
    high = d["High"].astype(float)
    low = d["Low"].astype(float)
    volume = d["Volume"].astype(float)

    last = _safe_float(close.iloc[-1])
    if last <= 0:
        return None

    # ------------------------------------------------------------
    # 1) 가격/모멘텀
    # ------------------------------------------------------------
    ret5 = _pct_change_from(close, 5)
    ret10 = _pct_change_from(close, 10)
    ret20 = _pct_change_from(close, 20)

    first_open = _safe_float(d["Open"].iloc[0], last)
    day_ret = (last / first_open - 1.0) * 100.0 if first_open > 0 else 0.0

    session_high = _safe_float(high.max(), last)
    high_distance = (
        (last / session_high - 1.0) * 100.0
        if session_high > 0 else 0.0
    )

    prev_high = _safe_float(high.iloc[:-1].tail(20).max(), last)
    breakout = bool(prev_high > 0 and last >= prev_high * 0.998)

    ema5 = _safe_float(
        close.ewm(span=5, adjust=False).mean().iloc[-1],
        last,
    )
    ema13 = _safe_float(
        close.ewm(span=13, adjust=False).mean().iloc[-1],
        last,
    )
    trend_up = bool(last >= ema5 and ema5 >= ema13)

    # ------------------------------------------------------------
    # 2) VWAP
    # ------------------------------------------------------------
    typical = (high + low + close) / 3.0
    total_volume = _safe_float(volume.sum(), 0.0)

    if total_volume > 0:
        vwap = _safe_float(
            (typical * volume).sum() / total_volume,
            last,
        )
    else:
        vwap = _safe_float(close.mean(), last)

    vwap_gap = (
        (last / vwap - 1.0) * 100.0
        if vwap > 0 else 0.0
    )

    # ------------------------------------------------------------
    # 3) 거래량 가속
    # ------------------------------------------------------------
    recent_vol = _safe_float(
        volume.tail(min(5, len(volume))).mean(),
        0.0,
    )
    prior = (
        volume.iloc[:-5].tail(20)
        if len(volume) > 5
        else pd.Series(dtype=float)
    )
    prior_vol = _safe_float(prior.mean(), 0.0) if len(prior) else 0.0

    if prior_vol > 0:
        vol_ratio = recent_vol / prior_vol
    else:
        vol_ratio = 1.0 if recent_vol > 0 else 0.0

    # ------------------------------------------------------------
    # 4) QQQ 대비 상대강도
    # ------------------------------------------------------------
    benchmark = benchmark or {}
    qqq_day = _safe_float(benchmark.get("day_ret", 0.0))
    qqq_5 = _safe_float(benchmark.get("ret5", 0.0))
    qqq_10 = _safe_float(benchmark.get("ret10", 0.0))
    benchmark_available = bool(benchmark.get("available", False))

    rel_day = day_ret - qqq_day if benchmark_available else 0.0
    rel5 = ret5 - qqq_5 if benchmark_available else 0.0
    rel10 = ret10 - qqq_10 if benchmark_available else 0.0

    # ------------------------------------------------------------
    # 5) 급락 후 순간반등 함정
    # ------------------------------------------------------------
    # 예: 당일 -3~-5%인데 최근 5분만 잠깐 튀는 종목.
    # 단, 시장 전체가 매우 약해서 QQQ보다 크게 강한 경우는 예외 가능.
    rebound_trap = bool(
        day_ret <= -1.5
        and ret5 >= 0.20
        and (
            (benchmark_available and rel_day < 0.75)
            or ((not benchmark_available) and vwap_gap < 0.0)
        )
    )

    # ------------------------------------------------------------
    # 6) 대장주 자격시험
    # ------------------------------------------------------------
    # 빨간 종목이라도 시장이 폭락하고 그 종목이 QQQ보다 현저히 강하면
    # 대장주가 될 수 있으므로 day_ret만으로 단순 차단하지 않는다.
    day_strength_ok = bool(
        day_ret >= -0.50
        or (benchmark_available and rel_day >= 1.00)
    )

    relative_ok = bool(
        (not benchmark_available)
        or rel_day >= -0.15
        or day_ret >= 1.50
    )

    high_ok = bool(high_distance >= -1.80)
    vwap_ok = bool(vwap_gap >= -0.35)
    momentum_ok = bool(ret5 >= 0.10 and ret10 >= 0.15)
    flow_ok = bool(vol_ratio >= 1.10 or breakout)

    leader_qualified = bool(
        day_strength_ok
        and relative_ok
        and high_ok
        and vwap_ok
        and trend_up
        and not rebound_trap
    )

    # ------------------------------------------------------------
    # 7) 점수
    #    최근반등 하나만으로 60점을 넘지 못하도록
    #    당일강도/상대강도 비중을 크게 올렸다.
    # ------------------------------------------------------------
    score = 0.0

    # 최근 모멘텀: 최대 36
    score += max(0.0, min(18.0, ret5 / 1.50 * 18.0))
    score += max(0.0, min(12.0, ret10 / 2.50 * 12.0))
    score += max(0.0, min(6.0, ret20 / 4.00 * 6.0))

    # 당일 강도: 최대 12
    score += max(
        0.0,
        min(12.0, (day_ret + 0.50) / 3.00 * 12.0),
    )

    # 시장 대비 상대강도: 최대 20
    if benchmark_available:
        score += max(
            0.0,
            min(13.0, (rel_day + 0.10) / 1.60 * 13.0),
        )
        score += max(
            0.0,
            min(7.0, (rel10 + 0.05) / 0.80 * 7.0),
        )
    else:
        # 벤치마크 데이터가 잠깐 없을 때 전체점수를 과도하게 낮추지 않는다.
        score += 8.0 if day_ret >= 0.5 else 3.0

    # 거래량: 최대 10
    score += max(
        0.0,
        min(10.0, (vol_ratio - 1.0) / 1.50 * 10.0),
    )

    # 고점 근접도: 최대 7
    if high_distance >= -0.40:
        score += 7.0
    elif high_distance >= -0.80:
        score += 5.0
    elif high_distance >= -1.30:
        score += 3.0
    elif high_distance >= -1.80:
        score += 1.0

    # VWAP: 최대 6
    if vwap_gap >= 0.30:
        score += 6.0
    elif vwap_gap >= 0.0:
        score += 4.0
    elif vwap_gap >= -0.35:
        score += 1.0

    # 돌파/정배열: 최대 9
    score += 6.0 if breakout else 0.0
    score += 3.0 if trend_up else 0.0

    # ------------------------------------------------------------
    # 8) 위험 감점
    # ------------------------------------------------------------
    # 5분 급등 추격 방지
    if ret5 > 4.0:
        score -= 12.0

    # 급락 후 순간반등은 강하게 감점
    if rebound_trap:
        score -= 30.0

    # 시장보다 약하면서 당일 크게 하락
    if (
        day_ret <= -2.0
        and (
            (benchmark_available and rel_day < 0.0)
            or not benchmark_available
        )
    ):
        score -= 20.0

    if high_distance < -2.5:
        score -= 15.0

    if vwap_gap < -1.0:
        score -= 10.0

    score = round(max(0.0, min(100.0, score)), 1)

    # ------------------------------------------------------------
    # 9) 최종 판정
    # ------------------------------------------------------------
    momentum_weak = bool(
        ret5 < -0.20
        or ret10 < -0.20
        or high_distance < -2.5
        or vwap_gap < -1.0
        or ((not trend_up) and ret10 < 0.0)
    )

    buy_candidate = bool(
        leader_qualified
        and score >= 62.0
        and momentum_ok
        and flow_ok
        and not momentum_weak
    )

    fail_reasons = []
    if rebound_trap:
        fail_reasons.append("급락후반등")
    if not day_strength_ok:
        fail_reasons.append("당일강도약함")
    if not relative_ok:
        fail_reasons.append("시장대비약함")
    if not high_ok:
        fail_reasons.append("고점거리큼")
    if not vwap_ok:
        fail_reasons.append("VWAP아래")
    if not trend_up:
        fail_reasons.append("단기추세비상승")
    if not momentum_ok:
        fail_reasons.append("5·10분모멘텀부족")
    if not flow_ok:
        fail_reasons.append("거래량·돌파부족")
    if score < 62.0:
        fail_reasons.append("점수미달")

    if buy_candidate:
        signal = "🟢 매수 후보"
    elif momentum_weak or rebound_trap:
        signal = "🔴 대장주 탈락"
    else:
        signal = "🟡 관망"

    return {
        "종목코드": str(symbol).upper(),
        "종목명": str(symbol).upper(),
        "거래소": EXCHANGE_MAP.get(str(symbol).upper(), "NASD"),

        "현재가": round(last, 4),

        "최근5분수익률": round(ret5, 3),
        "최근10분수익률": round(ret10, 3),
        "최근20분수익률": round(ret20, 3),

        # auto_engine / 블랙박스 호환용 별칭을 함께 둔다.
        "당일수익률": round(day_ret, 3),
        "당일등락률": round(day_ret, 3),
        "등락률": round(day_ret, 3),

        "QQQ당일수익률": round(qqq_day, 3),
        "QQQ5분수익률": round(qqq_5, 3),
        "QQQ10분수익률": round(qqq_10, 3),

        "상대강도": round(rel_day, 3),
        "상대강도5분": round(rel5, 3),
        "상대강도10분": round(rel10, 3),

        "거래량배수": round(vol_ratio, 2),
        "고점대비": round(high_distance, 3),

        "VWAP": round(vwap, 4),
        "VWAP괴리율": round(vwap_gap, 3),

        "돌파": breakout,
        "단기추세": "상승" if trend_up else "비상승",

        "급락반등함정": rebound_trap,
        "대장주자격": leader_qualified,
        "대장주탈락사유": " / ".join(fail_reasons),

        "모멘텀약화": momentum_weak,
        "종합점수": score,
        "주도주점수": score,
        "판정": signal,

        "스캔시각": datetime.now(ET).isoformat(timespec="seconds"),
    }


def _extract_symbol_frame(
    batch: pd.DataFrame,
    symbol: str,
    universe_size: int,
) -> pd.DataFrame | None:
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
    미국 대장주 TOP5 V3.

    - 1분봉 배치 1회
    - 유니버스 + QQQ를 같은 요청으로 조회
    - 실제 주문가격/잔고는 auto_engine의 KIS API가 다시 확인
    - stale 후보 재사용 방지를 위해 조회 실패는 예외로 올린다.
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

    # 속도를 위해 별도 QQQ 네트워크 호출을 하지 않고
    # 기존 배치에 QQQ 하나만 추가한다.
    download_symbols = list(symbols)
    if BENCHMARK_SYMBOL not in download_symbols:
        download_symbols.append(BENCHMARK_SYMBOL)

    try:
        import yfinance as yf
    except Exception as e:
        raise RuntimeError(f"yfinance import 실패: {e}") from e

    try:
        batch = yf.download(
            tickers=download_symbols,
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
        batch = yf.download(
            tickers=download_symbols,
            period="1d",
            interval="1m",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
            prepost=False,
        )
    except Exception as e:
        raise RuntimeError(
            f"미국 1분봉 배치 조회 실패: {type(e).__name__}: {e}"
        ) from e

    if batch is None or batch.empty:
        raise RuntimeError("미국 1분봉 배치 결과가 비어 있음")

    benchmark_frame = _extract_symbol_frame(
        batch,
        BENCHMARK_SYMBOL,
        len(download_symbols),
    )
    benchmark = _benchmark_metrics(benchmark_frame)

    rows = []
    for symbol in symbols:
        frame = _extract_symbol_frame(
            batch,
            symbol,
            len(download_symbols),
        )
        scored = (
            _score_frame(
                symbol,
                frame,
                benchmark=benchmark,
            )
            if frame is not None
            else None
        )
        if scored:
            rows.append(scored)

    if not rows:
        raise RuntimeError("분석 가능한 미국 1분봉 종목이 없음")

    df = pd.DataFrame(rows)

    # 진짜 대장주/매수후보를 최우선.
    # 그 다음은 점수 -> 상대강도 -> 당일강도 -> 최근5분 모멘텀 순.
    green_mask = df["판정"].astype(str).str.contains(
        "매수 후보",
        na=False,
    )
    df["_green"] = green_mask.astype(int)

    df = df.sort_values(
        [
            "_green",
            "종합점수",
            "상대강도",
            "당일등락률",
            "최근5분수익률",
            "거래량배수",
        ],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)

    df = df.drop(columns=["_green"])
    df["순위"] = [f"{i + 1}위" for i in range(len(df))]

    return df.head(5).reset_index(drop=True)

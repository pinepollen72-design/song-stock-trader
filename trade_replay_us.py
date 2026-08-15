from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from strategy_us import (
    BENCHMARK_SYMBOL,
    _benchmark_metrics,
    _extract_symbol_frame,
    _score_frame,
    _split_session_for_date,
)

ET = ZoneInfo("America/New_York")


def _resolve_state_dir() -> Path:
    explicit = os.getenv("SONG_TRADER_STATE_DIR", "").strip()
    if explicit:
        return Path(explicit)

    railway_volume = os.getenv("RAILWAY_VOLUME_MOUNT_PATH", "").strip()
    if railway_volume:
        return Path(railway_volume) / "song_trader_v2"

    return Path("/tmp/song_trader_v2")


STATE_DIR = _resolve_state_dir()
REPLAY_DIR = STATE_DIR / "replays"
REPLAY_DIR.mkdir(parents=True, exist_ok=True)


# A/B/C 비교 중 같은 날짜/같은 유니버스의 1분봉을 반복 다운로드하지 않도록
# 프로세스 메모리에 캐시한다. 재배포 시 자동 초기화된다.
_INTRADAY_CACHE: dict[tuple[str, tuple[str, ...]], pd.DataFrame] = {}


DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "AMD", "GOOGL",
    "AVGO", "NFLX", "PLTR", "MU", "INTC", "SMCI", "ARM", "COIN",
    "HOOD", "SOFI", "MSTR", "RBLX", "UBER", "CRWD", "PANW", "QCOM",
    "AMAT", "TSM", "MRVL", "LLY", "JPM", "BAC",
]


@dataclass
class ReplayTradeConfig:
    # 현재 Worker / AutoConfig 기본값과 동일하게 맞춤
    us_daily_budget_usd: float = 5000.0
    us_per_stock_budget_usd: float = 1500.0

    max_positions: int = 3
    max_daily_orders: int = 12

    min_score: float = 50.0
    stop_loss_pct: float = 3.0
    take1_pct: float = 3.0
    take2_pct: float = 5.0

    us_last_entry_time: str = "15:30"
    us_force_exit_time: str = "15:50"

    us_add2_trigger_pct: float = 0.40

    # BUY2 실험용 설정
    buy2_enabled: bool = True
    buy2_min_hold_minutes: float = 0.0
    buy2_max_rank: int = 5
    buy2_min_score: float = 50.0
    buy2_require_recent5_positive: bool = False
    buy2_require_recent10_positive: bool = False
    buy2_require_relative_strength_positive: bool = False

    # BUY1 시장상황 필터 실험용
    # NONE / BALANCED / STRICT
    buy1_market_filter_mode: str = "NONE"

    # BUY1 연속신호 확인 실험용
    buy1_confirmation_scans: int = 1
    buy1_confirmation_max_rank: int = 5
    buy1_require_relative_strength_hold: bool = False
    buy1_relative_strength_tolerance: float = 0.10

    # 초기 진입 크기 실험용
    # A: 50/0, B: 25/25, C: 33/17
    entry_initial_pct: float = 50.0
    entry_confirm_pct: float = 0.0
    entry_confirm_max_wait_minutes: float = 3.0

    # 조기청산 A/B/C 실험용
    # NONE / WEAK60 / WEAK60_STAGNANT
    early_exit_mode: str = "NONE"
    early_exit_min_hold_minutes: float = 60.0
    early_exit_loss_threshold_pct: float = -0.50
    early_exit_weak_points: int = 5

    # 정체청산(C) 조건
    stagnant_exit_pnl_max_pct: float = 0.20
    stagnant_exit_score_max: float = 50.0
    stagnant_exit_momentum_abs_max: float = 0.05

    us_profit_guard_trigger_pct: float = 1.20
    us_profit_guard_drawdown_pct: float = 0.80

    us_buy_limit_buffer_pct: float = 0.15
    us_sell_limit_buffer_pct: float = 0.15

    buy1_pct: int = 50
    buy2_pct: int = 50

    allow_single_share_over_stage_budget: bool = True

    # 실제 Worker cadence
    loop_seconds: int = 45
    rescan_seconds: int = 90

    # 리플레이 체결 가정:
    # 시장성 지정가가 즉시 체결된다고 보고,
    # 보수적으로 BUY는 상향 지정가 / SELL은 하향 지정가를 체결가로 사용.
    conservative_fill: bool = True


def _clock_seconds(hhmm: str) -> int:
    h, m = [int(x) for x in str(hhmm).split(":")]
    return h * 3600 + m * 60


def _seconds_of_day(ts: pd.Timestamp) -> int:
    return ts.hour * 3600 + ts.minute * 60 + ts.second


def _normalize_symbols(symbols: Iterable[str] | None) -> list[str]:
    seen = set()
    out = []

    for raw in symbols or DEFAULT_UNIVERSE:
        symbol = str(raw).strip().upper()
        if symbol and symbol not in seen:
            seen.add(symbol)
            out.append(symbol)

    return out


def _download_intraday(date_text: str, symbols: list[str]) -> pd.DataFrame:
    """
    목표일 + 이전 거래일을 같이 받아 전일 종가를 복원한다.
    A/B/C 비교에서는 동일 날짜 데이터를 메모리 캐시에서 재사용한다.
    """
    cache_key = (
        str(date_text),
        tuple(str(x).strip().upper() for x in symbols),
    )
    cached = _INTRADAY_CACHE.get(cache_key)
    if cached is not None and not cached.empty:
        return cached
    try:
        import yfinance as yf
    except Exception as e:
        raise RuntimeError(f"yfinance import 실패: {e}") from e

    day = pd.Timestamp(date_text)
    start_day = day - pd.Timedelta(days=5)
    next_day = day + pd.Timedelta(days=1)

    download_symbols = list(symbols)
    if BENCHMARK_SYMBOL not in download_symbols:
        download_symbols.append(BENCHMARK_SYMBOL)

    try:
        batch = yf.download(
            tickers=download_symbols,
            start=start_day.strftime("%Y-%m-%d"),
            end=next_day.strftime("%Y-%m-%d"),
            interval="1m",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
            prepost=False,
            timeout=25,
        )
    except TypeError:
        batch = yf.download(
            tickers=download_symbols,
            start=start_day.strftime("%Y-%m-%d"),
            end=next_day.strftime("%Y-%m-%d"),
            interval="1m",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
            prepost=False,
        )
    except Exception as e:
        raise RuntimeError(
            f"{date_text} 미국 1분봉 다운로드 실패: {type(e).__name__}: {e}"
        ) from e

    if batch is None or batch.empty:
        raise RuntimeError(f"{date_text} 미국 1분봉 결과가 비어 있음")

    _INTRADAY_CACHE[cache_key] = batch
    return batch


def _ensure_et_index(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()

    if not isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index)

    if d.index.tz is None:
        d.index = d.index.tz_localize("UTC").tz_convert(ET)
    else:
        d.index = d.index.tz_convert(ET)

    return d.sort_index()


def _slice_until(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    d = _ensure_et_index(frame)
    return d[d.index <= cutoff].copy()


def _latest_close(frame: pd.DataFrame, cutoff: pd.Timestamp) -> float:
    sliced = _slice_until(frame, cutoff)
    if sliced.empty:
        return 0.0

    columns = {str(c).lower(): c for c in sliced.columns}
    close_col = columns.get("close")
    if close_col is None:
        return 0.0

    try:
        return float(pd.to_numeric(sliced[close_col], errors="coerce").dropna().iloc[-1])
    except Exception:
        return 0.0


def _limit_price(config: ReplayTradeConfig, side: str, reference: float) -> float:
    if reference <= 0:
        return 0.0

    if side == "BUY":
        price = reference * (1.0 + config.us_buy_limit_buffer_pct / 100.0)
    else:
        price = reference * (1.0 - config.us_sell_limit_buffer_pct / 100.0)

    return round(max(0.01, price), 2)


def _fill_price(config: ReplayTradeConfig, side: str, reference: float) -> float:
    if config.conservative_fill:
        return _limit_price(config, side, reference)
    return round(reference, 2)


def _stage_budget(config: ReplayTradeConfig, stage: int) -> float:
    weights = [int(config.buy1_pct), int(config.buy2_pct)]
    total = max(1, sum(weights))
    idx = max(0, min(1, int(stage) - 1))
    return float(config.us_per_stock_budget_usd) * weights[idx] / total


def _initial_entry_budget(config: ReplayTradeConfig) -> float:
    pct = max(0.0, min(100.0, float(config.entry_initial_pct)))
    return float(config.us_per_stock_budget_usd) * pct / 100.0


def _confirm_cumulative_budget(config: ReplayTradeConfig) -> float:
    pct = max(
        0.0,
        min(
            100.0,
            float(config.entry_initial_pct)
            + float(config.entry_confirm_pct),
        ),
    )
    return float(config.us_per_stock_budget_usd) * pct / 100.0


def _first_entry_qty(config: ReplayTradeConfig, reference: float) -> int:
    """
    초기 선진입 수량.
    고가 종목은 소수점 주식이 불가능하므로,
    설정 비중 예산보다 1주 가격이 크더라도 종목 총예산 안이면 1주 허용.
    """
    limit_price = _limit_price(config, "BUY", reference)
    if limit_price <= 0:
        return 0

    budget = _initial_entry_budget(config)
    qty = int(budget // limit_price)

    if qty > 0:
        return qty

    if (
        config.allow_single_share_over_stage_budget
        and limit_price <= float(config.us_per_stock_budget_usd)
    ):
        return 1

    return 0


def _confirm_entry_qty(
    config: ReplayTradeConfig,
    reference: float,
    invested_cost: float,
) -> int:
    """
    다음 스캔 확인 후 보강 수량.
    확인보강은 누적 50% 목표를 넘지 않도록 보수적으로 계산한다.
    """
    if float(config.entry_confirm_pct) <= 0:
        return 0

    limit_price = _limit_price(config, "BUY", reference)
    if limit_price <= 0:
        return 0

    target = _confirm_cumulative_budget(config)
    remaining = max(0.0, target - float(invested_cost))

    return int(remaining // limit_price)


def _second_entry_qty(config: ReplayTradeConfig, reference: float) -> int:
    limit_price = _limit_price(config, "BUY", reference)
    if limit_price <= 0:
        return 0
    return int(_stage_budget(config, 2) // limit_price)


def _qqq_vwap_gap(frame: pd.DataFrame) -> float:
    """
    QQQ 현재가가 장중 VWAP 대비 몇 % 위/아래인지 계산.
    """
    if frame is None or frame.empty:
        return 0.0

    d = frame.copy()
    columns = {str(c).lower(): c for c in d.columns}

    required = ["high", "low", "close", "volume"]
    if not all(k in columns for k in required):
        return 0.0

    high = pd.to_numeric(d[columns["high"]], errors="coerce")
    low = pd.to_numeric(d[columns["low"]], errors="coerce")
    close = pd.to_numeric(d[columns["close"]], errors="coerce")
    volume = pd.to_numeric(d[columns["volume"]], errors="coerce").fillna(0.0)

    valid = close.dropna()
    if valid.empty:
        return 0.0

    last = float(valid.iloc[-1])
    typical = (high + low + close) / 3.0
    total_volume = float(volume.sum())

    if total_volume > 0:
        vwap = float((typical * volume).sum() / total_volume)
    else:
        vwap = float(valid.mean())

    if vwap <= 0:
        return 0.0

    return (last / vwap - 1.0) * 100.0


def _market_entry_allowed(
    config: ReplayTradeConfig,
    market_state: dict,
) -> tuple[bool, str]:
    """
    BUY1 시장필터 A/B/C 실험.

    NONE:
      현재와 동일. 시장상황으로 차단하지 않음.

    BALANCED:
      확실한 risk-off 구간만 차단.
      - QQQ 당일 -0.6% 이하
      - 5분/10분 모두 음수
      - VWAP 아래
      또는
      - QQQ가 음수이고 10분 흐름도 음수인데 매수후보가 1개 이하

    STRICT:
      시장 확인을 더 강하게 요구.
      - QQQ VWAP -0.15% 이상
      - QQQ 10분 -0.10% 이상
      - QQQ 당일 -0.30% 이상 또는 5/10분 모두 양수
      - 전체 유니버스 매수후보 2개 이상
    """
    mode = str(config.buy1_market_filter_mode or "NONE").upper()

    qqq_day = float(market_state.get("qqq_day", 0.0) or 0.0)
    qqq_5 = float(market_state.get("qqq_5", 0.0) or 0.0)
    qqq_10 = float(market_state.get("qqq_10", 0.0) or 0.0)
    qqq_vwap_gap = float(market_state.get("qqq_vwap_gap", 0.0) or 0.0)
    green_count = int(market_state.get("green_count", 0) or 0)

    detail = (
        f"QQQ {qqq_day:+.2f}% · "
        f"5분 {qqq_5:+.2f}% · 10분 {qqq_10:+.2f}% · "
        f"VWAP {qqq_vwap_gap:+.2f}% · 후보 {green_count}개"
    )

    if mode == "NONE":
        return True, f"시장필터 없음 · {detail}"

    if mode == "BALANCED":
        hard_risk_off = bool(
            qqq_day <= -0.60
            and qqq_5 < 0.0
            and qqq_10 < 0.0
            and qqq_vwap_gap < 0.0
        )
        thin_weak_market = bool(
            qqq_day < -0.20
            and qqq_10 < 0.0
            and green_count <= 1
        )

        allowed = not (hard_risk_off or thin_weak_market)
        reason = (
            "BALANCED 통과"
            if allowed
            else "BALANCED 신규진입 차단"
        )
        return allowed, f"{reason} · {detail}"

    if mode == "STRICT":
        allowed = bool(
            qqq_vwap_gap >= -0.15
            and qqq_10 >= -0.10
            and (
                qqq_day >= -0.30
                or (qqq_5 > 0.0 and qqq_10 > 0.0)
            )
            and green_count >= 2
        )
        reason = (
            "STRICT 통과"
            if allowed
            else "STRICT 신규진입 차단"
        )
        return allowed, f"{reason} · {detail}"

    return True, f"알 수 없는 시장필터({mode}) → 차단 안 함 · {detail}"


def _early_exit_state(
    config: ReplayTradeConfig,
    row,
    pnl_pct: float,
    hold_minutes: float,
) -> dict:
    """
    60분 이후 조기청산 실험용 상태 계산.
    데이터가 없으면 안전하게 조기청산하지 않는다.
    """
    mode = str(config.early_exit_mode or "NONE").upper()

    if mode == "NONE" or row is None:
        return {
            "trigger": False,
            "action": "",
            "reason": "",
            "weak_points": 0,
        }

    if hold_minutes < float(config.early_exit_min_hold_minutes):
        return {
            "trigger": False,
            "action": "",
            "reason": "",
            "weak_points": 0,
        }

    signal = str(row.get("판정", ""))
    rank = int(row.get("순위", 999) or 999)
    score = float(row.get("종합점수", 0) or 0)
    vwap_gap = float(row.get("VWAP괴리율", 0) or 0)
    ret5 = float(row.get("최근5분수익률", 0) or 0)
    ret10 = float(row.get("최근10분수익률", 0) or 0)
    rel = float(row.get("상대강도", 0) or 0)

    points = 0
    reasons = []

    if pnl_pct < 0:
        points += 1
        reasons.append("손실중")
    if "매수 후보" not in signal:
        points += 1
        reasons.append("매수후보탈락")
    if rank > 5:
        points += 1
        reasons.append("TOP5탈락")
    if score < 62.0:
        points += 1
        reasons.append("점수62미만")
    if vwap_gap < 0:
        points += 1
        reasons.append("VWAP아래")
    if ret5 < 0:
        points += 1
        reasons.append("5분음수")
    if ret10 < 0:
        points += 1
        reasons.append("10분음수")
    if rel < 0:
        points += 1
        reasons.append("상대강도음수")

    strong_weak = bool(
        pnl_pct <= float(config.early_exit_loss_threshold_pct)
        and points >= int(config.early_exit_weak_points)
    )

    if strong_weak:
        return {
            "trigger": True,
            "action": "EARLY_EXIT_WEAK60",
            "reason": (
                f"60분 강한약화 · 보유 {hold_minutes:.1f}분 · "
                f"손익 {pnl_pct:.2f}% · 약화 {points}점 · "
                + " / ".join(reasons)
            ),
            "weak_points": points,
        }

    if mode == "WEAK60_STAGNANT":
        stagnant = bool(
            pnl_pct <= float(config.stagnant_exit_pnl_max_pct)
            and "매수 후보" not in signal
            and score < float(config.stagnant_exit_score_max)
            and abs(ret5) <= float(config.stagnant_exit_momentum_abs_max)
            and abs(ret10) <= float(config.stagnant_exit_momentum_abs_max)
        )

        if stagnant:
            return {
                "trigger": True,
                "action": "EARLY_EXIT_STAGNANT60",
                "reason": (
                    f"60분 정체청산 · 보유 {hold_minutes:.1f}분 · "
                    f"손익 {pnl_pct:.2f}% · 점수 {score:.1f} · "
                    f"5분 {ret5:+.2f}% · 10분 {ret10:+.2f}% · "
                    f"판정 {signal}"
                ),
                "weak_points": points,
            }

    return {
        "trigger": False,
        "action": "",
        "reason": "",
        "weak_points": points,
    }


def _rank_snapshot(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    df["_green"] = (
        df["판정"]
        .astype(str)
        .str.contains("매수 후보", na=False)
        .astype(int)
    )

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
    df["순위"] = [i + 1 for i in range(len(df))]

    return df


def run_trade_replay(
    date_text: str = "2026-08-14",
    symbols: Iterable[str] | None = None,
    config: ReplayTradeConfig | None = None,
) -> dict:
    """
    실제 미국 Worker 흐름을 모사한 주문 없는 시뮬레이션.

    모사 대상:
    - 90초 TOP5 재스캔
    - 45초 보유종목 관리
    - TOP5 중 매수후보 신규 진입
    - 최대 3종목
    - 1차/2차 50:50
    - 2차매수 +0.40% + 최신 모멘텀
    - 손절 -3%
    - 1차 익절 +3%, 2차 익절 +5%
    - 수익보호 +1.2% 이후 고점 대비 -0.8%p
    - 15:30 ET 신규매수 종료
    - 15:50 ET 전량청산

    실제 KIS 주문은 절대 호출하지 않는다.
    """
    config = config or ReplayTradeConfig()
    symbols = _normalize_symbols(symbols)

    batch = _download_intraday(date_text, symbols)
    universe_size = len(symbols) + (0 if BENCHMARK_SYMBOL in symbols else 1)

    benchmark_full = _extract_symbol_frame(
        batch,
        BENCHMARK_SYMBOL,
        universe_size,
    )
    if benchmark_full is None or benchmark_full.empty:
        raise RuntimeError("QQQ 1분봉을 찾지 못했습니다.")

    qqq_session, qqq_prev_close, qqq_prev_ok = _split_session_for_date(
        benchmark_full,
        date_text,
    )
    if qqq_session is None or qqq_session.empty:
        raise RuntimeError("QQQ 목표일 정규장 데이터를 찾지 못했습니다.")

    symbol_data = {}

    for symbol in symbols:
        full = _extract_symbol_frame(batch, symbol, universe_size)
        if full is None or full.empty:
            continue

        session, prev_close, prev_ok = _split_session_for_date(
            full,
            date_text,
        )

        if session is None or session.empty:
            continue

        symbol_data[symbol] = {
            "session": session,
            "prev_close": float(prev_close or 0.0),
            "prev_ok": bool(prev_ok),
        }

    if not symbol_data:
        raise RuntimeError("리플레이 가능한 종목 데이터가 없습니다.")

    date_ts = pd.Timestamp(date_text, tz=ET)

    # 실제 정규장 09:30 ~ 16:00
    now = date_ts.replace(hour=9, minute=30, second=0)
    end = date_ts.replace(hour=15, minute=59, second=59)

    last_entry_sec = _clock_seconds(config.us_last_entry_time)
    force_exit_sec = _clock_seconds(config.us_force_exit_time)

    next_scan = now

    positions: dict[str, dict] = {}
    latest_top5 = pd.DataFrame()
    latest_ranked = pd.DataFrame()
    latest_market_state = {
        "qqq_day": 0.0,
        "qqq_5": 0.0,
        "qqq_10": 0.0,
        "qqq_vwap_gap": 0.0,
        "green_count": 0,
    }

    # 90초 스캔 단위로 종목별 매수후보 지속성을 추적한다.
    candidate_streaks: dict[str, int] = {}
    candidate_prev_rel: dict[str, float] = {}
    candidate_rel_holds: dict[str, bool] = {}

    daily_buy_amount = 0.0
    daily_orders = 0

    events: list[dict] = []
    scan_rows: list[dict] = []

    realized_pnl = 0.0
    gross_buy_value = 0.0
    gross_sell_value = 0.0

    def add_event(
        *,
        ts: pd.Timestamp,
        symbol: str,
        action: str,
        side: str,
        qty: int,
        ref_price: float,
        fill_price: float,
        reason: str,
        pnl_pct: float | None = None,
        score: float | None = None,
        rank: int | None = None,
        realized: float = 0.0,
    ) -> None:
        nonlocal realized_pnl, gross_buy_value, gross_sell_value, daily_orders

        value = float(fill_price) * int(qty)

        if side == "BUY":
            gross_buy_value += value
        else:
            gross_sell_value += value
            realized_pnl += float(realized)

        daily_orders += 1

        events.append({
            "시간ET": ts.isoformat(),
            "종목코드": symbol,
            "액션": action,
            "구분": side,
            "수량": int(qty),
            "기준가": round(float(ref_price), 4),
            "가정체결가": round(float(fill_price), 4),
            "주문금액": round(value, 2),
            "손익률": "" if pnl_pct is None else round(float(pnl_pct), 3),
            "실현손익USD": round(float(realized), 2),
            "종합점수": "" if score is None else round(float(score), 1),
            "TOP5순위": "" if rank is None else int(rank),
            "이유": reason,
        })

    while now <= end:
        # --------------------------------------------------------
        # 1) 90초마다 전체 유니버스 스캔 -> TOP5
        # --------------------------------------------------------
        if now >= next_scan and _seconds_of_day(now) < force_exit_sec:
            qqq_slice = _slice_until(qqq_session, now)
            benchmark = _benchmark_metrics(
                qqq_slice,
                prev_close=qqq_prev_close,
                prev_close_available=qqq_prev_ok,
            )

            scored_rows = []

            for symbol, meta in symbol_data.items():
                sliced = _slice_until(meta["session"], now)

                scored = _score_frame(
                    symbol,
                    sliced,
                    benchmark=benchmark,
                    prev_close=meta["prev_close"],
                    prev_close_available=meta["prev_ok"],
                )

                if scored:
                    scored_rows.append(scored)

            ranked = _rank_snapshot(scored_rows)
            latest_ranked = ranked.copy() if not ranked.empty else pd.DataFrame()
            latest_top5 = ranked.head(5).copy() if not ranked.empty else pd.DataFrame()

            green_count = 0
            if not ranked.empty:
                green_count = int(
                    ranked["판정"]
                    .astype(str)
                    .str.contains("매수 후보", na=False)
                    .sum()
                )

            latest_market_state = {
                "qqq_day": float(benchmark.get("day_ret", 0.0) or 0.0),
                "qqq_5": float(benchmark.get("ret5", 0.0) or 0.0),
                "qqq_10": float(benchmark.get("ret10", 0.0) or 0.0),
                "qqq_vwap_gap": float(_qqq_vwap_gap(qqq_slice)),
                "green_count": green_count,
            }

            # 이번 스캔에서의 매수후보 연속성 갱신.
            current_green: dict[str, dict] = {}
            if not ranked.empty:
                for _, candidate_row in ranked.iterrows():
                    cs = str(candidate_row.get("종목코드", "")).upper()
                    if (
                        cs
                        and "매수 후보" in str(candidate_row.get("판정", ""))
                        and int(candidate_row.get("순위", 999) or 999) <= 5
                    ):
                        current_green[cs] = candidate_row.to_dict()

            tracked_symbols = set(candidate_streaks) | set(current_green)

            for cs in tracked_symbols:
                row_now = current_green.get(cs)

                if row_now is None:
                    candidate_streaks[cs] = 0
                    candidate_rel_holds[cs] = False
                    candidate_prev_rel.pop(cs, None)
                    continue

                rel_now = float(row_now.get("상대강도", 0.0) or 0.0)
                prev_rel = candidate_prev_rel.get(cs)

                if int(candidate_streaks.get(cs, 0)) > 0 and prev_rel is not None:
                    candidate_rel_holds[cs] = bool(
                        rel_now
                        >= float(prev_rel)
                        - float(config.buy1_relative_strength_tolerance)
                    )
                else:
                    candidate_rel_holds[cs] = True

                candidate_streaks[cs] = int(candidate_streaks.get(cs, 0)) + 1
                candidate_prev_rel[cs] = rel_now

            if not latest_top5.empty:
                for _, row in latest_top5.iterrows():
                    scan_rows.append({
                        "시간ET": now.isoformat(),
                        "순위": int(row.get("순위", 0) or 0),
                        "종목코드": str(row.get("종목코드", "")),
                        "판정": str(row.get("판정", "")),
                        "종합점수": float(row.get("종합점수", 0) or 0),
                        "당일등락률": float(row.get("당일등락률", 0) or 0),
                        "상대강도": float(row.get("상대강도", 0) or 0),
                        "최근5분수익률": float(row.get("최근5분수익률", 0) or 0),
                        "최근10분수익률": float(row.get("최근10분수익률", 0) or 0),
                        "거래량배수": float(row.get("거래량배수", 0) or 0),
                        "급락반등함정": bool(row.get("급락반등함정", False)),
                        "대장주자격": bool(row.get("대장주자격", False)),
                    })

            next_scan = now + pd.Timedelta(seconds=int(config.rescan_seconds))

        top5_map = {}
        if latest_top5 is not None and not latest_top5.empty:
            for _, row in latest_top5.iterrows():
                top5_map[str(row.get("종목코드", "")).upper()] = row

        ranked_map = {}
        if latest_ranked is not None and not latest_ranked.empty:
            for _, row in latest_ranked.iterrows():
                ranked_map[str(row.get("종목코드", "")).upper()] = row

        # --------------------------------------------------------
        # 2) 보유 종목 관리 - 실제 Worker처럼 신규매수보다 먼저
        # --------------------------------------------------------
        for symbol in list(positions.keys()):
            pos = positions.get(symbol)
            if pos is None:
                continue

            meta = symbol_data.get(symbol)
            if not meta:
                continue

            price = _latest_close(meta["session"], now)
            if price <= 0:
                continue

            qty = int(pos["qty"])
            avg = float(pos["avg_price"])

            if qty <= 0 or avg <= 0:
                positions.pop(symbol, None)
                continue

            pnl = (price / avg - 1.0) * 100.0

            peak = max(float(pos.get("peak_pnl", pnl)), pnl)
            pos["peak_pnl"] = peak
            drawdown = max(0.0, peak - pnl)

            # 15:50 ET 강제청산 최우선
            if _seconds_of_day(now) >= force_exit_sec:
                fill = _fill_price(config, "SELL", price)
                realized = (fill - avg) * qty

                add_event(
                    ts=now,
                    symbol=symbol,
                    action="FORCE_SELL",
                    side="SELL",
                    qty=qty,
                    ref_price=price,
                    fill_price=fill,
                    reason=f"당일 강제청산 {config.us_force_exit_time} ET",
                    pnl_pct=pnl,
                    realized=realized,
                )

                positions.pop(symbol, None)
                continue

            # 손절
            if pnl <= -abs(float(config.stop_loss_pct)):
                fill = _fill_price(config, "SELL", price)
                realized = (fill - avg) * qty

                add_event(
                    ts=now,
                    symbol=symbol,
                    action="STOP_LOSS",
                    side="SELL",
                    qty=qty,
                    ref_price=price,
                    fill_price=fill,
                    reason=f"손절 {pnl:.2f}%",
                    pnl_pct=pnl,
                    realized=realized,
                )

                positions.pop(symbol, None)
                continue

            # ----------------------------------------------------
            # 60분 조기청산 실험
            # 기존 -3% 손절을 먼저 적용한 뒤, 아직 살아 있는 포지션만 평가.
            # ----------------------------------------------------
            try:
                opened_at_for_exit = pd.Timestamp(
                    str(pos.get("opened_at", "") or "")
                )
                hold_minutes_for_exit = (
                    now - opened_at_for_exit
                ).total_seconds() / 60.0
            except Exception:
                hold_minutes_for_exit = 0.0

            early_row = ranked_map.get(symbol)
            early_state = _early_exit_state(
                config,
                early_row,
                pnl,
                hold_minutes_for_exit,
            )

            if bool(early_state.get("trigger")):
                fill = _fill_price(config, "SELL", price)
                realized = (fill - avg) * qty

                add_event(
                    ts=now,
                    symbol=symbol,
                    action=str(early_state.get("action", "EARLY_EXIT")),
                    side="SELL",
                    qty=qty,
                    ref_price=price,
                    fill_price=fill,
                    reason=str(early_state.get("reason", "60분 조기청산")),
                    pnl_pct=pnl,
                    realized=realized,
                )

                positions.pop(symbol, None)
                continue

            # 1차 익절
            if pnl >= float(config.take1_pct) and not bool(pos.get("take1_sent")):
                sell_qty = max(1, qty // 2)
                fill = _fill_price(config, "SELL", price)
                realized = (fill - avg) * sell_qty

                add_event(
                    ts=now,
                    symbol=symbol,
                    action="TAKE1",
                    side="SELL",
                    qty=sell_qty,
                    ref_price=price,
                    fill_price=fill,
                    reason=f"1차 익절 {pnl:.2f}% · 약 50%",
                    pnl_pct=pnl,
                    realized=realized,
                )

                pos["qty"] = qty - sell_qty
                pos["take1_sent"] = True

                if pos["qty"] <= 0:
                    positions.pop(symbol, None)

                continue

            # 2차 익절
            if pnl >= float(config.take2_pct) and bool(pos.get("take1_sent")):
                fill = _fill_price(config, "SELL", price)
                realized = (fill - avg) * qty

                add_event(
                    ts=now,
                    symbol=symbol,
                    action="TAKE2",
                    side="SELL",
                    qty=qty,
                    ref_price=price,
                    fill_price=fill,
                    reason=f"2차 익절 {pnl:.2f}% · 잔여 전량",
                    pnl_pct=pnl,
                    realized=realized,
                )

                positions.pop(symbol, None)
                continue

            # 수익보호
            if (
                peak >= float(config.us_profit_guard_trigger_pct)
                and drawdown >= float(config.us_profit_guard_drawdown_pct)
            ):
                if not bool(pos.get("take1_sent")):
                    sell_qty = max(1, qty // 2)
                    fill = _fill_price(config, "SELL", price)
                    realized = (fill - avg) * sell_qty

                    add_event(
                        ts=now,
                        symbol=symbol,
                        action="PROFIT_GUARD1",
                        side="SELL",
                        qty=sell_qty,
                        ref_price=price,
                        fill_price=fill,
                        reason=(
                            f"수익보호 1차 · 최고 +{peak:.2f}% → 현재 {pnl:.2f}% "
                            f"({drawdown:.2f}%p 되밀림)"
                        ),
                        pnl_pct=pnl,
                        realized=realized,
                    )

                    pos["qty"] = qty - sell_qty
                    pos["take1_sent"] = True

                    if pos["qty"] <= 0:
                        positions.pop(symbol, None)

                    continue
                else:
                    fill = _fill_price(config, "SELL", price)
                    realized = (fill - avg) * qty

                    add_event(
                        ts=now,
                        symbol=symbol,
                        action="PROFIT_GUARD2",
                        side="SELL",
                        qty=qty,
                        ref_price=price,
                        fill_price=fill,
                        reason=(
                            f"수익보호 2차 · 최고 +{peak:.2f}% → 현재 {pnl:.2f}% "
                            f"({drawdown:.2f}%p 되밀림)"
                        ),
                        pnl_pct=pnl,
                        realized=realized,
                    )

                    positions.pop(symbol, None)
                    continue

            # 신규/추가매수 종료 이후에는 확인보강/BUY2 모두 금지
            if _seconds_of_day(now) >= last_entry_sec:
                continue

            # ----------------------------------------------------
            # 초기 선진입 확인보강:
            # 첫 매수 후 '새로운 다음 스캔'에서도 매수후보가 유지될 때만
            # 25% 또는 17%를 보강한다.
            # ----------------------------------------------------
            if (
                float(config.entry_confirm_pct) > 0.0
                and not bool(pos.get("entry_confirm_done", False))
                and not bool(pos.get("entry_confirm_expired", False))
                and daily_orders < int(config.max_daily_orders)
            ):
                opened_at = str(pos.get("opened_at", "") or "")
                try:
                    hold_minutes_for_confirm = (
                        now - pd.Timestamp(opened_at)
                    ).total_seconds() / 60.0
                except Exception:
                    hold_minutes_for_confirm = 0.0

                if (
                    hold_minutes_for_confirm
                    > float(config.entry_confirm_max_wait_minutes)
                ):
                    pos["entry_confirm_expired"] = True
                else:
                    row_confirm = top5_map.get(symbol)
                    streak_now = int(candidate_streaks.get(symbol, 0))
                    streak_at_open = int(pos.get("entry_streak_at_open", 0))

                    confirm_signal_ok = bool(
                        row_confirm is not None
                        and "매수 후보" in str(row_confirm.get("판정", ""))
                        and not bool(row_confirm.get("모멘텀약화", False))
                        and float(row_confirm.get("종합점수", 0) or 0)
                        >= float(config.min_score)
                        and streak_now > streak_at_open
                    )

                    if confirm_signal_ok:
                        invested_cost = float(
                            pos.get(
                                "invested_cost",
                                float(pos["avg_price"]) * int(pos["qty"]),
                            )
                        )
                        add_qty = _confirm_entry_qty(
                            config,
                            price,
                            invested_cost,
                        )

                        if add_qty > 0:
                            fill = _fill_price(config, "BUY", price)
                            cost = add_qty * fill

                            if (
                                invested_cost + cost
                                <= float(config.us_per_stock_budget_usd)
                                and daily_buy_amount + cost
                                <= float(config.us_daily_budget_usd)
                            ):
                                old_qty = int(pos["qty"])
                                old_avg = float(pos["avg_price"])
                                new_qty = old_qty + add_qty
                                new_avg = (
                                    old_avg * old_qty
                                    + fill * add_qty
                                ) / new_qty

                                score = float(
                                    row_confirm.get("종합점수", 0) or 0
                                )
                                rank = int(
                                    row_confirm.get("순위", 0) or 0
                                )

                                add_event(
                                    ts=now,
                                    symbol=symbol,
                                    action="BUY1_CONFIRM",
                                    side="BUY",
                                    qty=add_qty,
                                    ref_price=price,
                                    fill_price=fill,
                                    reason=(
                                        f"선진입 확인보강 · "
                                        f"초기 {config.entry_initial_pct:.0f}% + "
                                        f"확인 {config.entry_confirm_pct:.0f}% · "
                                        f"다음 스캔 매수후보 유지 · "
                                        f"점수 {score:.1f} · TOP{rank}"
                                    ),
                                    pnl_pct=pnl,
                                    score=score,
                                    rank=rank,
                                )

                                daily_buy_amount += cost
                                pos["qty"] = new_qty
                                pos["avg_price"] = new_avg
                                pos["invested_cost"] = invested_cost + cost

                        # 고가 종목 등으로 추가 1주를 살 수 없어도
                        # 다음 스캔 확인 자체는 끝난 것으로 처리한다.
                        pos["entry_confirm_done"] = True

            # 2차 매수: B_STRICT 등 실험 설정에 따라 조건 적용
            if (
                bool(config.buy2_enabled)
                and int(pos.get("buy_stage", 1)) == 1
                and (
                    float(config.entry_confirm_pct) <= 0.0
                    or bool(pos.get("entry_confirm_done", False))
                )
                and pnl >= float(config.us_add2_trigger_pct)
                and daily_orders < int(config.max_daily_orders)
            ):
                row = top5_map.get(symbol)

                opened_at = str(pos.get("opened_at", "") or "")
                try:
                    hold_minutes = (
                        now - pd.Timestamp(opened_at)
                    ).total_seconds() / 60.0
                except Exception:
                    hold_minutes = 0.0

                rank = int(row.get("순위", 999) or 999) if row is not None else 999
                score = float(row.get("종합점수", 0) or 0) if row is not None else 0.0
                recent5 = float(row.get("최근5분수익률", 0) or 0) if row is not None else 0.0
                recent10 = float(row.get("최근10분수익률", 0) or 0) if row is not None else 0.0
                relative_strength = float(row.get("상대강도", 0) or 0) if row is not None else 0.0

                momentum_ok = bool(
                    row is not None
                    and "매수 후보" in str(row.get("판정", ""))
                    and score >= float(config.buy2_min_score)
                    and rank <= int(config.buy2_max_rank)
                    and hold_minutes >= float(config.buy2_min_hold_minutes)
                    and not bool(row.get("모멘텀약화", False))
                    and (
                        (not config.buy2_require_recent5_positive)
                        or recent5 > 0.0
                    )
                    and (
                        (not config.buy2_require_recent10_positive)
                        or recent10 > 0.0
                    )
                    and (
                        (not config.buy2_require_relative_strength_positive)
                        or relative_strength > 0.0
                    )
                )

                if momentum_ok:
                    add_qty = _second_entry_qty(config, price)
                    fill = _fill_price(config, "BUY", price)
                    cost = add_qty * fill

                    if (
                        add_qty > 0
                        and daily_buy_amount + cost <= float(config.us_daily_budget_usd)
                    ):
                        old_qty = int(pos["qty"])
                        old_avg = float(pos["avg_price"])
                        new_qty = old_qty + add_qty
                        new_avg = (
                            old_avg * old_qty + fill * add_qty
                        ) / new_qty

                        add_event(
                            ts=now,
                            symbol=symbol,
                            action="BUY2",
                            side="BUY",
                            qty=add_qty,
                            ref_price=price,
                            fill_price=fill,
                            reason=(
                                f"2차 분할매수 +{pnl:.2f}% · "
                                f"보유 {hold_minutes:.1f}분 · TOP{rank} · "
                                f"점수 {score:.1f} · 5분 {recent5:+.2f}% · "
                                f"10분 {recent10:+.2f}% · 상대강도 {relative_strength:+.2f}%p"
                            ),
                            pnl_pct=pnl,
                            score=score,
                            rank=rank,
                        )

                        daily_buy_amount += cost
                        pos["qty"] = new_qty
                        pos["avg_price"] = new_avg
                        pos["invested_cost"] = float(
                            pos.get(
                                "invested_cost",
                                old_avg * old_qty,
                            )
                        ) + cost
                        pos["buy_stage"] = 2

        # --------------------------------------------------------
        # 3) 신규 진입: 최신 TOP5 중 조건 통과 종목
        # --------------------------------------------------------
        if (
            _seconds_of_day(now) < last_entry_sec
            and len(positions) < int(config.max_positions)
            and daily_orders < int(config.max_daily_orders)
            and latest_top5 is not None
            and not latest_top5.empty
        ):
            for _, row in latest_top5.iterrows():
                if len(positions) >= int(config.max_positions):
                    break
                if daily_orders >= int(config.max_daily_orders):
                    break

                symbol = str(row.get("종목코드", "")).upper()

                if not symbol or symbol in positions:
                    continue

                signal = str(row.get("판정", ""))
                combined = float(row.get("종합점수", 0) or 0)

                if "매수 후보" not in signal:
                    continue
                if combined < float(config.min_score):
                    continue
                if bool(row.get("모멘텀약화", False)):
                    continue

                market_ok, market_reason = _market_entry_allowed(
                    config,
                    latest_market_state,
                )
                if not market_ok:
                    continue

                streak = int(candidate_streaks.get(symbol, 0))
                rank_now = int(row.get("순위", 999) or 999)
                rel_hold = bool(candidate_rel_holds.get(symbol, True))

                confirmation_ok = bool(
                    streak >= int(config.buy1_confirmation_scans)
                    and rank_now <= int(config.buy1_confirmation_max_rank)
                    and (
                        (not config.buy1_require_relative_strength_hold)
                        or rel_hold
                    )
                )

                if not confirmation_ok:
                    continue

                meta = symbol_data.get(symbol)
                if not meta:
                    continue

                price = _latest_close(meta["session"], now)
                if price <= 0:
                    continue

                qty = _first_entry_qty(config, price)
                if qty <= 0:
                    continue

                fill = _fill_price(config, "BUY", price)
                cost = qty * fill

                if cost > float(config.us_per_stock_budget_usd):
                    continue

                if daily_buy_amount + cost > float(config.us_daily_budget_usd):
                    continue

                rank = int(row.get("순위", 0) or 0)

                add_event(
                    ts=now,
                    symbol=symbol,
                    action="BUY1",
                    side="BUY",
                    qty=qty,
                    ref_price=price,
                    fill_price=fill,
                    reason=(
                        f"TOP5 신규진입 {config.entry_initial_pct:.0f}% · "
                        f"점수 {combined:.1f} · "
                        f"당일 {float(row.get('당일등락률', 0) or 0):+.2f}% · "
                        f"상대강도 {float(row.get('상대강도', 0) or 0):+.2f}%p · "
                        f"연속후보 {streak}회 · TOP{rank_now} · "
                        f"상대강도유지={'YES' if rel_hold else 'NO'} · "
                        f"{market_reason}"
                    ),
                    score=combined,
                    rank=rank,
                )

                daily_buy_amount += cost

                positions[symbol] = {
                    "qty": int(qty),
                    "avg_price": float(fill),
                    "invested_cost": float(cost),
                    "buy_stage": 1,
                    "take1_sent": False,
                    "peak_pnl": 0.0,
                    "opened_at": now.isoformat(),
                    "entry_streak_at_open": int(
                        candidate_streaks.get(symbol, 0)
                    ),
                    "entry_confirm_done": bool(
                        float(config.entry_confirm_pct) <= 0.0
                    ),
                    "entry_confirm_expired": False,
                }

        now = now + pd.Timedelta(seconds=int(config.loop_seconds))

    # 만약 마지막 loop 정렬 때문에 15:50 강제청산을 놓친 잔여분이 있다면
    # 15:59 마지막 가격으로 정리 (실제 전략의 '당일 무조건 청산' 목적 보존).
    if positions:
        final_ts = date_ts.replace(hour=15, minute=59, second=0)

        for symbol in list(positions.keys()):
            pos = positions[symbol]
            meta = symbol_data[symbol]
            price = _latest_close(meta["session"], final_ts)

            if price <= 0:
                continue

            qty = int(pos["qty"])
            avg = float(pos["avg_price"])
            pnl = (price / avg - 1.0) * 100.0 if avg > 0 else 0.0
            fill = _fill_price(config, "SELL", price)
            realized = (fill - avg) * qty

            add_event(
                ts=final_ts,
                symbol=symbol,
                action="FORCE_SELL_FALLBACK",
                side="SELL",
                qty=qty,
                ref_price=price,
                fill_price=fill,
                reason="리플레이 종료 안전청산",
                pnl_pct=pnl,
                realized=realized,
            )

            positions.pop(symbol, None)

    event_df = pd.DataFrame(events)
    scan_df = pd.DataFrame(scan_rows)

    # ------------------------------------------------------------
    # 종목별 결과
    # ------------------------------------------------------------
    symbol_summary = []

    if not event_df.empty:
        traded_symbols = list(dict.fromkeys(event_df["종목코드"].tolist()))

        for symbol in traded_symbols:
            s = event_df[event_df["종목코드"] == symbol].copy()
            buys = s[s["구분"] == "BUY"]
            sells = s[s["구분"] == "SELL"]

            buy_value = float(buys["주문금액"].sum()) if not buys.empty else 0.0
            sell_value = float(sells["주문금액"].sum()) if not sells.empty else 0.0
            pnl_usd = float(sells["실현손익USD"].sum()) if not sells.empty else 0.0

            symbol_summary.append({
                "종목코드": symbol,
                "첫매수시각ET": str(buys.iloc[0]["시간ET"]) if not buys.empty else "",
                "마지막매도시각ET": str(sells.iloc[-1]["시간ET"]) if not sells.empty else "",
                "매수횟수": int(len(buys)),
                "매도횟수": int(len(sells)),
                "총매수금액USD": round(buy_value, 2),
                "총매도금액USD": round(sell_value, 2),
                "실현손익USD": round(pnl_usd, 2),
                "매수금액대비수익률": (
                    round(pnl_usd / buy_value * 100.0, 3)
                    if buy_value > 0 else 0.0
                ),
                "종료사유": str(sells.iloc[-1]["액션"]) if not sells.empty else "",
            })

    symbol_summary_df = pd.DataFrame(symbol_summary)

    if not symbol_summary_df.empty:
        symbol_summary_df = symbol_summary_df.sort_values(
            "첫매수시각ET",
            ascending=True,
        ).reset_index(drop=True)

    winning_symbols = 0
    losing_symbols = 0

    if not symbol_summary_df.empty:
        winning_symbols = int(
            (symbol_summary_df["실현손익USD"] > 0).sum()
        )
        losing_symbols = int(
            (symbol_summary_df["실현손익USD"] < 0).sum()
        )

    # 일일예산은 '누적 매수액' 제한이라 실현수익률 분모는
    # 실제 누적매수액과 설정 일일예산을 둘 다 제공한다.
    pnl_on_buys_pct = (
        realized_pnl / gross_buy_value * 100.0
        if gross_buy_value > 0 else 0.0
    )
    pnl_on_daily_budget_pct = (
        realized_pnl / float(config.us_daily_budget_usd) * 100.0
        if config.us_daily_budget_usd > 0 else 0.0
    )

    stamp = date_text.replace("-", "")
    event_path = REPLAY_DIR / f"us_trade_replay_{stamp}_events.csv"
    scan_path = REPLAY_DIR / f"us_trade_replay_{stamp}_scans.csv"
    symbol_path = REPLAY_DIR / f"us_trade_replay_{stamp}_symbols.csv"
    summary_path = REPLAY_DIR / f"us_trade_replay_{stamp}_summary.json"

    event_df.to_csv(
        event_path,
        index=False,
        encoding="utf-8-sig",
    )

    scan_df.to_csv(
        scan_path,
        index=False,
        encoding="utf-8-sig",
    )

    symbol_summary_df.to_csv(
        symbol_path,
        index=False,
        encoding="utf-8-sig",
    )

    payload = {
        "ok": True,
        "version": "trade-replay-v6-exit-abc",
        "date": date_text,
        "universe_count": len(symbols),
        "config": asdict(config),
        "assumptions": {
            "real_orders": False,
            "data": "yfinance 1-minute historical bars",
            "scan_cadence_seconds": int(config.rescan_seconds),
            "management_cadence_seconds": int(config.loop_seconds),
            "fill_model": (
                "conservative: BUY at +0.15% marketable limit, "
                "SELL at -0.15% marketable limit"
                if config.conservative_fill
                else "reference close"
            ),
            "fees_commissions": "not separately included",
            "note": (
                "KIS 실제 호가/체결순서와 1분봉 사이의 미세한 차이 때문에 "
                "실제 체결 결과와는 달라질 수 있음"
            ),
        },
        "summary": {
            "총주문횟수": int(len(event_df)),
            "매수주문횟수": int((event_df["구분"] == "BUY").sum()) if not event_df.empty else 0,
            "매도주문횟수": int((event_df["구분"] == "SELL").sum()) if not event_df.empty else 0,
            "거래종목수": int(len(symbol_summary_df)),
            "수익종목수": winning_symbols,
            "손실종목수": losing_symbols,
            "누적매수금액USD": round(gross_buy_value, 2),
            "누적매도금액USD": round(gross_sell_value, 2),
            "실현손익USD": round(realized_pnl, 2),
            "누적매수금액대비수익률": round(pnl_on_buys_pct, 3),
            "일일예산5000달러대비수익률": round(pnl_on_daily_budget_pct, 3),
        },
        "symbols": symbol_summary_df.to_dict("records"),
        "events": event_df.to_dict("records"),
        "files": {
            "events_csv": str(event_path),
            "scans_csv": str(scan_path),
            "symbols_csv": str(symbol_path),
            "summary_json": str(summary_path),
        },
    }

    summary_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    return payload


def _strategy_configs() -> dict[str, ReplayTradeConfig]:
    """
    BUY2 A/B/C 실험 설정.
    """
    return {
        "A_CURRENT": ReplayTradeConfig(
            buy2_enabled=True,
            us_add2_trigger_pct=0.40,
            buy2_min_hold_minutes=0.0,
            buy2_max_rank=5,
            buy2_min_score=50.0,
            buy2_require_recent5_positive=False,
            buy2_require_recent10_positive=False,
            buy2_require_relative_strength_positive=False,
        ),
        "B_STRICT": ReplayTradeConfig(
            buy2_enabled=True,
            us_add2_trigger_pct=0.80,
            buy2_min_hold_minutes=5.0,
            buy2_max_rank=3,
            buy2_min_score=70.0,
            buy2_require_recent5_positive=True,
            buy2_require_recent10_positive=True,
            buy2_require_relative_strength_positive=True,
        ),
        "C_NO_BUY2": ReplayTradeConfig(
            buy2_enabled=False,
            us_add2_trigger_pct=999.0,
        ),
    }


def compare_buy2_strategies(
    dates: Iterable[str] | None = None,
    symbols: Iterable[str] | None = None,
) -> dict:
    """
    여러 거래일에 대해 A/B/C BUY2 전략을 동일 조건으로 비교한다.

    V2 fast:
    - 날짜별 시장데이터는 1회만 다운로드
    - 같은 날짜에 A/B/C를 연속 실행
    - 실제 주문은 발생하지 않음
    """
    if dates is None:
        dates = [
            "2026-08-10",
            "2026-08-11",
            "2026-08-12",
            "2026-08-13",
            "2026-08-14",
        ]

    dates = [str(x).strip() for x in dates if str(x).strip()]
    symbols = _normalize_symbols(symbols)

    strategies = _strategy_configs()
    daily_rows = []

    # 날짜 우선: 한 번 받은 1분봉을 A/B/C가 즉시 재사용
    for date_text in dates:
        for strategy_name, cfg in strategies.items():
            try:
                result = run_trade_replay(
                    date_text=date_text,
                    symbols=symbols,
                    config=cfg,
                )

                summary = dict(result.get("summary", {}) or {})

                daily_rows.append({
                    "전략": strategy_name,
                    "날짜": date_text,
                    "성공": True,
                    "실현손익USD": float(summary.get("실현손익USD", 0) or 0),
                    "일일예산대비수익률": float(
                        summary.get("일일예산5000달러대비수익률", 0) or 0
                    ),
                    "거래종목수": int(summary.get("거래종목수", 0) or 0),
                    "수익종목수": int(summary.get("수익종목수", 0) or 0),
                    "손실종목수": int(summary.get("손실종목수", 0) or 0),
                    "매수주문횟수": int(summary.get("매수주문횟수", 0) or 0),
                    "매도주문횟수": int(summary.get("매도주문횟수", 0) or 0),
                    "BUY2횟수": int(
                        sum(
                            1
                            for e in result.get("events", []) or []
                            if str(e.get("액션", "")) == "BUY2"
                        )
                    ),
                    "손절횟수": int(
                        sum(
                            1
                            for e in result.get("events", []) or []
                            if str(e.get("액션", "")) == "STOP_LOSS"
                        )
                    ),
                })

            except Exception as e:
                daily_rows.append({
                    "전략": strategy_name,
                    "날짜": date_text,
                    "성공": False,
                    "오류": f"{type(e).__name__}: {e}",
                })

    daily_df = pd.DataFrame(daily_rows)
    aggregate_rows = []

    for strategy_name in strategies:
        s = daily_df[
            (daily_df["전략"] == strategy_name)
            & (daily_df["성공"] == True)
        ].copy()

        if s.empty:
            aggregate_rows.append({
                "전략": strategy_name,
                "성공일수": 0,
                "총실현손익USD": 0.0,
                "평균일손익USD": 0.0,
                "수익일수": 0,
                "손실일수": 0,
                "일승률": 0.0,
                "최악의날손익USD": 0.0,
                "최고의날손익USD": 0.0,
                "총BUY2횟수": 0,
                "총손절횟수": 0,
                "종목승률": 0.0,
            })
            continue

        total_profit_symbols = int(s["수익종목수"].sum())
        total_loss_symbols = int(s["손실종목수"].sum())
        closed_symbols = total_profit_symbols + total_loss_symbols

        aggregate_rows.append({
            "전략": strategy_name,
            "성공일수": int(len(s)),
            "총실현손익USD": round(float(s["실현손익USD"].sum()), 2),
            "평균일손익USD": round(float(s["실현손익USD"].mean()), 2),
            "수익일수": int((s["실현손익USD"] > 0).sum()),
            "손실일수": int((s["실현손익USD"] < 0).sum()),
            "일승률": round(
                float((s["실현손익USD"] > 0).mean() * 100.0),
                1,
            ),
            "최악의날손익USD": round(float(s["실현손익USD"].min()), 2),
            "최고의날손익USD": round(float(s["실현손익USD"].max()), 2),
            "총BUY2횟수": int(s["BUY2횟수"].sum()),
            "총손절횟수": int(s["손절횟수"].sum()),
            "종목승률": round(
                total_profit_symbols / closed_symbols * 100.0
                if closed_symbols > 0 else 0.0,
                1,
            ),
        })

    aggregate_df = pd.DataFrame(aggregate_rows)

    if not aggregate_df.empty:
        ranked = aggregate_df.sort_values(
            ["총실현손익USD", "최악의날손익USD", "종목승률"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        recommended = str(ranked.iloc[0]["전략"])
    else:
        recommended = ""

    stamp = datetime.now(ET).strftime("%Y%m%d_%H%M%S")
    daily_path = REPLAY_DIR / f"buy2_abc_daily_{stamp}.csv"
    aggregate_path = REPLAY_DIR / f"buy2_abc_summary_{stamp}.csv"
    json_path = REPLAY_DIR / f"buy2_abc_{stamp}.json"

    daily_df.to_csv(daily_path, index=False, encoding="utf-8-sig")
    aggregate_df.to_csv(aggregate_path, index=False, encoding="utf-8-sig")

    payload = {
        "ok": True,
        "version": "buy2-abc-v2-fast",
        "dates": dates,
        "universe_count": len(symbols),
        "aggregate": aggregate_df.to_dict("records"),
        "daily": daily_df.to_dict("records"),
        "recommended_by_replay": recommended,
        "warning": (
            "5거래일 리플레이 결과만으로 실제 전략을 확정하지 말고 "
            "더 많은 날짜로 추가 검증하는 것이 안전함"
        ),
        "files": {
            "daily_csv": str(daily_path),
            "aggregate_csv": str(aggregate_path),
            "json": str(json_path),
        },
    }

    json_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    return payload


def _entry_market_strategy_configs() -> dict[str, ReplayTradeConfig]:
    """
    BUY2는 기존 5일 검증에서 가장 유력했던 B_STRICT로 고정하고,
    BUY1 시장필터만 A/B/C로 비교한다.
    """
    common = dict(
        buy2_enabled=True,
        us_add2_trigger_pct=0.80,
        buy2_min_hold_minutes=5.0,
        buy2_max_rank=3,
        buy2_min_score=70.0,
        buy2_require_recent5_positive=True,
        buy2_require_recent10_positive=True,
        buy2_require_relative_strength_positive=True,
    )

    return {
        "A_ENTRY_CURRENT": ReplayTradeConfig(
            **common,
            buy1_market_filter_mode="NONE",
        ),
        "B_ENTRY_BALANCED": ReplayTradeConfig(
            **common,
            buy1_market_filter_mode="BALANCED",
        ),
        "C_ENTRY_STRICT": ReplayTradeConfig(
            **common,
            buy1_market_filter_mode="STRICT",
        ),
    }


def compare_buy1_market_strategies(
    dates: Iterable[str] | None = None,
    symbols: Iterable[str] | None = None,
) -> dict:
    """
    BUY1 시장필터 A/B/C 비교.
    실제 주문은 발생하지 않는다.
    """
    if dates is None:
        dates = ["2026-08-14"]

    dates = [str(x).strip() for x in dates if str(x).strip()]
    symbols = _normalize_symbols(symbols)
    strategies = _entry_market_strategy_configs()

    daily_rows = []

    for date_text in dates:
        for strategy_name, cfg in strategies.items():
            try:
                result = run_trade_replay(
                    date_text=date_text,
                    symbols=symbols,
                    config=cfg,
                )

                summary = dict(result.get("summary", {}) or {})
                events = list(result.get("events", []) or [])

                daily_rows.append({
                    "전략": strategy_name,
                    "날짜": date_text,
                    "성공": True,
                    "실현손익USD": float(summary.get("실현손익USD", 0) or 0),
                    "일일예산대비수익률": float(
                        summary.get("일일예산5000달러대비수익률", 0) or 0
                    ),
                    "거래종목수": int(summary.get("거래종목수", 0) or 0),
                    "수익종목수": int(summary.get("수익종목수", 0) or 0),
                    "손실종목수": int(summary.get("손실종목수", 0) or 0),
                    "BUY1횟수": int(
                        sum(1 for e in events if str(e.get("액션", "")) == "BUY1")
                    ),
                    "BUY2횟수": int(
                        sum(1 for e in events if str(e.get("액션", "")) == "BUY2")
                    ),
                    "손절횟수": int(
                        sum(1 for e in events if str(e.get("액션", "")) == "STOP_LOSS")
                    ),
                })
            except Exception as e:
                daily_rows.append({
                    "전략": strategy_name,
                    "날짜": date_text,
                    "성공": False,
                    "오류": f"{type(e).__name__}: {e}",
                })

    daily_df = pd.DataFrame(daily_rows)
    aggregate_rows = []

    for strategy_name in strategies:
        s = daily_df[
            (daily_df["전략"] == strategy_name)
            & (daily_df["성공"] == True)
        ].copy()

        if s.empty:
            aggregate_rows.append({
                "전략": strategy_name,
                "성공일수": 0,
                "총실현손익USD": 0.0,
                "평균일손익USD": 0.0,
                "수익일수": 0,
                "손실일수": 0,
                "일승률": 0.0,
                "최악의날손익USD": 0.0,
                "최고의날손익USD": 0.0,
                "총BUY1횟수": 0,
                "총BUY2횟수": 0,
                "총손절횟수": 0,
                "종목승률": 0.0,
            })
            continue

        profit_symbols = int(s["수익종목수"].sum())
        loss_symbols = int(s["손실종목수"].sum())
        total_closed = profit_symbols + loss_symbols

        aggregate_rows.append({
            "전략": strategy_name,
            "성공일수": int(len(s)),
            "총실현손익USD": round(float(s["실현손익USD"].sum()), 2),
            "평균일손익USD": round(float(s["실현손익USD"].mean()), 2),
            "수익일수": int((s["실현손익USD"] > 0).sum()),
            "손실일수": int((s["실현손익USD"] < 0).sum()),
            "일승률": round(
                float((s["실현손익USD"] > 0).mean() * 100.0),
                1,
            ),
            "최악의날손익USD": round(float(s["실현손익USD"].min()), 2),
            "최고의날손익USD": round(float(s["실현손익USD"].max()), 2),
            "총BUY1횟수": int(s["BUY1횟수"].sum()),
            "총BUY2횟수": int(s["BUY2횟수"].sum()),
            "총손절횟수": int(s["손절횟수"].sum()),
            "종목승률": round(
                profit_symbols / total_closed * 100.0
                if total_closed > 0 else 0.0,
                1,
            ),
        })

    aggregate_df = pd.DataFrame(aggregate_rows)

    if not aggregate_df.empty:
        ranked = aggregate_df.sort_values(
            ["총실현손익USD", "최악의날손익USD", "종목승률"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        recommended = str(ranked.iloc[0]["전략"])
    else:
        recommended = ""

    stamp = datetime.now(ET).strftime("%Y%m%d_%H%M%S")
    daily_path = REPLAY_DIR / f"buy1_market_abc_daily_{stamp}.csv"
    aggregate_path = REPLAY_DIR / f"buy1_market_abc_summary_{stamp}.csv"
    json_path = REPLAY_DIR / f"buy1_market_abc_{stamp}.json"

    daily_df.to_csv(daily_path, index=False, encoding="utf-8-sig")
    aggregate_df.to_csv(aggregate_path, index=False, encoding="utf-8-sig")

    payload = {
        "ok": True,
        "version": "buy1-market-abc-v1",
        "dates": dates,
        "universe_count": len(symbols),
        "buy2_fixed": "B_STRICT",
        "aggregate": aggregate_df.to_dict("records"),
        "daily": daily_df.to_dict("records"),
        "recommended_by_replay": recommended,
        "warning": (
            "시장필터도 소수 날짜에 맞춘 과최적화를 피해야 하므로 "
            "여러 거래일로 검증 후 실제 auto_engine.py에 적용해야 함"
        ),
        "files": {
            "daily_csv": str(daily_path),
            "aggregate_csv": str(aggregate_path),
            "json": str(json_path),
        },
    }

    json_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    return payload


def _buy1_confirmation_strategy_configs() -> dict[str, ReplayTradeConfig]:
    """
    시장필터는 사용하지 않고 BUY2는 B_STRICT로 고정.
    BUY1의 '신호 지속성'만 비교한다.
    """
    common = dict(
        buy1_market_filter_mode="NONE",
        buy2_enabled=True,
        us_add2_trigger_pct=0.80,
        buy2_min_hold_minutes=5.0,
        buy2_max_rank=3,
        buy2_min_score=70.0,
        buy2_require_recent5_positive=True,
        buy2_require_recent10_positive=True,
        buy2_require_relative_strength_positive=True,
    )

    return {
        "A_BUY1_IMMEDIATE": ReplayTradeConfig(
            **common,
            buy1_confirmation_scans=1,
            buy1_confirmation_max_rank=5,
            buy1_require_relative_strength_hold=False,
        ),
        "B_BUY1_2SCANS": ReplayTradeConfig(
            **common,
            buy1_confirmation_scans=2,
            buy1_confirmation_max_rank=5,
            buy1_require_relative_strength_hold=False,
        ),
        "C_BUY1_2SCANS_TOP3_RS": ReplayTradeConfig(
            **common,
            buy1_confirmation_scans=2,
            buy1_confirmation_max_rank=3,
            buy1_require_relative_strength_hold=True,
            buy1_relative_strength_tolerance=0.10,
        ),
    }


def compare_buy1_confirmation_strategies(
    dates: Iterable[str] | None = None,
    symbols: Iterable[str] | None = None,
) -> dict:
    """
    BUY1 연속신호 A/B/C 비교.
    실제 주문은 발생하지 않는다.
    """
    if dates is None:
        dates = ["2026-08-14"]

    dates = [str(x).strip() for x in dates if str(x).strip()]
    symbols = _normalize_symbols(symbols)
    strategies = _buy1_confirmation_strategy_configs()

    daily_rows = []

    for date_text in dates:
        for strategy_name, cfg in strategies.items():
            try:
                result = run_trade_replay(
                    date_text=date_text,
                    symbols=symbols,
                    config=cfg,
                )

                summary = dict(result.get("summary", {}) or {})
                events = list(result.get("events", []) or [])

                daily_rows.append({
                    "전략": strategy_name,
                    "날짜": date_text,
                    "성공": True,
                    "실현손익USD": float(summary.get("실현손익USD", 0) or 0),
                    "일일예산대비수익률": float(
                        summary.get("일일예산5000달러대비수익률", 0) or 0
                    ),
                    "거래종목수": int(summary.get("거래종목수", 0) or 0),
                    "수익종목수": int(summary.get("수익종목수", 0) or 0),
                    "손실종목수": int(summary.get("손실종목수", 0) or 0),
                    "BUY1횟수": int(
                        sum(1 for e in events if str(e.get("액션", "")) == "BUY1")
                    ),
                    "BUY2횟수": int(
                        sum(1 for e in events if str(e.get("액션", "")) == "BUY2")
                    ),
                    "손절횟수": int(
                        sum(1 for e in events if str(e.get("액션", "")) == "STOP_LOSS")
                    ),
                })
            except Exception as e:
                daily_rows.append({
                    "전략": strategy_name,
                    "날짜": date_text,
                    "성공": False,
                    "오류": f"{type(e).__name__}: {e}",
                })

    daily_df = pd.DataFrame(daily_rows)
    aggregate_rows = []

    for strategy_name in strategies:
        s = daily_df[
            (daily_df["전략"] == strategy_name)
            & (daily_df["성공"] == True)
        ].copy()

        if s.empty:
            aggregate_rows.append({
                "전략": strategy_name,
                "성공일수": 0,
                "총실현손익USD": 0.0,
                "평균일손익USD": 0.0,
                "수익일수": 0,
                "손실일수": 0,
                "일승률": 0.0,
                "최악의날손익USD": 0.0,
                "최고의날손익USD": 0.0,
                "총BUY1횟수": 0,
                "총BUY2횟수": 0,
                "총손절횟수": 0,
                "종목승률": 0.0,
            })
            continue

        profit_symbols = int(s["수익종목수"].sum())
        loss_symbols = int(s["손실종목수"].sum())
        total_closed = profit_symbols + loss_symbols

        aggregate_rows.append({
            "전략": strategy_name,
            "성공일수": int(len(s)),
            "총실현손익USD": round(float(s["실현손익USD"].sum()), 2),
            "평균일손익USD": round(float(s["실현손익USD"].mean()), 2),
            "수익일수": int((s["실현손익USD"] > 0).sum()),
            "손실일수": int((s["실현손익USD"] < 0).sum()),
            "일승률": round(
                float((s["실현손익USD"] > 0).mean() * 100.0),
                1,
            ),
            "최악의날손익USD": round(float(s["실현손익USD"].min()), 2),
            "최고의날손익USD": round(float(s["실현손익USD"].max()), 2),
            "총BUY1횟수": int(s["BUY1횟수"].sum()),
            "총BUY2횟수": int(s["BUY2횟수"].sum()),
            "총손절횟수": int(s["손절횟수"].sum()),
            "종목승률": round(
                profit_symbols / total_closed * 100.0
                if total_closed > 0 else 0.0,
                1,
            ),
        })

    aggregate_df = pd.DataFrame(aggregate_rows)

    if not aggregate_df.empty:
        ranked = aggregate_df.sort_values(
            ["총실현손익USD", "최악의날손익USD", "종목승률"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        recommended = str(ranked.iloc[0]["전략"])
    else:
        recommended = ""

    stamp = datetime.now(ET).strftime("%Y%m%d_%H%M%S")
    daily_path = REPLAY_DIR / f"buy1_confirm_abc_daily_{stamp}.csv"
    aggregate_path = REPLAY_DIR / f"buy1_confirm_abc_summary_{stamp}.csv"
    json_path = REPLAY_DIR / f"buy1_confirm_abc_{stamp}.json"

    daily_df.to_csv(daily_path, index=False, encoding="utf-8-sig")
    aggregate_df.to_csv(aggregate_path, index=False, encoding="utf-8-sig")

    payload = {
        "ok": True,
        "version": "buy1-confirm-abc-v1",
        "dates": dates,
        "universe_count": len(symbols),
        "buy2_fixed": "B_STRICT",
        "market_filter": "NONE",
        "aggregate": aggregate_df.to_dict("records"),
        "daily": daily_df.to_dict("records"),
        "recommended_by_replay": recommended,
        "warning": (
            "짧은 기간 리플레이 성과만으로 실제 수익을 보장할 수 없으며, "
            "더 많은 날짜와 모의매매에서 재검증해야 함"
        ),
        "files": {
            "daily_csv": str(daily_path),
            "aggregate_csv": str(aggregate_path),
            "json": str(json_path),
        },
    }

    json_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    return payload


def _entry_sizing_strategy_configs() -> dict[str, ReplayTradeConfig]:
    """
    V3.1 종목선정 유지.
    시장필터 없음.
    BUY1은 즉시 진입.
    BUY2는 B_STRICT로 고정.
    차이는 초기 선진입/다음 스캔 확인보강 비중뿐.
    """
    common = dict(
        buy1_market_filter_mode="NONE",
        buy1_confirmation_scans=1,
        buy1_confirmation_max_rank=5,
        buy1_require_relative_strength_hold=False,

        buy2_enabled=True,
        us_add2_trigger_pct=0.80,
        buy2_min_hold_minutes=5.0,
        buy2_max_rank=3,
        buy2_min_score=70.0,
        buy2_require_recent5_positive=True,
        buy2_require_recent10_positive=True,
        buy2_require_relative_strength_positive=True,
    )

    return {
        "A_SIZE_50_0_50": ReplayTradeConfig(
            **common,
            entry_initial_pct=50.0,
            entry_confirm_pct=0.0,
        ),
        "B_SIZE_25_25_50": ReplayTradeConfig(
            **common,
            entry_initial_pct=25.0,
            entry_confirm_pct=25.0,
            entry_confirm_max_wait_minutes=3.0,
        ),
        "C_SIZE_33_17_50": ReplayTradeConfig(
            **common,
            entry_initial_pct=33.0,
            entry_confirm_pct=17.0,
            entry_confirm_max_wait_minutes=3.0,
        ),
    }


def compare_entry_sizing_strategies(
    dates: Iterable[str] | None = None,
    symbols: Iterable[str] | None = None,
) -> dict:
    """
    초기 선진입 크기 A/B/C 비교.
    실제 주문은 발생하지 않는다.
    """
    if dates is None:
        dates = ["2026-08-14"]

    dates = [str(x).strip() for x in dates if str(x).strip()]
    symbols = _normalize_symbols(symbols)
    strategies = _entry_sizing_strategy_configs()

    daily_rows = []

    for date_text in dates:
        for strategy_name, cfg in strategies.items():
            try:
                result = run_trade_replay(
                    date_text=date_text,
                    symbols=symbols,
                    config=cfg,
                )

                summary = dict(result.get("summary", {}) or {})
                events = list(result.get("events", []) or [])

                daily_rows.append({
                    "전략": strategy_name,
                    "날짜": date_text,
                    "성공": True,
                    "실현손익USD": float(
                        summary.get("실현손익USD", 0) or 0
                    ),
                    "일일예산대비수익률": float(
                        summary.get(
                            "일일예산5000달러대비수익률",
                            0,
                        ) or 0
                    ),
                    "거래종목수": int(
                        summary.get("거래종목수", 0) or 0
                    ),
                    "수익종목수": int(
                        summary.get("수익종목수", 0) or 0
                    ),
                    "손실종목수": int(
                        summary.get("손실종목수", 0) or 0
                    ),
                    "BUY1횟수": int(
                        sum(
                            1 for e in events
                            if str(e.get("액션", "")) == "BUY1"
                        )
                    ),
                    "확인보강횟수": int(
                        sum(
                            1 for e in events
                            if str(e.get("액션", "")) == "BUY1_CONFIRM"
                        )
                    ),
                    "BUY2횟수": int(
                        sum(
                            1 for e in events
                            if str(e.get("액션", "")) == "BUY2"
                        )
                    ),
                    "손절횟수": int(
                        sum(
                            1 for e in events
                            if str(e.get("액션", "")) == "STOP_LOSS"
                        )
                    ),
                })

            except Exception as e:
                daily_rows.append({
                    "전략": strategy_name,
                    "날짜": date_text,
                    "성공": False,
                    "오류": f"{type(e).__name__}: {e}",
                })

    daily_df = pd.DataFrame(daily_rows)
    aggregate_rows = []

    for strategy_name in strategies:
        s = daily_df[
            (daily_df["전략"] == strategy_name)
            & (daily_df["성공"] == True)
        ].copy()

        if s.empty:
            aggregate_rows.append({
                "전략": strategy_name,
                "성공일수": 0,
                "총실현손익USD": 0.0,
                "평균일손익USD": 0.0,
                "수익일수": 0,
                "손실일수": 0,
                "일승률": 0.0,
                "최악의날손익USD": 0.0,
                "최고의날손익USD": 0.0,
                "총BUY1횟수": 0,
                "총확인보강횟수": 0,
                "총BUY2횟수": 0,
                "총손절횟수": 0,
                "종목승률": 0.0,
            })
            continue

        profit_symbols = int(s["수익종목수"].sum())
        loss_symbols = int(s["손실종목수"].sum())
        total_closed = profit_symbols + loss_symbols

        aggregate_rows.append({
            "전략": strategy_name,
            "성공일수": int(len(s)),
            "총실현손익USD": round(
                float(s["실현손익USD"].sum()),
                2,
            ),
            "평균일손익USD": round(
                float(s["실현손익USD"].mean()),
                2,
            ),
            "수익일수": int(
                (s["실현손익USD"] > 0).sum()
            ),
            "손실일수": int(
                (s["실현손익USD"] < 0).sum()
            ),
            "일승률": round(
                float(
                    (s["실현손익USD"] > 0).mean()
                    * 100.0
                ),
                1,
            ),
            "최악의날손익USD": round(
                float(s["실현손익USD"].min()),
                2,
            ),
            "최고의날손익USD": round(
                float(s["실현손익USD"].max()),
                2,
            ),
            "총BUY1횟수": int(
                s["BUY1횟수"].sum()
            ),
            "총확인보강횟수": int(
                s["확인보강횟수"].sum()
            ),
            "총BUY2횟수": int(
                s["BUY2횟수"].sum()
            ),
            "총손절횟수": int(
                s["손절횟수"].sum()
            ),
            "종목승률": round(
                (
                    profit_symbols
                    / total_closed
                    * 100.0
                )
                if total_closed > 0
                else 0.0,
                1,
            ),
        })

    aggregate_df = pd.DataFrame(aggregate_rows)

    if not aggregate_df.empty:
        ranked = aggregate_df.sort_values(
            [
                "총실현손익USD",
                "최악의날손익USD",
                "종목승률",
            ],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        recommended = str(ranked.iloc[0]["전략"])
    else:
        recommended = ""

    stamp = datetime.now(ET).strftime(
        "%Y%m%d_%H%M%S"
    )
    daily_path = (
        REPLAY_DIR
        / f"entry_sizing_abc_daily_{stamp}.csv"
    )
    aggregate_path = (
        REPLAY_DIR
        / f"entry_sizing_abc_summary_{stamp}.csv"
    )
    json_path = (
        REPLAY_DIR
        / f"entry_sizing_abc_{stamp}.json"
    )

    daily_df.to_csv(
        daily_path,
        index=False,
        encoding="utf-8-sig",
    )
    aggregate_df.to_csv(
        aggregate_path,
        index=False,
        encoding="utf-8-sig",
    )

    payload = {
        "ok": True,
        "version": "entry-sizing-abc-v1",
        "dates": dates,
        "universe_count": len(symbols),
        "buy2_fixed": "B_STRICT",
        "market_filter": "NONE",
        "strategies": {
            "A_SIZE_50_0_50": (
                "즉시 50% / 확인보강 없음 / BUY2 최대 50%"
            ),
            "B_SIZE_25_25_50": (
                "즉시 25% / 다음 스캔 확인 시 25% / BUY2 최대 50%"
            ),
            "C_SIZE_33_17_50": (
                "즉시 33% / 다음 스캔 확인 시 17% / BUY2 최대 50%"
            ),
        },
        "aggregate": aggregate_df.to_dict(
            "records"
        ),
        "daily": daily_df.to_dict("records"),
        "recommended_by_replay": recommended,
        "warning": (
            "리플레이 수익은 실제 체결·수수료·슬리피지와 다를 수 있고 "
            "미래 수익을 보장하지 않음. 여러 날짜와 모의매매 재검증 필요."
        ),
        "files": {
            "daily_csv": str(daily_path),
            "aggregate_csv": str(aggregate_path),
            "json": str(json_path),
        },
    }

    json_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    return payload


def _exit_strategy_configs() -> dict[str, ReplayTradeConfig]:
    """
    진입은 현재 최선 조합으로 고정:
    - BUY1 즉시 50%
    - 시장필터 없음
    - BUY1 연속확인 없음
    - BUY2 B_STRICT

    매도만 A/B/C 비교.
    """
    common = dict(
        entry_initial_pct=50.0,
        entry_confirm_pct=0.0,

        buy1_market_filter_mode="NONE",
        buy1_confirmation_scans=1,
        buy1_confirmation_max_rank=5,
        buy1_require_relative_strength_hold=False,

        buy2_enabled=True,
        us_add2_trigger_pct=0.80,
        buy2_min_hold_minutes=5.0,
        buy2_max_rank=3,
        buy2_min_score=70.0,
        buy2_require_recent5_positive=True,
        buy2_require_recent10_positive=True,
        buy2_require_relative_strength_positive=True,
    )

    return {
        "A_EXIT_CURRENT": ReplayTradeConfig(
            **common,
            early_exit_mode="NONE",
        ),
        "B_EXIT_WEAK60": ReplayTradeConfig(
            **common,
            early_exit_mode="WEAK60",
            early_exit_min_hold_minutes=60.0,
            early_exit_loss_threshold_pct=-0.50,
            early_exit_weak_points=5,
        ),
        "C_EXIT_WEAK60_STAGNANT": ReplayTradeConfig(
            **common,
            early_exit_mode="WEAK60_STAGNANT",
            early_exit_min_hold_minutes=60.0,
            early_exit_loss_threshold_pct=-0.50,
            early_exit_weak_points=5,
            stagnant_exit_pnl_max_pct=0.20,
            stagnant_exit_score_max=50.0,
            stagnant_exit_momentum_abs_max=0.05,
        ),
    }


def compare_exit_strategies(
    dates: Iterable[str] | None = None,
    symbols: Iterable[str] | None = None,
) -> dict:
    """
    60분 조기청산 A/B/C 실제 주문흐름 리플레이.
    조기청산으로 자리가 비면 이후 신규진입도 다시 계산된다.
    실제 KIS 주문은 발생하지 않는다.
    """
    if dates is None:
        dates = ["2026-08-14"]

    dates = [str(x).strip() for x in dates if str(x).strip()]
    symbols = _normalize_symbols(symbols)
    strategies = _exit_strategy_configs()

    daily_rows = []

    for date_text in dates:
        for strategy_name, cfg in strategies.items():
            try:
                result = run_trade_replay(
                    date_text=date_text,
                    symbols=symbols,
                    config=cfg,
                )

                summary = dict(result.get("summary", {}) or {})
                events = list(result.get("events", []) or [])

                daily_rows.append({
                    "전략": strategy_name,
                    "날짜": date_text,
                    "성공": True,
                    "실현손익USD": float(
                        summary.get("실현손익USD", 0) or 0
                    ),
                    "일일예산대비수익률": float(
                        summary.get(
                            "일일예산5000달러대비수익률",
                            0,
                        ) or 0
                    ),
                    "거래종목수": int(
                        summary.get("거래종목수", 0) or 0
                    ),
                    "수익종목수": int(
                        summary.get("수익종목수", 0) or 0
                    ),
                    "손실종목수": int(
                        summary.get("손실종목수", 0) or 0
                    ),
                    "BUY1횟수": int(
                        sum(
                            1 for e in events
                            if str(e.get("액션", "")) == "BUY1"
                        )
                    ),
                    "BUY2횟수": int(
                        sum(
                            1 for e in events
                            if str(e.get("액션", "")) == "BUY2"
                        )
                    ),
                    "손절횟수": int(
                        sum(
                            1 for e in events
                            if str(e.get("액션", "")) == "STOP_LOSS"
                        )
                    ),
                    "강한약화청산횟수": int(
                        sum(
                            1 for e in events
                            if str(e.get("액션", "")) == "EARLY_EXIT_WEAK60"
                        )
                    ),
                    "정체청산횟수": int(
                        sum(
                            1 for e in events
                            if str(e.get("액션", "")) == "EARLY_EXIT_STAGNANT60"
                        )
                    ),
                })

            except Exception as e:
                daily_rows.append({
                    "전략": strategy_name,
                    "날짜": date_text,
                    "성공": False,
                    "오류": f"{type(e).__name__}: {e}",
                })

    daily_df = pd.DataFrame(daily_rows)
    aggregate_rows = []

    for strategy_name in strategies:
        s = daily_df[
            (daily_df["전략"] == strategy_name)
            & (daily_df["성공"] == True)
        ].copy()

        if s.empty:
            aggregate_rows.append({
                "전략": strategy_name,
                "성공일수": 0,
                "총실현손익USD": 0.0,
                "평균일손익USD": 0.0,
                "수익일수": 0,
                "손실일수": 0,
                "일승률": 0.0,
                "최악의날손익USD": 0.0,
                "최고의날손익USD": 0.0,
                "총손절횟수": 0,
                "총강한약화청산": 0,
                "총정체청산": 0,
                "종목승률": 0.0,
            })
            continue

        profit_symbols = int(s["수익종목수"].sum())
        loss_symbols = int(s["손실종목수"].sum())
        total_closed = profit_symbols + loss_symbols

        aggregate_rows.append({
            "전략": strategy_name,
            "성공일수": int(len(s)),
            "총실현손익USD": round(
                float(s["실현손익USD"].sum()),
                2,
            ),
            "평균일손익USD": round(
                float(s["실현손익USD"].mean()),
                2,
            ),
            "수익일수": int(
                (s["실현손익USD"] > 0).sum()
            ),
            "손실일수": int(
                (s["실현손익USD"] < 0).sum()
            ),
            "일승률": round(
                float(
                    (s["실현손익USD"] > 0).mean()
                    * 100.0
                ),
                1,
            ),
            "최악의날손익USD": round(
                float(s["실현손익USD"].min()),
                2,
            ),
            "최고의날손익USD": round(
                float(s["실현손익USD"].max()),
                2,
            ),
            "총손절횟수": int(
                s["손절횟수"].sum()
            ),
            "총강한약화청산": int(
                s["강한약화청산횟수"].sum()
            ),
            "총정체청산": int(
                s["정체청산횟수"].sum()
            ),
            "종목승률": round(
                (
                    profit_symbols
                    / total_closed
                    * 100.0
                )
                if total_closed > 0
                else 0.0,
                1,
            ),
        })

    aggregate_df = pd.DataFrame(aggregate_rows)

    if not aggregate_df.empty:
        ranked = aggregate_df.sort_values(
            [
                "총실현손익USD",
                "최악의날손익USD",
                "종목승률",
            ],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        recommended = str(
            ranked.iloc[0]["전략"]
        )
    else:
        recommended = ""

    stamp = datetime.now(ET).strftime(
        "%Y%m%d_%H%M%S"
    )
    daily_path = (
        REPLAY_DIR
        / f"exit_abc_daily_{stamp}.csv"
    )
    aggregate_path = (
        REPLAY_DIR
        / f"exit_abc_summary_{stamp}.csv"
    )
    json_path = (
        REPLAY_DIR
        / f"exit_abc_{stamp}.json"
    )

    daily_df.to_csv(
        daily_path,
        index=False,
        encoding="utf-8-sig",
    )
    aggregate_df.to_csv(
        aggregate_path,
        index=False,
        encoding="utf-8-sig",
    )

    payload = {
        "ok": True,
        "version": "exit-abc-v1",
        "dates": dates,
        "universe_count": len(symbols),
        "entry_fixed": "BUY1 즉시 50%",
        "buy2_fixed": "B_STRICT",
        "strategies": {
            "A_EXIT_CURRENT": "-3% 손절 등 현재 매도",
            "B_EXIT_WEAK60": (
                "60분+ / 손익 -0.5% 이하 / 약화점수 5점 이상 조기청산"
            ),
            "C_EXIT_WEAK60_STAGNANT": (
                "B + 60분 정체(손익 +0.2% 이하, 매수후보탈락, "
                "점수50미만, 5·10분 절대값 0.05% 이하) 청산"
            ),
        },
        "aggregate": aggregate_df.to_dict(
            "records"
        ),
        "daily": daily_df.to_dict("records"),
        "recommended_by_replay": recommended,
        "warning": (
            "리플레이는 실제 체결과 다를 수 있고 미래 수익을 보장하지 않음. "
            "여러 날짜와 모의매매 재검증 필요."
        ),
        "files": {
            "daily_csv": str(daily_path),
            "aggregate_csv": str(aggregate_path),
            "json": str(json_path),
        },
    }

    json_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    return payload


if __name__ == "__main__":
    result = run_trade_replay(
        date_text=os.getenv("REPLAY_DATE", "2026-08-14"),
        symbols=[
            x.strip()
            for x in os.getenv(
                "REPLAY_SYMBOLS",
                ",".join(DEFAULT_UNIVERSE),
            ).split(",")
            if x.strip()
        ],
    )

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )

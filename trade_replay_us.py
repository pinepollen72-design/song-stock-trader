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
    """
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


def _first_entry_qty(config: ReplayTradeConfig, reference: float) -> int:
    limit_price = _limit_price(config, "BUY", reference)
    if limit_price <= 0:
        return 0

    qty = int(_stage_budget(config, 1) // limit_price)

    if qty > 0:
        return qty

    if (
        config.allow_single_share_over_stage_budget
        and limit_price <= float(config.us_per_stock_budget_usd)
    ):
        return 1

    return 0


def _second_entry_qty(config: ReplayTradeConfig, reference: float) -> int:
    limit_price = _limit_price(config, "BUY", reference)
    if limit_price <= 0:
        return 0
    return int(_stage_budget(config, 2) // limit_price)


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
            latest_top5 = ranked.head(5).copy() if not ranked.empty else pd.DataFrame()

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

            # 신규/추가매수 종료 이후에는 BUY2도 금지
            if _seconds_of_day(now) >= last_entry_sec:
                continue

            # 2차 매수: 실험 설정에 따라 조건을 바꿔 A/B/C 비교
            if (
                bool(config.buy2_enabled)
                and int(pos.get("buy_stage", 1)) == 1
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
                        f"TOP5 신규진입 · 점수 {combined:.1f} · "
                        f"당일 {float(row.get('당일등락률', 0) or 0):+.2f}% · "
                        f"상대강도 {float(row.get('상대강도', 0) or 0):+.2f}%p"
                    ),
                    score=combined,
                    rank=rank,
                )

                daily_buy_amount += cost

                positions[symbol] = {
                    "qty": int(qty),
                    "avg_price": float(fill),
                    "buy_stage": 1,
                    "take1_sent": False,
                    "peak_pnl": 0.0,
                    "opened_at": now.isoformat(),
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
        "version": "trade-replay-v2-buy2-lab",
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
    실제 주문은 발생하지 않는다.
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
    strategy_details = {}

    for strategy_name, cfg in strategies.items():
        per_day = []

        for date_text in dates:
            try:
                result = run_trade_replay(
                    date_text=date_text,
                    symbols=symbols,
                    config=cfg,
                )

                summary = dict(result.get("summary", {}) or {})

                row = {
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
                }

                per_day.append({
                    "date": date_text,
                    "summary": summary,
                    "symbols": result.get("symbols", []),
                })
                daily_rows.append(row)

            except Exception as e:
                daily_rows.append({
                    "전략": strategy_name,
                    "날짜": date_text,
                    "성공": False,
                    "오류": f"{type(e).__name__}: {e}",
                })

        strategy_details[strategy_name] = per_day

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

    # 1차 추천은 총손익 우선, 동률이면 최악의날 손실이 작은 전략 우선
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
        "version": "buy2-abc-v1",
        "dates": dates,
        "universe_count": len(symbols),
        "strategies": {
            key: asdict(value)
            for key, value in strategies.items()
        },
        "aggregate": aggregate_df.to_dict("records"),
        "daily": daily_df.to_dict("records"),
        "recommended_by_replay": recommended,
        "warning": (
            "5거래일 리플레이 결과는 과최적화 가능성이 있으므로 "
            "실제 전략 변경 전 더 많은 날짜로 재검증하는 것이 안전함"
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

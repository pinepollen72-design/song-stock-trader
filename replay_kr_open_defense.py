from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from replay_kr import (
    KRReplayConfig,
    _normalize_universe,
    _download_intraday,
    _build_top5_at,
    _price_at,
    _fill_price,
    _clock_seconds,
    _seconds_of_day,
    _bars_until,
    run_kr_trade_replay,
)

KST = ZoneInfo("Asia/Seoul")
OPEN_DEFENSE_VERSION = "kr-open-defense-d-v1"


def _resolve_state_dir() -> Path:
    explicit = os.getenv("SONG_TRADER_STATE_DIR", "").strip()
    if explicit:
        return Path(explicit)
    railway_volume = os.getenv("RAILWAY_VOLUME_MOUNT_PATH", "").strip()
    if railway_volume:
        return Path(railway_volume) / "song_trader_v2"
    return Path("/tmp/song_trader_v2")


STATE_DIR = _resolve_state_dir()
CACHE_DIR = STATE_DIR / "replays" / "kr_open_defense"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class OpenDefenseConfig:
    # C_NO_BUY2의 기본 위험/청산 규칙은 그대로 둔다.
    daily_budget_krw: int = 10_000_000
    per_stock_budget_krw: int = 3_000_000
    max_positions: int = 3
    max_daily_orders: int = 12
    min_score: float = 50.0
    stop_loss_pct: float = 3.0
    take1_pct: float = 3.0
    take2_pct: float = 5.0
    profit_guard_trigger_pct: float = 1.20
    profit_guard_drawdown_pct: float = 0.80
    last_entry_time: str = "14:50"
    force_exit_time: str = "15:15"
    scan_seconds: int = 90
    manage_seconds: int = 45
    scan_count: int = 8
    buy_slippage_pct: float = 0.10
    sell_slippage_pct: float = 0.10

    # -----------------------------
    # OPEN DEFENSE D
    # -----------------------------
    # 리플레이가 09:09부터 시작하므로 장초반 방어 구간을 09:09~09:20으로 정의.
    defense_start_time: str = "09:09"
    defense_end_time: str = "09:20"

    # C의 정상 BUY1은 종목당 예산의 50%(150만원).
    # D는 장초반에는 그 절반인 25%(75만원)만 먼저 진입한다.
    normal_entry_pct_of_stock_budget: float = 0.50
    defense_entry_pct_of_stock_budget: float = 0.25

    # 장초반에 동시에 3종목을 추격하지 않는다.
    defense_max_positions: int = 2

    # 첫 진입 후 4분 이상 버티고 모멘텀/VWAP가 살아 있을 때만
    # 나머지 25%를 OPEN_CONFIRM으로 채워 C의 정상 50% 크기로 만든다.
    confirm_min_hold_minutes: float = 4.0
    confirm_min_score: float = 65.0
    confirm_max_rank: int = 3
    confirm_min_pnl_pct: float = -0.10
    confirm_min_volume_ratio: float = 1.05
    confirm_require_ret3_nonnegative: bool = True
    confirm_require_ret5_positive: bool = True
    confirm_require_above_vwap: bool = True

    # 장초반 돌파 실패는 -3%까지 기다리지 않는다.
    defense_fail_window_minutes: float = 30.0
    defense_hard_fail_pct: float = -1.00
    vwap_fail_gap_pct: float = -0.25
    vwap_fail_ret3_pct: float = -0.15
    vwap_fail_ret5_pct: float = 0.00
    vwap_fail_min_hold_minutes: float = 3.0

    # 너무 벌어진 돌파인데 거래량 가속이 약하면 즉시 추격하지 않는다.
    chase_vwap_gap_pct: float = 2.00
    chase_max_volume_ratio: float = 1.20
    chase_min_day_return_pct: float = 5.0


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _rank_number(row) -> int:
    raw = str(row.get("순위", "") or "")
    digits = "".join(ch for ch in raw if ch.isdigit())
    try:
        return int(digits) if digits else 999
    except Exception:
        return 999


def _vwap_at(frame: pd.DataFrame, date_text: str, now: pd.Timestamp) -> float:
    bars = _bars_until(frame, now, date_text)
    if bars is None or bars.empty:
        return 0.0
    vol = pd.to_numeric(bars.get("Volume", 0), errors="coerce").fillna(0.0)
    if float(vol.sum()) <= 0:
        return 0.0
    high = pd.to_numeric(bars.get("High", 0), errors="coerce").fillna(0.0)
    low = pd.to_numeric(bars.get("Low", 0), errors="coerce").fillna(0.0)
    close = pd.to_numeric(bars.get("Close", 0), errors="coerce").fillna(0.0)
    typical = (high + low + close) / 3.0
    return float((typical * vol).sum() / vol.sum())


def _vwap_gap_pct(frame: pd.DataFrame, date_text: str, now: pd.Timestamp, price: float) -> float:
    vwap = _vwap_at(frame, date_text, now)
    if vwap <= 0 or price <= 0:
        return 0.0
    return (float(price) / vwap - 1.0) * 100.0


def _cache_path(date_text: str) -> Path:
    return CACHE_DIR / f"open_defense_{date_text}.json"


def _normal_entry_amount(cfg: OpenDefenseConfig) -> int:
    return int(cfg.per_stock_budget_krw * cfg.normal_entry_pct_of_stock_budget)


def _defense_entry_amount(cfg: OpenDefenseConfig) -> int:
    return int(cfg.per_stock_budget_krw * cfg.defense_entry_pct_of_stock_budget)


def _is_chase_wait(cfg: OpenDefenseConfig, row, vwap_gap: float) -> tuple[bool, str]:
    day_ret = _safe_float(row.get("등락률", 0))
    vol_ratio = _safe_float(row.get("거래량배수", 0))
    if (
        day_ret >= float(cfg.chase_min_day_return_pct)
        and vwap_gap >= float(cfg.chase_vwap_gap_pct)
        and vol_ratio <= float(cfg.chase_max_volume_ratio)
    ):
        return True, (
            f"OPEN_CHASE_WAIT · 당일 {day_ret:+.2f}% · "
            f"VWAP {vwap_gap:+.2f}% · 거래량 {vol_ratio:.2f}배"
        )
    return False, ""


def _confirm_allowed(
    cfg: OpenDefenseConfig,
    pos: dict,
    row,
    now: pd.Timestamp,
    pnl: float,
    vwap_gap: float,
) -> tuple[bool, str]:
    if row is None:
        return False, "TOP5 이탈"
    try:
        created = pd.Timestamp(pos.get("created_at"))
        if created.tzinfo is None:
            created = created.tz_localize(KST)
        hold_min = max(0.0, (now - created).total_seconds() / 60.0)
    except Exception:
        hold_min = 0.0

    score = _safe_float(row.get("종합점수", 0))
    rank = _rank_number(row)
    ret3 = _safe_float(row.get("최근3분수익률", 0))
    ret5 = _safe_float(row.get("최근5분수익률", 0))
    vol = _safe_float(row.get("거래량배수", 0))
    weak = bool(row.get("모멘텀약화", False))
    signal_ok = "매수 후보" in str(row.get("판정", ""))

    checks = [
        hold_min >= float(cfg.confirm_min_hold_minutes),
        signal_ok,
        not weak,
        score >= float(cfg.confirm_min_score),
        rank <= int(cfg.confirm_max_rank),
        pnl >= float(cfg.confirm_min_pnl_pct),
        vol >= float(cfg.confirm_min_volume_ratio),
    ]
    if cfg.confirm_require_ret3_nonnegative:
        checks.append(ret3 >= 0.0)
    if cfg.confirm_require_ret5_positive:
        checks.append(ret5 > 0.0)
    if cfg.confirm_require_above_vwap:
        checks.append(vwap_gap >= 0.0)

    if not all(checks):
        return False, (
            f"OPEN_CONFIRM_WAIT · 보유 {hold_min:.1f}분 · pnl {pnl:+.2f}% · "
            f"TOP{rank} · 점수 {score:.1f} · 3/5분 {ret3:+.2f}/{ret5:+.2f}% · "
            f"거래량 {vol:.2f}배 · VWAP {vwap_gap:+.2f}%"
        )
    return True, (
        f"OPEN_CONFIRM · 보유 {hold_min:.1f}분 · pnl {pnl:+.2f}% · "
        f"TOP{rank} · 점수 {score:.1f} · 3/5분 {ret3:+.2f}/{ret5:+.2f}% · "
        f"거래량 {vol:.2f}배 · VWAP {vwap_gap:+.2f}%"
    )


def run_kr_open_defense_replay(
    date_text: str = "2026-08-10",
    codes: Iterable[str] | None = None,
    config: OpenDefenseConfig | None = None,
    refresh: bool = False,
) -> dict:
    cfg = config or OpenDefenseConfig()
    cache_path = _cache_path(date_text)
    if not refresh and not codes and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("ok") is True and cached.get("version") == OPEN_DEFENSE_VERSION:
                cached["cached"] = True
                return cached
        except Exception:
            pass

    # C_NO_BUY2를 같은 데이터/슬리피지 조건으로 다시 계산해 정면 비교한다.
    c_cfg = KRReplayConfig(
        daily_budget_krw=cfg.daily_budget_krw,
        per_stock_budget_krw=cfg.per_stock_budget_krw,
        max_positions=cfg.max_positions,
        max_daily_orders=cfg.max_daily_orders,
        buy1_pct=50,
        buy2_pct=50,
        min_score=cfg.min_score,
        stop_loss_pct=cfg.stop_loss_pct,
        take1_pct=cfg.take1_pct,
        take2_pct=cfg.take2_pct,
        buy2_mode="NONE",
        profit_guard_trigger_pct=cfg.profit_guard_trigger_pct,
        profit_guard_drawdown_pct=cfg.profit_guard_drawdown_pct,
        last_entry_time=cfg.last_entry_time,
        force_exit_time=cfg.force_exit_time,
        scan_seconds=cfg.scan_seconds,
        manage_seconds=cfg.manage_seconds,
        scan_count=cfg.scan_count,
        buy_slippage_pct=cfg.buy_slippage_pct,
        sell_slippage_pct=cfg.sell_slippage_pct,
    )
    baseline = run_kr_trade_replay(
        date_text=date_text,
        codes=codes,
        config=c_cfg,
        use_cache=False,
    )

    universe = _normalize_universe(codes)
    frames, meta = _download_intraday(date_text, universe)
    if not frames:
        raise RuntimeError("해당 날짜의 국내 1분봉 데이터를 받지 못했습니다.")
    target_frames = {
        code: frame
        for code, frame in frames.items()
        if not frame[frame.index.strftime("%Y-%m-%d") == date_text].empty
    }
    if not target_frames:
        raise RuntimeError(f"{date_text} 국내 장중 1분봉 데이터가 없습니다.")

    date0 = pd.Timestamp(date_text, tz=KST)
    start = date0 + pd.Timedelta(hours=9, minutes=9)
    end = date0 + pd.Timedelta(hours=15, minutes=16)
    defense_start_sec = _clock_seconds(cfg.defense_start_time)
    defense_end_sec = _clock_seconds(cfg.defense_end_time)
    last_entry_sec = _clock_seconds(cfg.last_entry_time)
    force_exit_sec = _clock_seconds(cfg.force_exit_time)

    normal_amount = _normal_entry_amount(cfg)
    defense_amount = _defense_entry_amount(cfg)
    confirm_amount = max(0, normal_amount - defense_amount)

    positions: dict[str, dict] = {}
    events: list[dict] = []
    latest_top5 = pd.DataFrame()
    last_scan = None
    daily_buy_amount = 0.0
    daily_orders = 0
    diagnostic = {
        "defense_entries": 0,
        "normal_entries": 0,
        "open_confirms": 0,
        "open_fail_exits": 0,
        "open_vwap_fail_exits": 0,
        "chase_waits": 0,
        "opening_position_cap_blocks": 0,
    }
    chase_wait_seen: set[tuple[str, str]] = set()

    def add_event(ts, symbol, action, side, qty, ref_price, fill_price, reason,
                  pnl="", realized=0.0, score="", rank="", vwap_gap=""):
        nonlocal daily_orders
        events.append({
            "시간KST": ts.isoformat(),
            "종목코드": symbol,
            "종목명": meta.get(symbol, {}).get("name", symbol),
            "액션": action,
            "구분": side,
            "수량": int(qty),
            "기준가": round(float(ref_price), 2),
            "가정체결가": round(float(fill_price), 2),
            "주문금액KRW": int(round(float(fill_price) * int(qty))),
            "손익률": "" if pnl == "" else round(float(pnl), 3),
            "실현손익KRW": int(round(float(realized))),
            "종합점수": score,
            "TOP5순위": rank,
            "VWAP괴리율": "" if vwap_gap == "" else round(float(vwap_gap), 3),
            "이유": reason,
        })
        daily_orders += 1

    now = start
    while now <= end:
        if last_scan is None or (now - last_scan).total_seconds() >= int(cfg.scan_seconds):
            latest_top5 = _build_top5_at(target_frames, meta, date_text, now, cfg.scan_count)
            last_scan = now

        top5_map = {}
        if latest_top5 is not None and not latest_top5.empty:
            for _, r in latest_top5.iterrows():
                top5_map[str(r.get("종목코드", "")).zfill(6)] = r

        # 1) 보유종목 관리
        for symbol, pos in list(positions.items()):
            frame = target_frames.get(symbol)
            if frame is None:
                continue
            ref_price = _price_at(frame, date_text, now)
            if ref_price <= 0:
                continue
            qty = int(pos.get("qty", 0))
            avg = _safe_float(pos.get("avg_price", 0))
            if qty <= 0 or avg <= 0:
                continue
            pnl = (ref_price / avg - 1.0) * 100.0
            peak = max(_safe_float(pos.get("peak_pnl", pnl)), pnl)
            pos["peak_pnl"] = peak
            dd = max(0.0, peak - pnl)
            vg = _vwap_gap_pct(frame, date_text, now, ref_price)

            # 강제청산
            if _seconds_of_day(now) >= force_exit_sec:
                fill = _fill_price(c_cfg, "SELL", ref_price)
                realized = (fill - avg) * qty
                add_event(now, symbol, "FORCE_SELL", "SELL", qty, ref_price, fill,
                          f"당일 강제청산 {cfg.force_exit_time} KST", pnl, realized,
                          vwap_gap=vg)
                positions.pop(symbol, None)
                continue

            # OPEN DEFENSE 전용 조기 실패청산.
            if bool(pos.get("defense_position", False)):
                try:
                    created = pd.Timestamp(pos.get("created_at"))
                    if created.tzinfo is None:
                        created = created.tz_localize(KST)
                    hold_min = max(0.0, (now - created).total_seconds() / 60.0)
                except Exception:
                    hold_min = 0.0

                row = top5_map.get(symbol)
                ret3 = _safe_float(row.get("최근3분수익률", 0)) if row is not None else 0.0
                ret5 = _safe_float(row.get("최근5분수익률", 0)) if row is not None else 0.0

                if hold_min <= float(cfg.defense_fail_window_minutes):
                    if pnl <= float(cfg.defense_hard_fail_pct):
                        fill = _fill_price(c_cfg, "SELL", ref_price)
                        realized = (fill - avg) * qty
                        add_event(
                            now, symbol, "OPEN_FAIL_EXIT", "SELL", qty,
                            ref_price, fill,
                            f"장초반 돌파실패 조기청산 · pnl {pnl:.2f}% <= {cfg.defense_hard_fail_pct:.2f}%",
                            pnl, realized, vwap_gap=vg,
                        )
                        diagnostic["open_fail_exits"] += 1
                        positions.pop(symbol, None)
                        continue

                    if (
                        hold_min >= float(cfg.vwap_fail_min_hold_minutes)
                        and vg <= float(cfg.vwap_fail_gap_pct)
                        and ret3 <= float(cfg.vwap_fail_ret3_pct)
                        and ret5 <= float(cfg.vwap_fail_ret5_pct)
                    ):
                        fill = _fill_price(c_cfg, "SELL", ref_price)
                        realized = (fill - avg) * qty
                        add_event(
                            now, symbol, "OPEN_VWAP_FAIL", "SELL", qty,
                            ref_price, fill,
                            f"VWAP 돌파실패 · VWAP {vg:+.2f}% · 3/5분 {ret3:+.2f}/{ret5:+.2f}%",
                            pnl, realized, vwap_gap=vg,
                        )
                        diagnostic["open_vwap_fail_exits"] += 1
                        positions.pop(symbol, None)
                        continue

            # 기존 C 청산 규칙
            if pnl <= -abs(cfg.stop_loss_pct):
                fill = _fill_price(c_cfg, "SELL", ref_price)
                realized = (fill - avg) * qty
                add_event(now, symbol, "STOP_LOSS", "SELL", qty, ref_price, fill,
                          f"손절 {pnl:.2f}%", pnl, realized, vwap_gap=vg)
                positions.pop(symbol, None)
                continue

            if pnl >= cfg.take1_pct and not bool(pos.get("take1_sent")):
                sell_qty = max(1, qty // 2)
                fill = _fill_price(c_cfg, "SELL", ref_price)
                realized = (fill - avg) * sell_qty
                add_event(now, symbol, "TAKE1", "SELL", sell_qty, ref_price, fill,
                          f"1차 익절 {pnl:.2f}% · 약 50%", pnl, realized, vwap_gap=vg)
                pos["qty"] = qty - sell_qty
                pos["take1_sent"] = True
                if pos["qty"] <= 0:
                    positions.pop(symbol, None)
                continue

            if pnl >= cfg.take2_pct and bool(pos.get("take1_sent")):
                fill = _fill_price(c_cfg, "SELL", ref_price)
                realized = (fill - avg) * qty
                add_event(now, symbol, "TAKE2", "SELL", qty, ref_price, fill,
                          f"2차 익절 {pnl:.2f}% · 전량", pnl, realized, vwap_gap=vg)
                positions.pop(symbol, None)
                continue

            if peak >= cfg.profit_guard_trigger_pct and dd >= cfg.profit_guard_drawdown_pct:
                if not bool(pos.get("take1_sent")):
                    sell_qty = max(1, qty // 2)
                    fill = _fill_price(c_cfg, "SELL", ref_price)
                    realized = (fill - avg) * sell_qty
                    add_event(
                        now, symbol, "PROFIT_GUARD1", "SELL", sell_qty,
                        ref_price, fill,
                        f"수익보호 1차 · 최고 +{peak:.2f}% → 현재 {pnl:.2f}% ({dd:.2f}%p 되밀림)",
                        pnl, realized, vwap_gap=vg,
                    )
                    pos["qty"] = qty - sell_qty
                    pos["take1_sent"] = True
                    if pos["qty"] <= 0:
                        positions.pop(symbol, None)
                    continue
                else:
                    fill = _fill_price(c_cfg, "SELL", ref_price)
                    realized = (fill - avg) * qty
                    add_event(
                        now, symbol, "PROFIT_GUARD2", "SELL", qty,
                        ref_price, fill,
                        f"수익보호 2차 · 최고 +{peak:.2f}% → 현재 {pnl:.2f}% ({dd:.2f}%p 되밀림)",
                        pnl, realized, vwap_gap=vg,
                    )
                    positions.pop(symbol, None)
                    continue

            # OPEN_CONFIRM: 장초반 25% 진입을 정상 C의 50%까지 채우는 단 한 번의 확인매수.
            if (
                bool(pos.get("defense_position", False))
                and not bool(pos.get("open_confirmed", False))
                and _seconds_of_day(now) < defense_end_sec
                and daily_orders < cfg.max_daily_orders
            ):
                row = top5_map.get(symbol)
                ok, why = _confirm_allowed(cfg, pos, row, now, pnl, vg)
                if ok and confirm_amount > 0:
                    fill = _fill_price(c_cfg, "BUY", ref_price)
                    qty2 = int(confirm_amount // fill)
                    cost = fill * qty2
                    if qty2 > 0 and daily_buy_amount + cost <= cfg.daily_budget_krw:
                        old_qty = int(pos["qty"])
                        old_avg = _safe_float(pos["avg_price"])
                        new_qty = old_qty + qty2
                        new_avg = (old_avg * old_qty + fill * qty2) / new_qty
                        score = _safe_float(row.get("종합점수", 0)) if row is not None else ""
                        rank = str(row.get("순위", "")) if row is not None else ""
                        add_event(
                            now, symbol, "OPEN_CONFIRM", "BUY", qty2,
                            ref_price, fill, why, pnl, 0.0, score, rank, vg,
                        )
                        pos["qty"] = new_qty
                        pos["avg_price"] = new_avg
                        pos["open_confirmed"] = True
                        daily_buy_amount += cost
                        diagnostic["open_confirms"] += 1

        # 2) 신규진입
        if (
            _seconds_of_day(now) < last_entry_sec
            and len(positions) < cfg.max_positions
            and daily_orders < cfg.max_daily_orders
            and latest_top5 is not None
            and not latest_top5.empty
        ):
            sec = _seconds_of_day(now)
            defense_now = defense_start_sec <= sec < defense_end_sec
            effective_max_positions = (
                min(cfg.max_positions, cfg.defense_max_positions)
                if defense_now else cfg.max_positions
            )

            if defense_now and len(positions) >= effective_max_positions:
                diagnostic["opening_position_cap_blocks"] += 1
            else:
                for _, row in latest_top5.iterrows():
                    if len(positions) >= effective_max_positions or daily_orders >= cfg.max_daily_orders:
                        break
                    symbol = str(row.get("종목코드", "")).zfill(6)
                    if symbol in positions:
                        continue
                    signal = str(row.get("판정", ""))
                    score = _safe_float(row.get("종합점수", 0))
                    weak = bool(row.get("모멘텀약화", False))
                    if "매수 후보" not in signal or weak or score < cfg.min_score:
                        continue

                    frame = target_frames.get(symbol)
                    if frame is None:
                        continue
                    ref_price = _price_at(frame, date_text, now)
                    if ref_price <= 0:
                        continue
                    vg = _vwap_gap_pct(frame, date_text, now, ref_price)

                    if defense_now:
                        wait, wait_reason = _is_chase_wait(cfg, row, vg)
                        if wait:
                            key = (symbol, now.isoformat())
                            if key not in chase_wait_seen:
                                chase_wait_seen.add(key)
                                diagnostic["chase_waits"] += 1
                            continue
                        target_amount = defense_amount
                        action = "OPEN_DEFENSE_BUY"
                        defense_position = True
                    else:
                        target_amount = normal_amount
                        action = "BUY1"
                        defense_position = False

                    fill = _fill_price(c_cfg, "BUY", ref_price)
                    qty1 = int(target_amount // fill)
                    if qty1 <= 0:
                        # 장초반 방어 크기로 1주도 살 수 없으면 무리해서 예산을 넘기지 않는다.
                        continue
                    cost = fill * qty1
                    if daily_buy_amount + cost > cfg.daily_budget_krw:
                        continue

                    rank = str(row.get("순위", ""))
                    r3 = _safe_float(row.get("최근3분수익률", 0))
                    r5 = _safe_float(row.get("최근5분수익률", 0))
                    vr = _safe_float(row.get("거래량배수", 0))
                    prefix = "OPEN DEFENSE 25%" if defense_now else "C기준 50%"
                    add_event(
                        now, symbol, action, "BUY", qty1, ref_price, fill,
                        f"{prefix} · 점수 {score:.1f} · 3분 {r3:+.2f}% · 5분 {r5:+.2f}% · "
                        f"거래량 {vr:.2f}배 · VWAP {vg:+.2f}%",
                        "", 0.0, score, rank, vg,
                    )
                    positions[symbol] = {
                        "qty": qty1,
                        "avg_price": fill,
                        "created_at": now.isoformat(),
                        "take1_sent": False,
                        "peak_pnl": 0.0,
                        "defense_position": defense_position,
                        "open_confirmed": not defense_position,
                    }
                    daily_buy_amount += cost
                    if defense_now:
                        diagnostic["defense_entries"] += 1
                    else:
                        diagnostic["normal_entries"] += 1

        now += pd.Timedelta(seconds=int(cfg.manage_seconds))

    # 안전청산
    if positions:
        for symbol, pos in list(positions.items()):
            frame = target_frames.get(symbol)
            ref_price = _price_at(frame, date_text, end) if frame is not None else 0.0
            if ref_price <= 0:
                continue
            qty = int(pos.get("qty", 0))
            avg = _safe_float(pos.get("avg_price", 0))
            fill = _fill_price(c_cfg, "SELL", ref_price)
            pnl = (ref_price / avg - 1.0) * 100.0 if avg > 0 else 0.0
            realized = (fill - avg) * qty
            vg = _vwap_gap_pct(frame, date_text, end, ref_price)
            add_event(end, symbol, "FORCE_SELL_END", "SELL", qty, ref_price, fill,
                      "리플레이 종료 안전청산", pnl, realized, vwap_gap=vg)
            positions.pop(symbol, None)

    events_df = pd.DataFrame(events)
    buy_amount = float(events_df.loc[events_df["구분"] == "BUY", "주문금액KRW"].sum()) if not events_df.empty else 0.0
    sell_amount = float(events_df.loc[events_df["구분"] == "SELL", "주문금액KRW"].sum()) if not events_df.empty else 0.0
    realized = float(events_df["실현손익KRW"].sum()) if not events_df.empty else 0.0

    symbol_rows = []
    if not events_df.empty:
        for symbol, g in events_df.groupby("종목코드", sort=False):
            buys = g[g["구분"] == "BUY"]
            sells = g[g["구분"] == "SELL"]
            pnl_sum = float(g["실현손익KRW"].sum())
            symbol_rows.append({
                "종목코드": symbol,
                "종목명": str(g.iloc[0].get("종목명", symbol)),
                "매수횟수": int(len(buys)),
                "매도횟수": int(len(sells)),
                "총매수금액KRW": int(buys["주문금액KRW"].sum()) if not buys.empty else 0,
                "총매도금액KRW": int(sells["주문금액KRW"].sum()) if not sells.empty else 0,
                "실현손익KRW": int(round(pnl_sum)),
                "종료사유": str(sells.iloc[-1].get("액션", "")) if not sells.empty else "",
            })

    summary = {
        "총주문횟수": int(len(events_df)),
        "매수주문횟수": int((events_df["구분"] == "BUY").sum()) if not events_df.empty else 0,
        "매도주문횟수": int((events_df["구분"] == "SELL").sum()) if not events_df.empty else 0,
        "거래종목수": int(events_df["종목코드"].nunique()) if not events_df.empty else 0,
        "수익종목수": int(sum(1 for x in symbol_rows if x["실현손익KRW"] > 0)),
        "손실종목수": int(sum(1 for x in symbol_rows if x["실현손익KRW"] < 0)),
        "누적매수금액KRW": int(round(buy_amount)),
        "누적매도금액KRW": int(round(sell_amount)),
        "실현손익KRW": int(round(realized)),
        "누적매수금액대비수익률": round((realized / buy_amount * 100.0), 3) if buy_amount > 0 else 0.0,
        "일일예산1000만원대비수익률": round((realized / cfg.daily_budget_krw * 100.0), 3) if cfg.daily_budget_krw > 0 else 0.0,
    }

    c_summary = baseline.get("summary", {}) or {}
    c_pnl = int(c_summary.get("실현손익KRW", 0) or 0)
    d_pnl = int(summary.get("실현손익KRW", 0) or 0)

    payload = {
        "ok": True,
        "version": OPEN_DEFENSE_VERSION,
        "date": date_text,
        "strategy": "D_OPEN_DEFENSE",
        "comparison": {
            "C_NO_BUY2실현손익KRW": c_pnl,
            "D_OPEN_DEFENSE실현손익KRW": d_pnl,
            "D대비C순효과KRW": d_pnl - c_pnl,
            "C수익률": c_summary.get("일일예산1000만원대비수익률", 0),
            "D수익률": summary.get("일일예산1000만원대비수익률", 0),
        },
        "policy": {
            "base": "KR_C_NO_BUY2",
            "opening_window": f"{cfg.defense_start_time}~{cfg.defense_end_time} KST",
            "opening_first_size": "종목당 예산의 25%",
            "opening_confirm_size": "4분 이상 확인 후 추가 25%, 총 50%",
            "opening_max_positions": int(cfg.defense_max_positions),
            "early_fail_exit": f"장초반 방어포지션 -{abs(cfg.defense_hard_fail_pct):.2f}% 조기청산",
            "vwap_fail_exit": (
                f"보유 {cfg.vwap_fail_min_hold_minutes:.0f}분 이후 VWAP {cfg.vwap_fail_gap_pct:+.2f}% 이하 + "
                f"3분 {cfg.vwap_fail_ret3_pct:+.2f}% 이하 + 5분 {cfg.vwap_fail_ret5_pct:+.2f}% 이하"
            ),
            "chase_wait": (
                f"당일 +{cfg.chase_min_day_return_pct:.1f}% 이상 + VWAP +{cfg.chase_vwap_gap_pct:.1f}% 이상 + "
                f"거래량 {cfg.chase_max_volume_ratio:.2f}배 이하이면 즉시 추격 보류"
            ),
            "buy2": "OFF (OPEN_CONFIRM은 장초반 크기 복원용 1회 확인매수)",
            "after_opening_window": "기존 C의 50% BUY1 및 동일 청산규칙",
        },
        "diagnostic": diagnostic,
        "baseline_summary": c_summary,
        "d_summary": summary,
        "d_by_symbol": symbol_rows,
        "d_events": events,
        "config": asdict(cfg),
        "assumptions": {
            "real_orders": False,
            "data": "yfinance 1-minute historical bars",
            "candidate_reconstruction": "replay_kr와 동일한 65개 고정 유동성 종목군 근사",
            "future_data_visible": False,
            "slippage": f"매수 +{cfg.buy_slippage_pct:.2f}%, 매도 -{cfg.sell_slippage_pct:.2f}%",
            "fees_taxes": "별도 미포함",
            "important_limit": "과거 KIS 전체시장 실시간 거래량랭킹 원본이 없어 당시 전체시장 TOP5를 100% 복원한 것은 아님",
            "purpose": "D 전략 설계 검증용 리플레이. 실전/모의 주문에는 아직 연결하지 않음",
        },
        "cached": False,
    }

    if not codes:
        try:
            cache_path.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
        except Exception:
            pass
    return payload

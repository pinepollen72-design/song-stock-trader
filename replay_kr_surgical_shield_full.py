from __future__ import annotations

"""Read-only D-v2 SURGICAL SHIELD full-engine replay.

This version is deliberately profit-first. The previous GENTLE/FULL experiment cut
losses strongly but touched too many winning paths. SURGICAL keeps the original D-v2
profit engine and 50% size by default, intervening only on narrow loss-shaped setups.

Modes
- CONTROL: exact frozen D-v2 behavior for day-by-day parity.
- SURGICAL95: ultra-conservative shield, designed to maximize profit preservation.
- SURGICAL90: slightly stronger shield, accepted only if both profit-preservation
              measures remain at least 90%.

No real orders are sent. Only the already-cached Railway KIS 1-minute bars are read.
"""

import json
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Any

import pandas as pd

import replay_kr
import replay_kr_open_defense_v2 as d2

KST = d2.KST
SURGICAL_VERSION = "kr-d2-surgical-shield-full-engine-v1"

_THREAD: threading.Thread | None = None
_LOCK = threading.RLock()

# The expensive TOP5 reconstruction is identical for CONTROL/SURGICAL95/SURGICAL90.
# Cache it once per day/scan timestamp so the final 147-day test remains practical.
_TOP5_CACHE_DATE = ""
_TOP5_CACHE: dict[tuple[str, int], pd.DataFrame] = {}


def _make_c_cfg(cfg: d2.OpenDefenseConfig) -> replay_kr.KRReplayConfig:
    return replay_kr.KRReplayConfig(
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


def _top5_cached(
    target_frames: dict[str, pd.DataFrame],
    meta: dict[str, dict],
    date_text: str,
    now: pd.Timestamp,
    scan_count: int,
) -> pd.DataFrame:
    global _TOP5_CACHE_DATE, _TOP5_CACHE
    if _TOP5_CACHE_DATE != date_text:
        _TOP5_CACHE_DATE = date_text
        _TOP5_CACHE = {}
    key = (now.isoformat(), int(scan_count))
    value = _TOP5_CACHE.get(key)
    if value is None:
        value = replay_kr._build_top5_at(
            target_frames, meta, date_text, now, scan_count
        )
        if value is None:
            value = pd.DataFrame()
        _TOP5_CACHE[key] = value.copy()
    return value.copy()


def _hold_minutes(pos: dict, now: pd.Timestamp) -> float:
    try:
        created = pd.Timestamp(pos.get("created_at"))
        if created.tzinfo is None:
            created = created.tz_localize(KST)
        return max(0.0, (now - created).total_seconds() / 60.0)
    except Exception:
        return 0.0


def _row_state(row) -> dict:
    if row is None:
        return {
            "rank": 999,
            "ret3": -999.0,
            "ret5": -999.0,
            "score": 0.0,
            "weak": True,
            "signal": False,
            "volume": 0.0,
        }
    return {
        "rank": int(d2._rank_number(row)),
        "ret3": d2._safe_float(row.get("최근3분수익률", 0)),
        "ret5": d2._safe_float(row.get("최근5분수익률", 0)),
        "score": d2._safe_float(row.get("종합점수", 0)),
        "weak": bool(row.get("모멘텀약화", False)),
        "signal": "매수 후보" in str(row.get("판정", "")),
        "volume": d2._safe_float(row.get("거래량배수", 0)),
    }


def _fail_signals(cfg: d2.OpenDefenseConfig, state: dict, vwap_gap: float) -> list[str]:
    out: list[str] = []
    if vwap_gap <= float(cfg.soft_fail_vwap_gap_pct):
        out.append("BELOW_VWAP")
    if state["ret3"] <= float(cfg.soft_fail_ret3_pct):
        out.append("RET3_WEAK")
    if state["ret5"] <= float(cfg.soft_fail_ret5_pct):
        out.append("RET5_WEAK")
    if (
        state["score"] < float(cfg.soft_fail_score)
        or state["weak"]
        or not state["signal"]
    ):
        out.append("LEADER_STRENGTH_LOST")
    return out


def _is_a_plus(state: dict, vwap_gap: float) -> bool:
    return bool(
        state["signal"]
        and not state["weak"]
        and state["rank"] <= 3
        and state["score"] >= 65.0
        and 1.0 <= float(vwap_gap) <= 2.0
        and state["ret3"] > 0.0
        and state["ret5"] > 0.0
    )


def run_kr_surgical_shield_replay(
    date_text: str,
    codes: Iterable[str] | None = None,
    config: d2.OpenDefenseConfig | None = None,
    mode: str = "CONTROL",
) -> dict:
    """Run one day of D-v2 with a profit-preserving loss shield.

    CONTROL intentionally follows the proven full-engine D-v2 path exactly.
    SURGICAL95/90 change only narrow risk behavior; TAKE1, TAKE2 and the original
    PROFIT_GUARD thresholds remain unchanged.
    """
    cfg = config or d2.OpenDefenseConfig()
    mode = str(mode or "CONTROL").upper().strip()
    if mode not in {"CONTROL", "SURGICAL95", "SURGICAL90"}:
        raise ValueError("mode must be CONTROL, SURGICAL95, or SURGICAL90")
    shield = mode != "CONTROL"
    strict95 = mode == "SURGICAL95"
    balanced90 = mode == "SURGICAL90"
    # Legacy FULL-only branches are intentionally disabled in SURGICAL.
    full = False
    c_cfg = _make_c_cfg(cfg)

    universe = replay_kr._normalize_universe(codes)
    frames, meta = replay_kr._download_intraday(date_text, universe)
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
    defense_start_sec = replay_kr._clock_seconds(cfg.defense_start_time)
    defense_end_sec = replay_kr._clock_seconds(cfg.defense_end_time)
    last_entry_sec = replay_kr._clock_seconds(cfg.last_entry_time)
    force_exit_sec = replay_kr._clock_seconds(cfg.force_exit_time)
    late_1045_sec = replay_kr._clock_seconds("10:45")
    late_1300_sec = replay_kr._clock_seconds("13:00")

    normal_amount = d2._normal_entry_amount(cfg)       # current D-v2 50%
    defense_amount = d2._defense_entry_amount(cfg)     # current D-v2 25%
    confirm_amount = max(0, normal_amount - defense_amount)

    positions: dict[str, dict] = {}
    events: list[dict] = []
    latest_top5 = pd.DataFrame()
    last_scan = None
    daily_buy_amount = 0.0
    daily_orders = 0

    # PG2 graduated re-arm state: a freshly rolled-over name cannot immediately
    # receive full risk again. After 10m it may probe at 25%; proven strength re-arms 50%.
    pg2_disarmed_at: dict[str, pd.Timestamp] = {}
    pg2_block_seen: set[tuple[str, str]] = set()

    # FULL-only daily risk state: after a normal BUY1 final stop/emergency, ordinary
    # subsequent BUY1s use 25% until a winning exit resets normal size. A+ setups are
    # never size-cut by this global state.
    risk_reduced_after_stop = False

    diagnostic = {
        "strong_open_entries": 0,
        "defense_entries": 0,
        "normal_entries": 0,
        "open_confirms": 0,
        "open_emergency_exits": 0,
        "open_soft_fail_exits": 0,
        "profit_guard_grace_blocks": 0,
        "chase_waits": 0,
        "defense_position_cap_blocks": 0,
        # Profit-preserve shield diagnostics
        "adaptive_25pct_entries": 0,
        "adaptive_vwap_below_entries": 0,
        "adaptive_overheat_entries": 0,
        "adaptive_late_entries": 0,
        "adaptive_post_stop_entries": 0,
        "open_strong_caution_entries": 0,
        "normal_early_partial_exits": 0,
        "normal_early_partial_small_qty_skips": 0,
        "normal_early_emergency_exits": 0,
        "normal_stop_losses": 0,
        "pre_guard_exits": 0,
        "pre_guard_small_qty_skips": 0,
        "pg2_disarm_activations": 0,
        "pg2_hard_block_signals": 0,
        "pg2_probe_entries": 0,
        "pg2_rearm_successes": 0,
        "risk_state_activations": 0,
        "risk_state_resets": 0,
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

    def reset_risk_state_if_needed():
        nonlocal risk_reduced_after_stop
        if full and risk_reduced_after_stop:
            risk_reduced_after_stop = False
            diagnostic["risk_state_resets"] += 1

    def activate_risk_state_if_needed():
        nonlocal risk_reduced_after_stop
        if full and not risk_reduced_after_stop:
            risk_reduced_after_stop = True
            diagnostic["risk_state_activations"] += 1

    now = start
    while now <= end:
        if last_scan is None or (now - last_scan).total_seconds() >= int(cfg.scan_seconds):
            latest_top5 = _top5_cached(
                target_frames, meta, date_text, now, cfg.scan_count
            )
            last_scan = now

        top5_map: dict[str, Any] = {}
        if latest_top5 is not None and not latest_top5.empty:
            for _, r in latest_top5.iterrows():
                top5_map[str(r.get("종목코드", "")).zfill(6)] = r

        # ------------------------------------------------------------------
        # 1) Existing-position management
        # ------------------------------------------------------------------
        for symbol, pos in list(positions.items()):
            frame = target_frames.get(symbol)
            if frame is None:
                continue
            ref_price = replay_kr._price_at(frame, date_text, now)
            if ref_price <= 0:
                continue
            qty = int(pos.get("qty", 0))
            avg = d2._safe_float(pos.get("avg_price", 0))
            if qty <= 0 or avg <= 0:
                continue

            pnl = (ref_price / avg - 1.0) * 100.0
            peak = max(d2._safe_float(pos.get("peak_pnl", pnl)), pnl)
            pos["peak_pnl"] = peak
            dd = max(0.0, peak - pnl)
            vg = d2._vwap_gap_pct(frame, date_text, now, ref_price)
            row = top5_map.get(symbol)
            state = _row_state(row)

            # Normal force exit: unchanged.
            if replay_kr._seconds_of_day(now) >= force_exit_sec:
                fill = replay_kr._fill_price(c_cfg, "SELL", ref_price)
                realized = (fill - avg) * qty
                add_event(
                    now, symbol, "FORCE_SELL", "SELL", qty,
                    ref_price, fill, f"당일 강제청산 {cfg.force_exit_time} KST",
                    pnl, realized, vwap_gap=vg,
                )
                positions.pop(symbol, None)
                continue

            # Existing D-v2 OPEN DEFENSE early-failure logic: unchanged.
            if bool(pos.get("defense_position", False)):
                hold_min = _hold_minutes(pos, now)
                if hold_min <= float(cfg.defense_fail_window_minutes):
                    if pnl <= float(cfg.defense_emergency_fail_pct):
                        fill = replay_kr._fill_price(c_cfg, "SELL", ref_price)
                        realized = (fill - avg) * qty
                        add_event(
                            now, symbol, "OPEN_EMERGENCY_EXIT", "SELL", qty,
                            ref_price, fill,
                            f"장초반 비상청산 · pnl {pnl:.2f}% <= {cfg.defense_emergency_fail_pct:.2f}%",
                            pnl, realized, vwap_gap=vg,
                        )
                        diagnostic["open_emergency_exits"] += 1
                        positions.pop(symbol, None)
                        continue

                    if (
                        hold_min >= float(cfg.soft_fail_min_hold_minutes)
                        and pnl <= float(cfg.defense_soft_fail_pct)
                    ):
                        fail = _fail_signals(cfg, state, vg)
                        if len(fail) >= int(cfg.soft_fail_min_signals):
                            fill = replay_kr._fill_price(c_cfg, "SELL", ref_price)
                            realized = (fill - avg) * qty
                            add_event(
                                now, symbol, "OPEN_SOFT_FAIL_EXIT", "SELL", qty,
                                ref_price, fill,
                                f"복합 돌파실패 · pnl {pnl:.2f}% · " + ",".join(fail),
                                pnl, realized, vwap_gap=vg,
                            )
                            diagnostic["open_soft_fail_exits"] += 1
                            positions.pop(symbol, None)
                            continue

            # SURGICAL: no partial trim and no broad -1% intervention.
            # One earlier full exit is allowed only after a deep loss plus structural
            # failure. This uses no extra sell order versus the eventual STOP_LOSS.
            is_normal_buy1 = str(pos.get("entry_action", "")) == "BUY1"
            if shield and is_normal_buy1:
                hold_n = _hold_minutes(pos, now)
                strict_fail: list[str] = []
                if vg <= -0.50:
                    strict_fail.append("VWAP_BREAK")
                if state["ret3"] <= -0.40:
                    strict_fail.append("RET3_BREAK")
                if state["ret5"] <= -0.30:
                    strict_fail.append("RET5_BREAK")
                if (
                    state["score"] < 55.0
                    or state["weak"]
                    or not state["signal"]
                    or state["rank"] > 5
                ):
                    strict_fail.append("LEADER_LOST")

                # A trade that already showed meaningful favorable excursion gets
                # extra room. This is the core winner-protection rule.
                had_positive_excursion = peak >= 0.80
                if strict95:
                    loss_line = -2.40 if not had_positive_excursion else -2.70
                    min_hold = 5.0
                    needed = 3 if not had_positive_excursion else 4
                else:
                    loss_line = -2.10 if not had_positive_excursion else -2.50
                    min_hold = 4.0
                    needed = 3 if not had_positive_excursion else 4

                if hold_n >= min_hold and pnl <= loss_line and len(strict_fail) >= needed:
                    fill = replay_kr._fill_price(c_cfg, "SELL", ref_price)
                    realized = (fill - avg) * qty
                    add_event(
                        now, symbol, "NORMAL_SURGICAL_EXIT", "SELL", qty,
                        ref_price, fill,
                        f"일반 BUY1 수술형 조기종료 · 보유 {hold_n:.1f}분 · "
                        f"pnl {pnl:.2f}% <= {loss_line:.2f}% · " + ",".join(strict_fail),
                        pnl, realized, vwap_gap=vg,
                    )
                    diagnostic["normal_early_emergency_exits"] += 1
                    positions.pop(symbol, None)
                    continue

                if had_positive_excursion and pnl <= (-2.10 if balanced90 else -2.40):
                    diagnostic["normal_early_partial_small_qty_skips"] += 1

            # Original -3% stop remains the final fallback for every position.
            if pnl <= -abs(cfg.stop_loss_pct):
                fill = replay_kr._fill_price(c_cfg, "SELL", ref_price)
                realized = (fill - avg) * qty
                add_event(
                    now, symbol, "STOP_LOSS", "SELL", qty,
                    ref_price, fill, f"손절 {pnl:.2f}%",
                    pnl, realized, vwap_gap=vg,
                )
                if is_normal_buy1:
                    diagnostic["normal_stop_losses"] += 1
                    activate_risk_state_if_needed()
                positions.pop(symbol, None)
                continue

            # OPEN_CONFIRM is unchanged. OPEN_STRONG caution positions deliberately use
            # defense_position=True, so a survivor can restore the original 50% size.
            if (
                bool(pos.get("defense_position", False))
                and not bool(pos.get("open_confirmed", False))
                and replay_kr._seconds_of_day(now) < defense_end_sec
                and daily_orders < cfg.max_daily_orders
            ):
                ok, why = d2._confirm_allowed(cfg, pos, row, now, pnl, vg)
                if ok and confirm_amount > 0:
                    fill = replay_kr._fill_price(c_cfg, "BUY", ref_price)
                    qty2 = int(confirm_amount // fill)
                    cost = fill * qty2
                    if qty2 > 0 and daily_buy_amount + cost <= cfg.daily_budget_krw:
                        old_qty = int(pos["qty"])
                        old_avg = d2._safe_float(pos["avg_price"])
                        new_qty = old_qty + qty2
                        new_avg = (old_avg * old_qty + fill * qty2) / new_qty
                        score = state["score"] if row is not None else ""
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
                        continue

            # Original profit engine is unchanged.
            if pnl >= cfg.take1_pct and not bool(pos.get("take1_sent")):
                sell_qty = max(1, qty // 2)
                fill = replay_kr._fill_price(c_cfg, "SELL", ref_price)
                realized = (fill - avg) * sell_qty
                add_event(
                    now, symbol, "TAKE1", "SELL", sell_qty,
                    ref_price, fill, f"1차 익절 {pnl:.2f}% · 약 50%",
                    pnl, realized, vwap_gap=vg,
                )
                pos["qty"] = qty - sell_qty
                pos["take1_sent"] = True
                pg2_disarmed_at.pop(symbol, None)
                reset_risk_state_if_needed()
                if pos["qty"] <= 0:
                    positions.pop(symbol, None)
                continue

            if pnl >= cfg.take2_pct and bool(pos.get("take1_sent")):
                fill = replay_kr._fill_price(c_cfg, "SELL", ref_price)
                realized = (fill - avg) * qty
                add_event(
                    now, symbol, "TAKE2", "SELL", qty,
                    ref_price, fill, f"2차 익절 {pnl:.2f}% · 전량",
                    pnl, realized, vwap_gap=vg,
                )
                pg2_disarmed_at.pop(symbol, None)
                reset_risk_state_if_needed()
                positions.pop(symbol, None)
                continue

            suppress_profit_guard = False
            if bool(pos.get("defense_position", False)) and not bool(pos.get("open_confirmed", False)):
                suppress_profit_guard = (
                    _hold_minutes(pos, now) < float(cfg.profit_guard_confirm_grace_minutes)
                )

            if peak >= cfg.profit_guard_trigger_pct and dd >= cfg.profit_guard_drawdown_pct:
                if suppress_profit_guard:
                    diagnostic["profit_guard_grace_blocks"] += 1
                    continue

                if not bool(pos.get("take1_sent")):
                    sell_qty = max(1, qty // 2)
                    fill = replay_kr._fill_price(c_cfg, "SELL", ref_price)
                    realized = (fill - avg) * sell_qty
                    add_event(
                        now, symbol, "PROFIT_GUARD1", "SELL", sell_qty,
                        ref_price, fill,
                        f"수익보호 1차 · 최고 +{peak:.2f}% → 현재 {pnl:.2f}% ({dd:.2f}%p 되밀림)",
                        pnl, realized, vwap_gap=vg,
                    )
                    pos["qty"] = qty - sell_qty
                    pos["take1_sent"] = True
                    if pnl > 0:
                        reset_risk_state_if_needed()
                    if pos["qty"] <= 0:
                        positions.pop(symbol, None)
                    continue
                else:
                    fill = replay_kr._fill_price(c_cfg, "SELL", ref_price)
                    realized = (fill - avg) * qty
                    add_event(
                        now, symbol, "PROFIT_GUARD2", "SELL", qty,
                        ref_price, fill,
                        f"수익보호 2차 · 최고 +{peak:.2f}% → 현재 {pnl:.2f}% ({dd:.2f}%p 되밀림)",
                        pnl, realized, vwap_gap=vg,
                    )
                    if shield:
                        pg2_disarmed_at[symbol] = now
                        diagnostic["pg2_disarm_activations"] += 1
                    if pnl > 0:
                        reset_risk_state_if_needed()
                    positions.pop(symbol, None)
                    continue

        # ------------------------------------------------------------------
        # 2) New entries
        # ------------------------------------------------------------------
        if (
            replay_kr._seconds_of_day(now) < last_entry_sec
            and len(positions) < cfg.max_positions
            and daily_orders < cfg.max_daily_orders
            and latest_top5 is not None
            and not latest_top5.empty
        ):
            sec = replay_kr._seconds_of_day(now)
            defense_now = defense_start_sec <= sec < defense_end_sec

            for _, row in latest_top5.iterrows():
                if len(positions) >= cfg.max_positions or daily_orders >= cfg.max_daily_orders:
                    break

                symbol = str(row.get("종목코드", "")).zfill(6)
                if symbol in positions:
                    continue
                signal = str(row.get("판정", ""))
                score = d2._safe_float(row.get("종합점수", 0))
                weak = bool(row.get("모멘텀약화", False))
                if "매수 후보" not in signal or weak or score < cfg.min_score:
                    continue

                frame = target_frames.get(symbol)
                if frame is None:
                    continue
                ref_price = replay_kr._price_at(frame, date_text, now)
                if ref_price <= 0:
                    continue
                vg = d2._vwap_gap_pct(frame, date_text, now, ref_price)
                state = _row_state(row)

                # Opening window keeps D-v2 behavior, except a would-be strong open
                # that is already >+2.5% above VWAP starts at 25% instead of 50%.
                # It remains eligible for normal OPEN_CONFIRM, preserving upside.
                if defense_now:
                    strong_open, _ = d2._is_strong_open(cfg, row, vg)
                    if strong_open:
                        caution_gap = 3.00 if strict95 else 2.50
                        if shield and vg >= caution_gap:
                            target_amount = defense_amount
                            action = "OPEN_STRONG_CAUTION_BUY"
                            defense_position = True
                            diagnostic["open_strong_caution_entries"] += 1
                        else:
                            target_amount = normal_amount
                            action = "OPEN_STRONG_BUY"
                            defense_position = False
                    else:
                        defense_count = sum(
                            1 for p in positions.values()
                            if bool(p.get("defense_position", False))
                        )
                        if defense_count >= int(cfg.defense_max_positions):
                            diagnostic["defense_position_cap_blocks"] += 1
                            continue
                        wait, _ = d2._is_chase_wait(cfg, row, vg)
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
                    action = "BUY1"
                    defense_position = False
                    target_amount = normal_amount

                    if shield:
                        risk_reasons: list[str] = []

                        # PG2: short hard lock, then require a clean re-arm before
                        # taking full risk again. TAKE2 re-entry is untouched.
                        disarmed = pg2_disarmed_at.get(symbol)
                        if disarmed is not None:
                            elapsed = max(0.0, (now - disarmed).total_seconds() / 60.0)
                            hard_lock = 5.0 if strict95 else 7.0
                            rearmed = bool(
                                elapsed >= hard_lock
                                and state["rank"] <= 3
                                and vg >= 0.0
                                and state["ret3"] > 0.0
                                and state["ret5"] > 0.0
                                and not state["weak"]
                                and state["signal"]
                            )
                            if elapsed < hard_lock:
                                key = (symbol, now.isoformat())
                                if key not in pg2_block_seen:
                                    pg2_block_seen.add(key)
                                    diagnostic["pg2_hard_block_signals"] += 1
                                continue
                            if rearmed:
                                pg2_disarmed_at.pop(symbol, None)
                                diagnostic["pg2_rearm_successes"] += 1
                            else:
                                # Do not create a small probe that can alter the order
                                # path. Wait for a proper re-arm instead.
                                continue

                        # Default remains the original 50%. Only narrow loss-shaped
                        # entry states are reduced to 25%. No time-of-day penalty.
                        if strict95:
                            below_risk = bool(
                                vg <= -0.50
                                and (state["ret3"] <= 0.20 or state["ret5"] <= 0.30)
                            )
                            overheat_risk = bool(
                                vg >= 3.00
                                and not (
                                    state["ret3"] >= 1.50
                                    and state["ret5"] >= 1.50
                                    and state["volume"] >= 1.50
                                )
                            )
                        else:
                            below_risk = bool(
                                vg < 0.0
                                and (state["ret3"] <= 0.35 or state["ret5"] <= 0.40)
                            )
                            overheat_risk = bool(
                                vg >= 2.80
                                and not (
                                    state["ret3"] >= 1.50
                                    and state["ret5"] >= 1.50
                                )
                            )

                        if below_risk:
                            risk_reasons.append("SELECTIVE_VWAP_BELOW")
                        if overheat_risk:
                            risk_reasons.append("SELECTIVE_OVERHEAT")

                        if risk_reasons:
                            target_amount = defense_amount
                            diagnostic["adaptive_25pct_entries"] += 1
                            if "SELECTIVE_VWAP_BELOW" in risk_reasons:
                                diagnostic["adaptive_vwap_below_entries"] += 1
                            if "SELECTIVE_OVERHEAT" in risk_reasons:
                                diagnostic["adaptive_overheat_entries"] += 1

                fill = replay_kr._fill_price(c_cfg, "BUY", ref_price)
                qty1 = int(target_amount // fill)
                if qty1 <= 0:
                    continue
                cost = fill * qty1
                if daily_buy_amount + cost > cfg.daily_budget_krw:
                    continue

                rank = str(row.get("순위", ""))
                r3 = state["ret3"]
                r5 = state["ret5"]
                vr = state["volume"]
                if action == "OPEN_STRONG_BUY":
                    prefix = "OPEN STRONG 50%"
                elif action == "OPEN_STRONG_CAUTION_BUY":
                    prefix = "OPEN STRONG CAUTION 25%"
                elif action == "OPEN_DEFENSE_BUY":
                    prefix = "OPEN DEFENSE 25%"
                elif target_amount == defense_amount and shield:
                    prefix = "PROFIT-PRESERVE BUY1 25%"
                else:
                    prefix = "C기준 50%"

                add_event(
                    now, symbol, action, "BUY", qty1,
                    ref_price, fill,
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
                    "entry_action": action,
                    "early_fail_reduced": False,
                    "pre_guard_sent": False,
                }
                daily_buy_amount += cost

                if action == "OPEN_STRONG_BUY":
                    diagnostic["strong_open_entries"] += 1
                elif action in {"OPEN_DEFENSE_BUY", "OPEN_STRONG_CAUTION_BUY"}:
                    diagnostic["defense_entries"] += 1
                else:
                    diagnostic["normal_entries"] += 1

        now += pd.Timedelta(seconds=int(cfg.manage_seconds))

    # Safety liquidation if anything somehow remains after the normal force-exit loop.
    if positions:
        for symbol, pos in list(positions.items()):
            frame = target_frames.get(symbol)
            ref_price = replay_kr._price_at(frame, date_text, end) if frame is not None else 0.0
            if ref_price <= 0:
                continue
            qty = int(pos.get("qty", 0))
            avg = d2._safe_float(pos.get("avg_price", 0))
            fill = replay_kr._fill_price(c_cfg, "SELL", ref_price)
            pnl = (ref_price / avg - 1.0) * 100.0 if avg > 0 else 0.0
            realized = (fill - avg) * qty
            vg = d2._vwap_gap_pct(frame, date_text, end, ref_price)
            add_event(
                end, symbol, "FORCE_SELL_END", "SELL", qty,
                ref_price, fill, "리플레이 종료 안전청산",
                pnl, realized, vwap_gap=vg,
            )
            positions.pop(symbol, None)

    events_df = pd.DataFrame(events)
    if events_df.empty:
        buy_amount = sell_amount = realized = 0.0
    else:
        buy_amount = float(events_df.loc[events_df["구분"] == "BUY", "주문금액KRW"].sum())
        sell_amount = float(events_df.loc[events_df["구분"] == "SELL", "주문금액KRW"].sum())
        realized = float(events_df["실현손익KRW"].sum())

    summary = {
        "총주문횟수": int(len(events_df)),
        "매수주문횟수": int((events_df["구분"] == "BUY").sum()) if not events_df.empty else 0,
        "매도주문횟수": int((events_df["구분"] == "SELL").sum()) if not events_df.empty else 0,
        "거래종목수": int(events_df["종목코드"].nunique()) if not events_df.empty else 0,
        "누적매수금액KRW": int(round(buy_amount)),
        "누적매도금액KRW": int(round(sell_amount)),
        "실현손익KRW": int(round(realized)),
        "누적매수금액대비수익률": round((realized / buy_amount * 100.0), 3) if buy_amount > 0 else 0.0,
        "일일예산1000만원대비수익률": round((realized / cfg.daily_budget_krw * 100.0), 3)
        if cfg.daily_budget_krw > 0 else 0.0,
    }

    return {
        "ok": True,
        "version": SURGICAL_VERSION,
        "date": date_text,
        "strategy": f"D2_PROFIT_PRESERVE_{mode}",
        "mode": mode,
        "diagnostic": diagnostic,
        "summary": summary,
        "events": events,
        "config": asdict(cfg),
        "rules": {
            "profit_engine_unchanged": {
                "take1_pct": cfg.take1_pct,
                "take2_pct": cfg.take2_pct,
                "profit_guard_trigger_pct": cfg.profit_guard_trigger_pct,
                "profit_guard_drawdown_pct": cfg.profit_guard_drawdown_pct,
            },
            "surgical": {
                "default_size": "original BUY1 50% remains default; no late-time size penalty",
                "entry_reduction": "25% only on selective below-VWAP weakness or weak overheat",
                "normal_loss_exit": "single full exit only on deep loss + >=3 strict structural failures; no partial trim",
                "winner_protection": "peak >= +0.8% requires deeper loss and all 4 failures before surgical exit",
                "PG2": "5m(S95)/7m(S90) hard lock, then full 50% only after TOP3+VWAP+3m+5m re-arm",
                "open_strong_overheat": "S95 >=+3.0%, S90 >=+2.5% VWAP starts 25% and may OPEN_CONFIRM back to 50%",
                "removed": ["late-time penalty", "PRE_GUARD", "post-stop risk state", "-1% partial trim"],
            },
        },
        "assumptions": {
            "real_orders": False,
            "future_data_visible": False,
            "fees_taxes": "별도 미포함",
            "purpose": "surgical D-v3 loss reduction full-engine validation",
        },
    }


# =============================================================================
# 147-day orchestrator. Imported/called by replay_kr_long_backtest only.
# =============================================================================

def _base():
    # Delayed import avoids circular import: replay_kr_long_backtest imports this module.
    import replay_kr_long_backtest as base
    return base


def _paths():
    base = _base()
    root = base.ROOT / "surgical_shield_full_engine"
    day_dir = root / "daily"
    state_file = root / "state.json"
    result_file = root / "result.json"
    root.mkdir(parents=True, exist_ok=True)
    day_dir.mkdir(parents=True, exist_ok=True)
    return root, day_dir, state_file, result_file


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path, default=None):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _state(**updates) -> dict:
    _, _, state_file, _ = _paths()
    with _LOCK:
        cur = _read_json(state_file, {}) or {}
        cur.update(updates)
        cur.setdefault("version", SURGICAL_VERSION)
        cur["updated_at"] = datetime.now(KST).isoformat(timespec="seconds")
        _atomic_write_json(state_file, cur)
        return cur


def _public_state() -> dict:
    _, _, state_file, _ = _paths()
    state = _read_json(state_file, {}) or {}
    if not state:
        state = {
            "ok": True,
            "version": SURGICAL_VERSION,
            "status": "not_started",
            "result_ready": False,
        }
    out = dict(state)
    out["ok"] = True
    out["thread_alive"] = bool(_THREAD and _THREAD.is_alive())
    return out


def _day_path(date_text: str) -> Path:
    _, day_dir, _, _ = _paths()
    return day_dir / f"{date_text}.json.gz"


def _pack(x: dict) -> dict:
    sm = x.get("summary", {}) or {}
    events = [e for e in list(x.get("events", []) or []) if isinstance(e, dict)]
    realized = [int(e.get("실현손익KRW", 0) or 0) for e in events]
    actions: dict[str, int] = {}
    for e in events:
        a = str(e.get("액션", "") or "")
        if a:
            actions[a] = actions.get(a, 0) + 1
    return {
        "pnl_KRW": int(sm.get("실현손익KRW", 0) or 0),
        "buy_orders": int(sm.get("매수주문횟수", 0) or 0),
        "sell_orders": int(sm.get("매도주문횟수", 0) or 0),
        "total_orders": int(sm.get("총주문횟수", 0) or 0),
        "traded_symbols": int(sm.get("거래종목수", 0) or 0),
        "gross_profit_KRW": int(sum(v for v in realized if v > 0)),
        "gross_loss_KRW": int(sum(v for v in realized if v < 0)),
        "take1_events": int(actions.get("TAKE1", 0)),
        "take2_events": int(actions.get("TAKE2", 0)),
        "profit_guard1_events": int(actions.get("PROFIT_GUARD1", 0)),
        "profit_guard2_events": int(actions.get("PROFIT_GUARD2", 0)),
        "stop_loss_events": int(actions.get("STOP_LOSS", 0)),
        "diagnostic": x.get("diagnostic", {}) or {},
    }


def _aggregate(label: str, key: str, rows: list[dict]) -> dict:
    base = _base()
    vals = [int((r.get(key) or {}).get("pnl_KRW", 0) or 0) for r in rows]
    total = int(sum(vals))
    mdd, _, _ = base._max_drawdown(vals)
    diag_keys = [
        "adaptive_25pct_entries",
        "adaptive_vwap_below_entries",
        "adaptive_overheat_entries",
        "adaptive_late_entries",
        "adaptive_post_stop_entries",
        "open_strong_caution_entries",
        "normal_early_partial_exits",
        "normal_early_emergency_exits",
        "normal_stop_losses",
        "pre_guard_exits",
        "pg2_disarm_activations",
        "pg2_hard_block_signals",
        "pg2_probe_entries",
        "pg2_rearm_successes",
        "risk_state_activations",
        "risk_state_resets",
    ]
    diag = {
        k: int(sum(int(((r.get(key) or {}).get("diagnostic") or {}).get(k, 0) or 0) for r in rows))
        for k in diag_keys
    }
    return {
        "id": key,
        "label": label,
        "total_KRW": total,
        "average_daily_KRW": round(total / len(vals), 1) if vals else 0.0,
        "positive_days": int(sum(1 for v in vals if v > 0)),
        "negative_days": int(sum(1 for v in vals if v < 0)),
        "max_cumulative_drawdown_KRW": int(mdd),
        "buy_orders": int(sum(int((r.get(key) or {}).get("buy_orders", 0) or 0) for r in rows)),
        "total_orders": int(sum(int((r.get(key) or {}).get("total_orders", 0) or 0) for r in rows)),
        "gross_profit_KRW": int(sum(int((r.get(key) or {}).get("gross_profit_KRW", 0) or 0) for r in rows)),
        "gross_loss_KRW": int(sum(int((r.get(key) or {}).get("gross_loss_KRW", 0) or 0) for r in rows)),
        "take1_events": int(sum(int((r.get(key) or {}).get("take1_events", 0) or 0) for r in rows)),
        "take2_events": int(sum(int((r.get(key) or {}).get("take2_events", 0) or 0) for r in rows)),
        "profit_guard1_events": int(sum(int((r.get(key) or {}).get("profit_guard1_events", 0) or 0) for r in rows)),
        "profit_guard2_events": int(sum(int((r.get(key) or {}).get("profit_guard2_events", 0) or 0) for r in rows)),
        "stop_loss_events": int(sum(int((r.get(key) or {}).get("stop_loss_events", 0) or 0) for r in rows)),
        **diag,
    }


def _profit_preservation(rows: list[dict], key: str) -> dict:
    control_profit = 0
    sacrificed = 0
    gained_on_positive = 0
    loss_reduction = 0
    loss_worsening = 0
    positive_days_hurt = 0
    negative_days_improved = 0

    for r in rows:
        c = int((r.get("CONTROL_D2") or {}).get("pnl_KRW", 0) or 0)
        v = int((r.get(key) or {}).get("pnl_KRW", 0) or 0)
        if c > 0:
            control_profit += c
            if v < c:
                sacrificed += c - v
                positive_days_hurt += 1
            else:
                gained_on_positive += v - c
        elif c < 0:
            if v > c:
                loss_reduction += v - c
                negative_days_improved += 1
            elif v < c:
                loss_worsening += c - v

    preserved = max(0, control_profit - sacrificed)
    pct = round(100.0 * preserved / control_profit, 2) if control_profit > 0 else 100.0
    control_gross_profit = int(sum(int((r.get("CONTROL_D2") or {}).get("gross_profit_KRW", 0) or 0) for r in rows))
    variant_gross_profit = int(sum(int((r.get(key) or {}).get("gross_profit_KRW", 0) or 0) for r in rows))
    gross_pct = round(100.0 * variant_gross_profit / control_gross_profit, 2) if control_gross_profit > 0 else 100.0
    return {
        "control_positive_day_profit_KRW": int(control_profit),
        "profit_sacrificed_on_control_positive_days_KRW": int(sacrificed),
        "extra_profit_on_control_positive_days_KRW": int(gained_on_positive),
        "profit_preservation_pct": pct,
        "control_gross_profit_KRW": control_gross_profit,
        "variant_gross_profit_KRW": variant_gross_profit,
        "gross_profit_preservation_pct": gross_pct,
        "control_positive_days_hurt": int(positive_days_hurt),
        "loss_reduction_on_control_negative_days_KRW": int(loss_reduction),
        "loss_worsening_on_control_negative_days_KRW": int(loss_worsening),
        "control_negative_days_improved": int(negative_days_improved),
        "passes_90pct_profit_preservation": bool(pct >= 90.0 and gross_pct >= 90.0),
        "passes_95pct_profit_preservation": bool(pct >= 95.0 and gross_pct >= 95.0),
    }


def _monthly(rows: list[dict], key: str) -> list[dict]:
    out: dict[str, dict] = {}
    for r in rows:
        month = str(r.get("date", ""))[:7]
        if not month:
            continue
        c = int((r.get("CONTROL_D2") or {}).get("pnl_KRW", 0) or 0)
        v = int((r.get(key) or {}).get("pnl_KRW", 0) or 0)
        m = out.setdefault(month, {
            "month": month,
            "days": 0,
            "control_KRW": 0,
            "variant_KRW": 0,
            "delta_KRW": 0,
        })
        m["days"] += 1
        m["control_KRW"] += c
        m["variant_KRW"] += v
        m["delta_KRW"] += v - c
    return [out[k] for k in sorted(out)]


def _job(result: dict) -> None:
    base = _base()
    _, _, _, result_file = _paths()
    try:
        daily_base = [
            r for r in list(result.get("daily", []) or [])
            if isinstance(r, dict) and r.get("date")
        ]
        daily_base.sort(key=lambda r: str(r.get("date")))
        if not daily_base:
            raise RuntimeError("기존 147일 daily 결과가 없습니다.")

        cfg = base._runner_frozen_config(result)
        codes = base._codes()

        # Exact same cached KIS bars; no new KIS/API calls.
        replay_kr._download_intraday = base._cached_only_provider
        d2._download_intraday = base._cached_only_provider

        _state(
            status="running",
            phase="SURGICAL_SHIELD_FULL_ENGINE",
            started_at=datetime.now(KST).isoformat(timespec="seconds"),
            total_days=len(daily_base),
            completed_days=0,
            progress_pct=0.0,
            result_ready=False,
            message="CONTROL / SURGICAL95 / SURGICAL90 수술형 손실방어 전체엔진 비교 시작",
            last_error="",
        )

        rows: list[dict] = []
        errors: list[dict] = []
        parity_mismatches: list[dict] = []

        for idx, base_row in enumerate(daily_base, start=1):
            date_text = str(base_row.get("date"))

            while True:
                live, label = base._in_protected_live_window()
                if not live:
                    break
                _state(
                    status="paused_live_window",
                    phase="PAUSED",
                    pause_reason=label,
                    current_date=date_text,
                    completed_days=len(rows),
                    total_days=len(daily_base),
                    message="실시간 자동매매 보호를 위해 수익보존형 전체엔진 검증 일시정지",
                )
                time.sleep(30.0)

            _state(
                status="running",
                phase="SURGICAL_SHIELD_FULL_ENGINE",
                current_date=date_text,
                completed_days=len(rows),
                total_days=len(daily_base),
                progress_pct=round(100.0 * (idx - 1) / len(daily_base), 1),
                message=f"수술형 전체엔진 {idx}/{len(daily_base)} · {date_text}",
            )

            day_path = _day_path(date_text)
            cached = base._load_gzip_json(day_path, None)
            if isinstance(cached, dict) and cached.get("version") == SURGICAL_VERSION:
                row = cached
            else:
                try:
                    control = run_kr_surgical_shield_replay(
                        date_text=date_text, codes=codes, config=cfg, mode="CONTROL"
                    )
                    s95 = run_kr_surgical_shield_replay(
                        date_text=date_text, codes=codes, config=cfg, mode="SURGICAL95"
                    )
                    s90 = run_kr_surgical_shield_replay(
                        date_text=date_text, codes=codes, config=cfg, mode="SURGICAL90"
                    )
                    row = {
                        "version": SURGICAL_VERSION,
                        "date": date_text,
                        "cached_D2_KRW": int(base_row.get("D2_KRW", 0) or 0),
                        "CONTROL_D2": _pack(control),
                        "SURGICAL95": _pack(s95),
                        "SURGICAL90": _pack(s90),
                    }
                    row["parity_delta_KRW"] = (
                        int(row["CONTROL_D2"]["pnl_KRW"]) - int(row["cached_D2_KRW"])
                    )
                    for key in ("SURGICAL95", "SURGICAL90"):
                        row[key]["delta_vs_control_KRW"] = (
                            int(row[key]["pnl_KRW"]) - int(row["CONTROL_D2"]["pnl_KRW"])
                        )
                    base._save_gzip_json(day_path, row)
                except Exception as exc:
                    err = {"date": date_text, "error": f"{type(exc).__name__}: {exc}"}
                    errors.append(err)
                    _state(last_error=err["error"][:1000])
                    continue

            rows.append(row)
            parity_delta = int(
                row.get(
                    "parity_delta_KRW",
                    int((row.get("CONTROL_D2") or {}).get("pnl_KRW", 0))
                    - int(row.get("cached_D2_KRW", 0)),
                )
            )
            if parity_delta != 0:
                parity_mismatches.append({
                    "date": date_text,
                    "cached_D2_KRW": int(row.get("cached_D2_KRW", 0) or 0),
                    "control_D2_KRW": int((row.get("CONTROL_D2") or {}).get("pnl_KRW", 0) or 0),
                    "delta_KRW": parity_delta,
                })

            _state(
                status="running",
                phase="SURGICAL_SHIELD_FULL_ENGINE",
                current_date=date_text,
                completed_days=len(rows),
                error_days=len(errors),
                total_days=len(daily_base),
                progress_pct=round(100.0 * idx / len(daily_base), 1),
                message=f"수술형 전체엔진 {idx}/{len(daily_base)} 완료",
            )

        control = _aggregate("D-v2 원본 전체엔진", "CONTROL_D2", rows)
        s95 = _aggregate(
            "SURGICAL95 · 수익보존 최우선 수술형 방어",
            "SURGICAL95", rows,
        )
        s90 = _aggregate(
            "SURGICAL90 · 90% 수익보존 하한의 균형형 방어",
            "SURGICAL90", rows,
        )
        for v in (s95, s90):
            v["delta_vs_control_KRW"] = int(v["total_KRW"] - control["total_KRW"])

        expected_total = int(((result.get("overall") or {}).get("D2_total_KRW", 0)) or 0)
        parity_ok = bool(
            not parity_mismatches
            and not errors
            and len(rows) == len(daily_base)
            and int(control["total_KRW"]) == expected_total
        )

        def day_delta(row: dict, key: str) -> dict:
            c = int((row.get("CONTROL_D2") or {}).get("pnl_KRW", 0) or 0)
            v = int((row.get(key) or {}).get("pnl_KRW", 0) or 0)
            return {
                "date": row.get("date"),
                "control_KRW": c,
                "variant_KRW": v,
                "delta_KRW": v - c,
            }

        preservation = {
            "SURGICAL95": _profit_preservation(rows, "SURGICAL95"),
            "SURGICAL90": _profit_preservation(rows, "SURGICAL90"),
        }
        monthly = {
            "SURGICAL95": _monthly(rows, "SURGICAL95"),
            "SURGICAL90": _monthly(rows, "SURGICAL90"),
        }

        # A strategy can look fine in aggregate while damaging the few months that
        # actually made money. Require every originally-positive month to keep at
        # least 85% of its control profit as an additional anti-overfit guard.
        for key, pp in preservation.items():
            ratios = []
            for m in monthly[key]:
                c = int(m.get("control_KRW", 0) or 0)
                v = int(m.get("variant_KRW", 0) or 0)
                if c > 0:
                    ratios.append(100.0 * v / c)
            floor = round(min(ratios), 2) if ratios else 100.0
            pp["positive_month_profit_retention_min_pct"] = floor
            pp["passes_positive_month_85pct_floor"] = bool(floor >= 85.0)

        candidates95 = []
        candidates90 = []
        if parity_ok:
            for v in (s95, s90):
                pp = preservation[v["id"]]
                improving = v["total_KRW"] > control["total_KRW"]
                month_ok = pp["passes_positive_month_85pct_floor"]
                if improving and month_ok and pp["passes_95pct_profit_preservation"]:
                    candidates95.append(v)
                if improving and month_ok and pp["passes_90pct_profit_preservation"]:
                    candidates90.append(v)
        # Profit preservation wins the tie: if any 95%-tier variant exists, never
        # select a merely 90%-tier variant just for a slightly better total PnL.
        if candidates95:
            best = max(candidates95, key=lambda x: int(x["total_KRW"]))
            accepted_tier = "95pct"
        elif candidates90:
            best = max(candidates90, key=lambda x: int(x["total_KRW"]))
            accepted_tier = "90pct"
        else:
            best = None
            accepted_tier = None

        target_improvement = 1_000_000
        stretch_total = -500_000
        best_delta = int((best or {}).get("delta_vs_control_KRW", 0) or 0)

        diffs_95 = [day_delta(r, "SURGICAL95") for r in rows]
        diffs_90 = [day_delta(r, "SURGICAL90") for r in rows]

        payload = {
            "ok": True,
            "version": SURGICAL_VERSION,
            "mode": "PATH_CONSISTENT_SURGICAL_SHIELD_A_B_C",
            "read_only": True,
            "period": result.get("period", {}),
            "completed_at": datetime.now(KST).isoformat(timespec="seconds"),
            "days_expected": len(daily_base),
            "days_completed": len(rows),
            "errors": errors,
            "parity": {
                "required": True,
                "ok": parity_ok,
                "cached_D2_expected_total_KRW": expected_total,
                "control_total_KRW": int(control["total_KRW"]),
                "total_delta_KRW": int(control["total_KRW"] - expected_total),
                "mismatch_days": parity_mismatches,
                "rule": "CONTROL must match frozen D-v2 day-by-day before shield results are accepted.",
            },
            "variants": [control, s95, s90],
            "profit_preservation": preservation,
            "monthly": monthly,
            "best_if_parity_and_profit_preserved": best,
            "accepted_profit_preservation_tier": accepted_tier,
            "target": {
                "baseline_total_KRW": int(control["total_KRW"]),
                "first_goal_improvement_KRW": target_improvement,
                "first_goal_total_KRW": int(control["total_KRW"] + target_improvement),
                "stretch_goal_total_KRW": stretch_total,
                "best_accepted_improvement_KRW": best_delta,
                "remaining_to_first_goal_KRW": max(0, target_improvement - best_delta),
                "first_goal_reached": bool(best_delta >= target_improvement),
            },
            "acceptance": {
                "parity_exact_required": True,
                "profit_preservation_min_pct": 90.0,
                "preferred_profit_preservation_pct": 95.0,
                "positive_month_profit_retention_floor_pct": 85.0,
                "must_improve_total_pnl": True,
                "do_not_select_by_total_only": True,
            },
            "top_surgical95_improved_days": sorted(diffs_95, key=lambda x: x["delta_KRW"], reverse=True)[:10],
            "top_surgical95_worsened_days": sorted(diffs_95, key=lambda x: x["delta_KRW"])[:10],
            "top_surgical90_improved_days": sorted(diffs_90, key=lambda x: x["delta_KRW"], reverse=True)[:10],
            "top_surgical90_worsened_days": sorted(diffs_90, key=lambda x: x["delta_KRW"])[:10],
            "rules": {
                "profit_engine": "TAKE1 +3%, TAKE2 +5%, original PROFIT_GUARD unchanged",
                "COMMON": [
                    "default normal BUY1 remains 50%",
                    "no late-time size penalty, no PRE_GUARD, no post-stop risk state",
                    "no -1% partial trim; at most one deep structural-failure exit",
                    "peak >= +0.8% gets winner immunity and a deeper exit threshold",
                    "PG2 re-entry requires re-arm; TAKE2 re-entry unchanged",
                ],
                "SURGICAL95": [
                    "25% only for VWAP <= -0.5% with weak short momentum, or weak overheat >= +3.0%",
                    "normal BUY1 surgical exit: >=5m, pnl <= -2.4%, >=3 strict failures",
                    "winner-immunity exit: pnl <= -2.7% and all 4 failures",
                    "PG2 hard lock 5m, then full 50% only after TOP3+VWAP+3m+5m re-arm",
                    "OPEN_STRONG caution starts at VWAP >= +3.0%",
                ],
                "SURGICAL90": [
                    "25% only for VWAP below with weak short momentum, or weak overheat >= +2.8%",
                    "normal BUY1 surgical exit: >=4m, pnl <= -2.1%, >=3 strict failures",
                    "winner-immunity exit: pnl <= -2.5% and all 4 failures",
                    "PG2 hard lock 7m, then full 50% only after TOP3+VWAP+3m+5m re-arm",
                    "OPEN_STRONG caution starts at VWAP >= +2.5%",
                ],
            },
            "daily": rows,
            "important_limit": (
                "Historical candidate selection still uses the same fixed liquidity universe as the frozen D-v2 long replay; "
                "this validates path consistency inside that replay universe, not exact historical whole-market KIS TOP5. "
                "Fees/taxes are not included."
            ),
        }
        _atomic_write_json(result_file, payload)
        _state(
            status="completed",
            phase="DONE",
            completed_days=len(rows),
            error_days=len(errors),
            total_days=len(daily_base),
            progress_pct=100.0,
            result_ready=True,
            parity_ok=parity_ok,
            finished_at=datetime.now(KST).isoformat(timespec="seconds"),
            message="수술형 손실방어 전체엔진 검증 완료",
            last_error="",
        )
    except Exception as exc:
        _state(
            status="error",
            phase="ERROR",
            result_ready=result_file.exists(),
            last_error=f"{type(exc).__name__}: {exc}"[:1200],
            message="수술형 손실방어 전체엔진 검증 오류",
        )


def ensure_surgical_shield_started(result: dict) -> dict:
    """Called by the existing result route. Returns result or starts one background job."""
    global _THREAD
    base = _base()
    _, _, _, result_file = _paths()
    existing = _read_json(result_file, {}) or {}
    base_total = int(((result.get("overall") or {}).get("D2_total_KRW", 0)) or 0)

    if (
        existing.get("ok") is True
        and existing.get("version") == SURGICAL_VERSION
        and int(((existing.get("parity") or {}).get("cached_D2_expected_total_KRW", base_total)) or base_total)
        == base_total
    ):
        compact = dict(existing)
        compact.pop("daily", None)
        return compact

    # Avoid running two CPU-heavy full-engine jobs at the same time.
    runner_thread = getattr(base, "_RUNNER_FULL_THREAD", None)
    if runner_thread is not None and runner_thread.is_alive():
        state = _public_state()
        state.update({
            "status": "waiting_runner_full",
            "phase": "WAIT",
            "result_ready": False,
            "message": "기존 Runner 전체엔진 검증 종료 후 수술형 손실방어를 시작합니다.",
            "started": False,
        })
        return state

    with _LOCK:
        if _THREAD and _THREAD.is_alive():
            return _public_state()
        _THREAD = threading.Thread(
            target=_job,
            args=(dict(result),),
            daemon=True,
            name="kr-d2-surgical-shield-full-engine",
        )
        _THREAD.start()
        state = _public_state()
        state["started"] = True
        return state

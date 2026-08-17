from __future__ import annotations

"""D4 LOSS ROUTER + BAD DAY BRAKE full-engine replay.

Purpose
-------
Preserve the frozen D-v2 exit engine and opening behavior while targeting only
the loss shapes repeatedly found in the 147-day replay:

1) deep-below-VWAP rebound traps,
2) overheated chase entries,
3) same-day re-entry after STOP_LOSS,
4) PROFIT_GUARD2 re-entry without a fresh setup,
5) weak-market (RED) non-A+ entries,
6) continued new buying after the system has clearly lost control of the day.

No real orders are sent. The job reads only the cached KIS 1-minute bars already
stored by the long backtest. All decisions use bars available at that timestamp;
future bars are never used.

Modes
-----
CONTROL
    Exact frozen D-v2 path for parity.

LOSS_TRAPS
    Ordinary BUY1 only:
    - same-day STOP_LOSS re-entry is blocked,
    - if first seen <= -1.50% below VWAP, wait for real recovery + breakout,
    - if first seen >= +3.50% above VWAP, require pullback then re-breakout.

PG2_REARM
    LOSS_TRAPS plus PROFIT_GUARD2 re-entry requires a 10-minute re-arm and a
    fresh persistent-leader breakout.

LOSS_ROUTER
    PG2_REARM plus market regime as a *selective* risk weight:
    - RED market does NOT automatically block an A+ leader,
    - other ordinary BUY1s in RED must earn a fresh breakout.

ROUTER_BRAKE
    LOSS_ROUTER plus an intraday circuit breaker:
    - after 2 STOP_LOSS events, OR
    - after realized P&L <= -100,000 KRW,
    stop all new buying for that day.
    Existing positions keep the original D-v2 STOP/TAKE/PROFIT_GUARD exits.

The opening 09:09~09:20 D-v2 entry rules are unchanged until/if the daily brake
actually activates. The exit engine is unchanged in every mode.
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
import replay_kr_d3_edge_entry_full as d3

KST = d2.KST
D4_VERSION = "kr-d4-loss-router-brake-full-engine-v1"

_THREAD: threading.Thread | None = None
_LOCK = threading.RLock()


def _base():
    import replay_kr_long_backtest as base
    return base


def _paths():
    base = _base()
    root = base.ROOT / "d4_loss_router_full_engine"
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
        cur.setdefault("version", D4_VERSION)
        cur["updated_at"] = datetime.now(KST).isoformat(timespec="seconds")
        _atomic_write_json(state_file, cur)
        return cur


def _public_state() -> dict:
    _, _, state_file, _ = _paths()
    state = _read_json(state_file, {}) or {}
    if not state:
        state = {
            "ok": True,
            "version": D4_VERSION,
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


def _is_a_plus(state: dict, vwap_gap: float) -> bool:
    """High-quality leader allowed directly even in RED regime.

    This deliberately requires stronger evidence than the frozen D-v2 minimum:
    TOP1~2, score >= 75, positive short momentum, meaningful volume, and a
    non-overheated location above VWAP.
    """
    return bool(
        state["signal"]
        and not state["weak"]
        and state["rank"] <= 2
        and state["score"] >= 75.0
        and 0.0 <= float(vwap_gap) <= 2.80
        and state["ret3"] >= 0.50
        and state["ret5"] >= 0.80
        and state["volume"] >= 1.20
    )


def _pack(x: dict) -> dict:
    sm = x.get("summary", {}) or {}
    events = [e for e in list(x.get("events", []) or []) if isinstance(e, dict)]
    realized = [
        int(e.get("실현손익KRW", 0) or 0)
        for e in events
        if str(e.get("구분", "")).upper() == "SELL"
    ]
    actions: dict[str, int] = {}
    for e in events:
        a = str(e.get("액션", "") or "")
        if a:
            actions[a] = actions.get(a, 0) + 1
    wins = [v for v in realized if v > 0]
    losses = [v for v in realized if v < 0]
    return {
        "pnl_KRW": int(sm.get("실현손익KRW", 0) or 0),
        "buy_orders": int(sm.get("매수주문횟수", 0) or 0),
        "sell_orders": int(sm.get("매도주문횟수", 0) or 0),
        "total_orders": int(sm.get("총주문횟수", 0) or 0),
        "traded_symbols": int(sm.get("거래종목수", 0) or 0),
        "gross_profit_KRW": int(sum(wins)),
        "gross_loss_KRW": int(sum(losses)),
        "positive_sell_events": len(wins),
        "negative_sell_events": len(losses),
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
    gp = int(sum(int((r.get(key) or {}).get("gross_profit_KRW", 0) or 0) for r in rows))
    gl = int(sum(int((r.get(key) or {}).get("gross_loss_KRW", 0) or 0) for r in rows))
    win_events = int(sum(int((r.get(key) or {}).get("positive_sell_events", 0) or 0) for r in rows))
    loss_events = int(sum(int((r.get(key) or {}).get("negative_sell_events", 0) or 0) for r in rows))

    diag_keys = [
        "deep_vwap_first_blocks",
        "deep_vwap_recovery_waits",
        "deep_vwap_recovery_entries",
        "overheat_first_blocks",
        "overheat_pullback_waits",
        "overheat_rebreak_waits",
        "overheat_rebreak_entries",
        "stop_reentry_blocks",
        "pg2_lock_blocks",
        "pg2_rearm_waits",
        "pg2_rearm_entries",
        "red_non_a_plus_waits",
        "red_a_plus_direct_entries",
        "red_breakout_entries",
        "brake_activations",
        "brake_stop2_activations",
        "brake_loss_activations",
        "brake_blocked_entry_ticks",
        "brake_blocked_open_confirms",
        "normal_stop_losses",
    ]
    diag = {
        k: int(
            sum(
                int(((r.get(key) or {}).get("diagnostic") or {}).get(k, 0) or 0)
                for r in rows
            )
        )
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
        "gross_profit_KRW": gp,
        "gross_loss_KRW": gl,
        "profit_factor": round(gp / abs(gl), 4) if gl < 0 else None,
        "positive_sell_events": win_events,
        "negative_sell_events": loss_events,
        "sell_event_win_rate_pct": (
            round(100.0 * win_events / (win_events + loss_events), 2)
            if (win_events + loss_events)
            else 0.0
        ),
        "average_positive_sell_KRW": round(gp / win_events, 1) if win_events else 0.0,
        "average_negative_sell_KRW": round(gl / loss_events, 1) if loss_events else 0.0,
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
    control_gp = int(
        sum(int((r.get("CONTROL_D2") or {}).get("gross_profit_KRW", 0) or 0) for r in rows)
    )
    variant_gp = int(
        sum(int((r.get(key) or {}).get("gross_profit_KRW", 0) or 0) for r in rows)
    )
    gross_pct = round(100.0 * variant_gp / control_gp, 2) if control_gp > 0 else 100.0

    return {
        "control_positive_day_profit_KRW": int(control_profit),
        "profit_sacrificed_on_control_positive_days_KRW": int(sacrificed),
        "extra_profit_on_control_positive_days_KRW": int(gained_on_positive),
        "profit_preservation_pct": pct,
        "control_gross_profit_KRW": control_gp,
        "variant_gross_profit_KRW": variant_gp,
        "gross_profit_preservation_pct": gross_pct,
        "control_positive_days_hurt": int(positive_days_hurt),
        "loss_reduction_on_control_negative_days_KRW": int(loss_reduction),
        "loss_worsening_on_control_negative_days_KRW": int(loss_worsening),
        "control_negative_days_improved": int(negative_days_improved),
        "passes_positive_day_90pct": bool(pct >= 90.0),
        "passes_gross_profit_95pct": bool(gross_pct >= 95.0),
    }


def _monthly(rows: list[dict], key: str) -> list[dict]:
    out: dict[str, dict] = {}
    for r in rows:
        month = str(r.get("date", ""))[:7]
        if not month:
            continue
        c = int((r.get("CONTROL_D2") or {}).get("pnl_KRW", 0) or 0)
        v = int((r.get(key) or {}).get("pnl_KRW", 0) or 0)
        m = out.setdefault(
            month,
            {"month": month, "days": 0, "control_KRW": 0, "variant_KRW": 0, "delta_KRW": 0},
        )
        m["days"] += 1
        m["control_KRW"] += c
        m["variant_KRW"] += v
        m["delta_KRW"] += v - c
    return [out[k] for k in sorted(out)]


def _profitable_month_preservation(monthly_rows: list[dict]) -> dict:
    vals = []
    for m in monthly_rows:
        c = int(m.get("control_KRW", 0) or 0)
        v = int(m.get("variant_KRW", 0) or 0)
        if c > 0:
            pct = max(0.0, 100.0 * v / c)
            vals.append(
                {
                    "month": m.get("month"),
                    "control_KRW": c,
                    "variant_KRW": v,
                    "preservation_pct": round(pct, 2),
                }
            )
    return {
        "months": vals,
        "min_preservation_pct": min((x["preservation_pct"] for x in vals), default=100.0),
        "passes_85pct_each_profitable_month": all(x["preservation_pct"] >= 85.0 for x in vals),
    }


def run_kr_d4_loss_router_replay(
    date_text: str,
    codes: Iterable[str] | None = None,
    config: d2.OpenDefenseConfig | None = None,
    mode: str = "CONTROL",
) -> dict:
    cfg = config or d2.OpenDefenseConfig()
    mode = str(mode or "CONTROL").upper().strip()
    valid_modes = {"CONTROL", "LOSS_TRAPS", "PG2_REARM", "LOSS_ROUTER", "ROUTER_BRAKE"}
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {sorted(valid_modes)}")

    use_loss_traps = mode != "CONTROL"
    use_pg2 = mode in {"PG2_REARM", "LOSS_ROUTER", "ROUTER_BRAKE"}
    use_market = mode in {"LOSS_ROUTER", "ROUTER_BRAKE"}
    use_brake = mode == "ROUTER_BRAKE"

    c_cfg = d3._make_c_cfg(cfg)

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

    normal_amount = d2._normal_entry_amount(cfg)
    defense_amount = d2._defense_entry_amount(cfg)
    confirm_amount = max(0, normal_amount - defense_amount)

    positions: dict[str, dict] = {}
    events: list[dict] = []
    latest_top5 = pd.DataFrame()
    last_scan = None
    daily_buy_amount = 0.0
    daily_orders = 0
    leader_streaks: dict[str, int] = {}
    last_market_snapshot: dict = {}

    stopped_symbols: set[str] = set()
    pg2_disarmed_at: dict[str, pd.Timestamp] = {}
    deep_pending: set[str] = set()
    overheat_state: dict[str, dict] = {}

    realized_today = 0.0
    stop_loss_count = 0
    brake_active = False
    brake_reason = ""

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
        "normal_stop_losses": 0,
        "deep_vwap_first_blocks": 0,
        "deep_vwap_recovery_waits": 0,
        "deep_vwap_recovery_entries": 0,
        "overheat_first_blocks": 0,
        "overheat_pullback_waits": 0,
        "overheat_rebreak_waits": 0,
        "overheat_rebreak_entries": 0,
        "stop_reentry_blocks": 0,
        "pg2_lock_blocks": 0,
        "pg2_rearm_waits": 0,
        "pg2_rearm_entries": 0,
        "red_non_a_plus_waits": 0,
        "red_a_plus_direct_entries": 0,
        "red_breakout_entries": 0,
        "brake_activations": 0,
        "brake_stop2_activations": 0,
        "brake_loss_activations": 0,
        "brake_blocked_entry_ticks": 0,
        "brake_blocked_open_confirms": 0,
    }
    chase_wait_seen: set[tuple[str, str]] = set()
    one_shot_seen: set[tuple[str, str]] = set()

    def once(key: str, symbol: str) -> bool:
        marker = (key, symbol + "|" + now.isoformat())
        if marker in one_shot_seen:
            return False
        one_shot_seen.add(marker)
        return True

    def maybe_activate_brake() -> None:
        nonlocal brake_active, brake_reason
        if not use_brake or brake_active:
            return
        if stop_loss_count >= 2:
            brake_active = True
            brake_reason = "STOP_LOSS_2"
            diagnostic["brake_activations"] += 1
            diagnostic["brake_stop2_activations"] += 1
        elif realized_today <= -100000.0:
            brake_active = True
            brake_reason = "REALIZED_PNL_-100K"
            diagnostic["brake_activations"] += 1
            diagnostic["brake_loss_activations"] += 1

    def add_event(
        ts,
        symbol,
        action,
        side,
        qty,
        ref_price,
        fill_price,
        reason,
        pnl="",
        realized=0.0,
        score="",
        rank="",
        vwap_gap="",
    ):
        nonlocal daily_orders, realized_today, stop_loss_count
        events.append(
            {
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
            }
        )
        daily_orders += 1

        if str(side).upper() == "SELL":
            realized_today += float(realized or 0.0)
            if action == "STOP_LOSS":
                stop_loss_count += 1
                stopped_symbols.add(symbol)
            maybe_activate_brake()

    now = start
    while now <= end:
        if last_scan is None or (now - last_scan).total_seconds() >= int(cfg.scan_seconds):
            latest_top5 = d3._top5_cached(
                target_frames, meta, date_text, now, cfg.scan_count
            )
            last_scan = now

            qualifying: set[str] = set()
            if latest_top5 is not None and not latest_top5.empty:
                for _, rr in latest_top5.iterrows():
                    ss = str(rr.get("종목코드", "")).zfill(6)
                    st = d3._row_state(rr)
                    if st["rank"] <= 3 and st["signal"] and not st["weak"]:
                        qualifying.add(ss)
            for ss in list(leader_streaks):
                if ss not in qualifying:
                    leader_streaks[ss] = 0
            for ss in qualifying:
                leader_streaks[ss] = int(leader_streaks.get(ss, 0)) + 1

            if use_market:
                last_market_snapshot = d3._market_snapshot_cached(
                    target_frames, date_text, now, latest_top5
                )

        top5_map: dict[str, Any] = {}
        if latest_top5 is not None and not latest_top5.empty:
            for _, r in latest_top5.iterrows():
                top5_map[str(r.get("종목코드", "")).zfill(6)] = r

        # ------------------------------------------------------------------
        # 1) Existing-position management: frozen D-v2 exits
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
            state = d3._row_state(row)

            if replay_kr._seconds_of_day(now) >= force_exit_sec:
                fill = replay_kr._fill_price(c_cfg, "SELL", ref_price)
                realized = (fill - avg) * qty
                add_event(
                    now,
                    symbol,
                    "FORCE_SELL",
                    "SELL",
                    qty,
                    ref_price,
                    fill,
                    f"당일 강제청산 {cfg.force_exit_time} KST",
                    pnl,
                    realized,
                    vwap_gap=vg,
                )
                positions.pop(symbol, None)
                continue

            if bool(pos.get("defense_position", False)):
                hold_min = d3._hold_minutes(pos, now)
                if hold_min <= float(cfg.defense_fail_window_minutes):
                    if pnl <= float(cfg.defense_emergency_fail_pct):
                        fill = replay_kr._fill_price(c_cfg, "SELL", ref_price)
                        realized = (fill - avg) * qty
                        add_event(
                            now,
                            symbol,
                            "OPEN_EMERGENCY_EXIT",
                            "SELL",
                            qty,
                            ref_price,
                            fill,
                            f"장초반 비상청산 · pnl {pnl:.2f}% <= {cfg.defense_emergency_fail_pct:.2f}%",
                            pnl,
                            realized,
                            vwap_gap=vg,
                        )
                        diagnostic["open_emergency_exits"] += 1
                        positions.pop(symbol, None)
                        continue

                    if (
                        hold_min >= float(cfg.soft_fail_min_hold_minutes)
                        and pnl <= float(cfg.defense_soft_fail_pct)
                    ):
                        fail = d3._fail_signals(cfg, state, vg)
                        if len(fail) >= int(cfg.soft_fail_min_signals):
                            fill = replay_kr._fill_price(c_cfg, "SELL", ref_price)
                            realized = (fill - avg) * qty
                            add_event(
                                now,
                                symbol,
                                "OPEN_SOFT_FAIL_EXIT",
                                "SELL",
                                qty,
                                ref_price,
                                fill,
                                f"복합 돌파실패 · pnl {pnl:.2f}% · " + ",".join(fail),
                                pnl,
                                realized,
                                vwap_gap=vg,
                            )
                            diagnostic["open_soft_fail_exits"] += 1
                            positions.pop(symbol, None)
                            continue

            is_normal_buy1 = str(pos.get("entry_action", "")) == "BUY1"

            if pnl <= -abs(cfg.stop_loss_pct):
                fill = replay_kr._fill_price(c_cfg, "SELL", ref_price)
                realized = (fill - avg) * qty
                add_event(
                    now,
                    symbol,
                    "STOP_LOSS",
                    "SELL",
                    qty,
                    ref_price,
                    fill,
                    f"손절 {pnl:.2f}%",
                    pnl,
                    realized,
                    vwap_gap=vg,
                )
                if is_normal_buy1:
                    diagnostic["normal_stop_losses"] += 1
                positions.pop(symbol, None)
                continue

            # OPEN_CONFIRM stays frozen unless the daily brake has actually fired.
            if (
                bool(pos.get("defense_position", False))
                and not bool(pos.get("open_confirmed", False))
                and replay_kr._seconds_of_day(now) < defense_end_sec
                and daily_orders < cfg.max_daily_orders
            ):
                if use_brake and brake_active:
                    diagnostic["brake_blocked_open_confirms"] += 1
                else:
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
                                now,
                                symbol,
                                "OPEN_CONFIRM",
                                "BUY",
                                qty2,
                                ref_price,
                                fill,
                                why,
                                pnl,
                                0.0,
                                score,
                                rank,
                                vg,
                            )
                            pos["qty"] = new_qty
                            pos["avg_price"] = new_avg
                            pos["open_confirmed"] = True
                            daily_buy_amount += cost
                            diagnostic["open_confirms"] += 1
                            continue

            if pnl >= cfg.take1_pct and not bool(pos.get("take1_sent")):
                sell_qty = max(1, qty // 2)
                fill = replay_kr._fill_price(c_cfg, "SELL", ref_price)
                realized = (fill - avg) * sell_qty
                add_event(
                    now,
                    symbol,
                    "TAKE1",
                    "SELL",
                    sell_qty,
                    ref_price,
                    fill,
                    f"1차 익절 {pnl:.2f}% · 약 50%",
                    pnl,
                    realized,
                    vwap_gap=vg,
                )
                pos["qty"] = qty - sell_qty
                pos["take1_sent"] = True
                pg2_disarmed_at.pop(symbol, None)
                if pos["qty"] <= 0:
                    positions.pop(symbol, None)
                continue

            if pnl >= cfg.take2_pct and bool(pos.get("take1_sent")):
                fill = replay_kr._fill_price(c_cfg, "SELL", ref_price)
                realized = (fill - avg) * qty
                add_event(
                    now,
                    symbol,
                    "TAKE2",
                    "SELL",
                    qty,
                    ref_price,
                    fill,
                    f"2차 익절 {pnl:.2f}% · 전량",
                    pnl,
                    realized,
                    vwap_gap=vg,
                )
                pg2_disarmed_at.pop(symbol, None)
                positions.pop(symbol, None)
                continue

            suppress_profit_guard = False
            if bool(pos.get("defense_position", False)) and not bool(pos.get("open_confirmed", False)):
                suppress_profit_guard = (
                    d3._hold_minutes(pos, now)
                    < float(cfg.profit_guard_confirm_grace_minutes)
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
                        now,
                        symbol,
                        "PROFIT_GUARD1",
                        "SELL",
                        sell_qty,
                        ref_price,
                        fill,
                        f"수익보호 1차 · 최고 +{peak:.2f}% → 현재 {pnl:.2f}% ({dd:.2f}%p 되밀림)",
                        pnl,
                        realized,
                        vwap_gap=vg,
                    )
                    pos["qty"] = qty - sell_qty
                    pos["take1_sent"] = True
                    if pos["qty"] <= 0:
                        positions.pop(symbol, None)
                    continue
                else:
                    fill = replay_kr._fill_price(c_cfg, "SELL", ref_price)
                    realized = (fill - avg) * qty
                    add_event(
                        now,
                        symbol,
                        "PROFIT_GUARD2",
                        "SELL",
                        qty,
                        ref_price,
                        fill,
                        f"수익보호 2차 · 최고 +{peak:.2f}% → 현재 {pnl:.2f}% ({dd:.2f}%p 되밀림)",
                        pnl,
                        realized,
                        vwap_gap=vg,
                    )
                    if use_pg2:
                        pg2_disarmed_at[symbol] = now
                    positions.pop(symbol, None)
                    continue

        # ------------------------------------------------------------------
        # 2) New entries
        # ------------------------------------------------------------------
        can_scan_entries = (
            replay_kr._seconds_of_day(now) < last_entry_sec
            and len(positions) < cfg.max_positions
            and daily_orders < cfg.max_daily_orders
            and latest_top5 is not None
            and not latest_top5.empty
        )

        if can_scan_entries:
            if use_brake and brake_active:
                diagnostic["brake_blocked_entry_ticks"] += 1
            else:
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
                    state = d3._row_state(row)

                    gate_notes: list[str] = []

                    # Opening window is frozen D-v2.
                    if defense_now:
                        strong_open, _ = d2._is_strong_open(cfg, row, vg)
                        if strong_open:
                            target_amount = normal_amount
                            action = "OPEN_STRONG_BUY"
                            defense_position = False
                        else:
                            defense_count = sum(
                                1
                                for p in positions.values()
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
                        target_amount = normal_amount
                        action = "BUY1"
                        defense_position = False

                        # --------------------------------------------------
                        # D4 LOSS ROUTER: ordinary BUY1 permission only
                        # --------------------------------------------------
                        if use_loss_traps and symbol in stopped_symbols:
                            if once("STOP_REENTRY", symbol):
                                diagnostic["stop_reentry_blocks"] += 1
                            continue

                        # PROFIT_GUARD2 re-arm. TAKE2 re-entry is untouched.
                        if use_pg2 and symbol in pg2_disarmed_at:
                            elapsed = max(
                                0.0,
                                (now - pg2_disarmed_at[symbol]).total_seconds() / 60.0,
                            )
                            if elapsed < 10.0:
                                if once("PG2_LOCK", symbol):
                                    diagnostic["pg2_lock_blocks"] += 1
                                continue

                            streak = int(leader_streaks.get(symbol, 0))
                            ok_rearm, _ = d3._breakout_allowed(
                                frame, date_text, now, ref_price, state, streak, vg
                            )
                            rearm_quality = bool(
                                state["rank"] <= 3
                                and state["score"] >= 65.0
                                and state["volume"] >= 1.05
                                and vg >= 0.0
                                and state["ret3"] > 0.0
                                and state["ret5"] > 0.0
                                and not state["weak"]
                                and state["signal"]
                            )
                            if not (ok_rearm and rearm_quality):
                                if once("PG2_REARM_WAIT", symbol):
                                    diagnostic["pg2_rearm_waits"] += 1
                                continue
                            pg2_disarmed_at.pop(symbol, None)
                            diagnostic["pg2_rearm_entries"] += 1
                            gate_notes.append("PG2_REARM_OK")

                        # Deep-below-VWAP rebound trap.
                        if use_loss_traps and vg <= -1.50 and symbol not in deep_pending:
                            deep_pending.add(symbol)
                            diagnostic["deep_vwap_first_blocks"] += 1
                            continue

                        if use_loss_traps and symbol in deep_pending:
                            streak = int(leader_streaks.get(symbol, 0))
                            ok_recover, _ = d3._breakout_allowed(
                                frame, date_text, now, ref_price, state, streak, vg
                            )
                            if not ok_recover:
                                if once("DEEP_RECOVERY_WAIT", symbol):
                                    diagnostic["deep_vwap_recovery_waits"] += 1
                                continue
                            deep_pending.discard(symbol)
                            diagnostic["deep_vwap_recovery_entries"] += 1
                            gate_notes.append("DEEP_RECOVERY_BREAKOUT")

                        # Overheated chase: first observe -> require pullback -> re-breakout.
                        if use_loss_traps:
                            o = overheat_state.get(symbol)
                            if o is None and vg >= 3.50:
                                overheat_state[symbol] = {
                                    "first_at": now,
                                    "peak_vg": float(vg),
                                    "pullback_seen": False,
                                }
                                diagnostic["overheat_first_blocks"] += 1
                                continue

                            o = overheat_state.get(symbol)
                            if o is not None:
                                o["peak_vg"] = max(float(o.get("peak_vg", vg)), float(vg))
                                if (
                                    not bool(o.get("pullback_seen", False))
                                    and float(vg) <= float(o["peak_vg"]) - 0.50
                                ):
                                    o["pullback_seen"] = True

                                if not bool(o.get("pullback_seen", False)):
                                    if once("OVERHEAT_PULLBACK_WAIT", symbol):
                                        diagnostic["overheat_pullback_waits"] += 1
                                    continue

                                streak = int(leader_streaks.get(symbol, 0))
                                ok_rebreak, _ = d3._breakout_allowed(
                                    frame, date_text, now, ref_price, state, streak, vg
                                )
                                if not ok_rebreak:
                                    if once("OVERHEAT_REBREAK_WAIT", symbol):
                                        diagnostic["overheat_rebreak_waits"] += 1
                                    continue
                                overheat_state.pop(symbol, None)
                                diagnostic["overheat_rebreak_entries"] += 1
                                gate_notes.append("OVERHEAT_PULLBACK_REBREAK")

                        # RED market becomes a selective gate, never a blanket ban.
                        if use_market:
                            snap = last_market_snapshot or d3._market_snapshot_cached(
                                target_frames, date_text, now, latest_top5
                            )
                            if snap.get("regime") == "RED":
                                if _is_a_plus(state, vg):
                                    diagnostic["red_a_plus_direct_entries"] += 1
                                    gate_notes.append("RED_A_PLUS_DIRECT")
                                else:
                                    streak = int(leader_streaks.get(symbol, 0))
                                    ok_red, _ = d3._breakout_allowed(
                                        frame, date_text, now, ref_price, state, streak, vg
                                    )
                                    if not ok_red:
                                        if once("RED_WAIT", symbol):
                                            diagnostic["red_non_a_plus_waits"] += 1
                                        continue
                                    diagnostic["red_breakout_entries"] += 1
                                    gate_notes.append("RED_FRESH_BREAKOUT")

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
                    elif action == "OPEN_DEFENSE_BUY":
                        prefix = "OPEN DEFENSE 25%"
                    elif mode == "CONTROL":
                        prefix = "C기준 50%"
                    else:
                        prefix = f"D4 {mode} 50%"

                    if gate_notes:
                        prefix += " · " + ",".join(gate_notes)

                    add_event(
                        now,
                        symbol,
                        action,
                        "BUY",
                        qty1,
                        ref_price,
                        fill,
                        f"{prefix} · 점수 {score:.1f} · 3분 {r3:+.2f}% · 5분 {r5:+.2f}% · "
                        f"거래량 {vr:.2f}배 · VWAP {vg:+.2f}%",
                        "",
                        0.0,
                        score,
                        rank,
                        vg,
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
                    }
                    daily_buy_amount += cost

                    if action == "OPEN_STRONG_BUY":
                        diagnostic["strong_open_entries"] += 1
                    elif action == "OPEN_DEFENSE_BUY":
                        diagnostic["defense_entries"] += 1
                    else:
                        diagnostic["normal_entries"] += 1

        now += pd.Timedelta(seconds=int(cfg.manage_seconds))

    if positions:
        for symbol, pos in list(positions.items()):
            frame = target_frames.get(symbol)
            ref_price = (
                replay_kr._price_at(frame, date_text, end)
                if frame is not None
                else 0.0
            )
            if ref_price <= 0:
                continue
            qty = int(pos.get("qty", 0))
            avg = d2._safe_float(pos.get("avg_price", 0))
            fill = replay_kr._fill_price(c_cfg, "SELL", ref_price)
            pnl = (ref_price / avg - 1.0) * 100.0 if avg > 0 else 0.0
            realized = (fill - avg) * qty
            vg = d2._vwap_gap_pct(frame, date_text, end, ref_price)
            add_event(
                end,
                symbol,
                "FORCE_SELL_END",
                "SELL",
                qty,
                ref_price,
                fill,
                "리플레이 종료 안전청산",
                pnl,
                realized,
                vwap_gap=vg,
            )
            positions.pop(symbol, None)

    events_df = pd.DataFrame(events)
    if events_df.empty:
        buy_amount = sell_amount = realized = 0.0
    else:
        buy_amount = float(
            events_df.loc[events_df["구분"] == "BUY", "주문금액KRW"].sum()
        )
        sell_amount = float(
            events_df.loc[events_df["구분"] == "SELL", "주문금액KRW"].sum()
        )
        realized = float(events_df["실현손익KRW"].sum())

    summary = {
        "총주문횟수": int(len(events_df)),
        "매수주문횟수": int((events_df["구분"] == "BUY").sum()) if not events_df.empty else 0,
        "매도주문횟수": int((events_df["구분"] == "SELL").sum()) if not events_df.empty else 0,
        "거래종목수": int(events_df["종목코드"].nunique()) if not events_df.empty else 0,
        "누적매수금액KRW": int(round(buy_amount)),
        "누적매도금액KRW": int(round(sell_amount)),
        "실현손익KRW": int(round(realized)),
        "누적매수금액대비수익률": (
            round(realized / buy_amount * 100.0, 3) if buy_amount > 0 else 0.0
        ),
        "일일예산1000만원대비수익률": (
            round(realized / cfg.daily_budget_krw * 100.0, 3)
            if cfg.daily_budget_krw > 0
            else 0.0
        ),
    }

    return {
        "ok": True,
        "version": D4_VERSION,
        "date": date_text,
        "strategy": f"D4_LOSS_ROUTER_{mode}",
        "mode": mode,
        "diagnostic": diagnostic,
        "summary": summary,
        "events": events,
        "config": asdict(cfg),
        "day_risk_state": {
            "realized_pnl_KRW": int(round(realized_today)),
            "stop_loss_count": int(stop_loss_count),
            "brake_active": bool(brake_active),
            "brake_reason": brake_reason,
        },
        "rules": {
            "opening": "frozen D-v2 09:09~09:20; only an activated daily brake can stop additional buying",
            "exits": "frozen D-v2 STOP -3%, TAKE1 +3%, TAKE2 +5%, PROFIT_GUARD unchanged",
            "loss_traps": {
                "deep_below_vwap_first_block_pct": -1.50,
                "deep_recovery": "persistent TOP3 + VWAP>=0 + ret3/ret5>0 + prior five-bar high breakout",
                "overheat_first_block_pct": 3.50,
                "overheat_pullback_required_pct_points": 0.50,
                "overheat_reentry": "pullback must occur, then persistent TOP3 five-bar re-breakout",
                "stop_loss_same_day_reentry": "blocked",
            },
            "pg2_rearm": {
                "minimum_minutes": 10.0,
                "fresh_setup_required": True,
                "take2_reentry_untouched": True,
            },
            "market_router": {
                "red_blanket_ban": False,
                "a_plus_direct": "TOP<=2, score>=75, VWAP 0~2.8, ret3>=0.5, ret5>=0.8, volume>=1.2",
                "red_non_a_plus": "fresh persistent-leader five-bar breakout required",
            },
            "daily_brake": {
                "stop_loss_trigger_count": 2,
                "realized_pnl_trigger_KRW": -100000,
                "effect": "stop all new BUY orders; manage/sell existing positions normally",
            },
        },
        "assumptions": {
            "real_orders": False,
            "future_data_visible": False,
            "fees_taxes": "별도 미포함",
            "purpose": "D4 loss-shape routing + intraday bad-day brake validation",
        },
    }


def _job(
    result: dict,
    provider,
    codes,
    frozen_config: d2.OpenDefenseConfig,
    protected_window_fn,
) -> None:
    base = _base()
    _, _, _, result_file = _paths()
    try:
        daily_base = [
            r
            for r in list(result.get("daily", []) or [])
            if isinstance(r, dict) and r.get("date")
        ]
        daily_base.sort(key=lambda r: str(r.get("date")))
        if not daily_base:
            raise RuntimeError("기존 147일 daily 결과가 없습니다.")

        replay_kr._download_intraday = provider
        d2._download_intraday = provider

        modes = ["CONTROL", "LOSS_TRAPS", "PG2_REARM", "LOSS_ROUTER", "ROUTER_BRAKE"]

        _state(
            status="running",
            phase="D4_LOSS_ROUTER_FULL_ENGINE",
            started_at=datetime.now(KST).isoformat(timespec="seconds"),
            total_days=len(daily_base),
            completed_days=0,
            progress_pct=0.0,
            result_ready=False,
            error_days=0,
            message="D4 CONTROL / LOSS_TRAPS / PG2_REARM / LOSS_ROUTER / ROUTER_BRAKE 비교 시작",
            last_error="",
        )

        rows: list[dict] = []
        errors: list[dict] = []
        parity_mismatches: list[dict] = []

        for idx, base_row in enumerate(daily_base, start=1):
            date_text = str(base_row.get("date"))

            while True:
                live, label = protected_window_fn()
                if not live:
                    break
                _state(
                    status="paused_live_window",
                    phase="PAUSED",
                    pause_reason=label,
                    current_date=date_text,
                    completed_days=len(rows),
                    total_days=len(daily_base),
                    message="실시간 자동매매 보호를 위해 D4 검증 일시정지",
                )
                time.sleep(30.0)

            _state(
                status="running",
                phase="D4_LOSS_ROUTER_FULL_ENGINE",
                current_date=date_text,
                completed_days=len(rows),
                error_days=len(errors),
                total_days=len(daily_base),
                progress_pct=round(100.0 * (idx - 1) / len(daily_base), 1),
                message=f"D4 LOSS ROUTER {idx}/{len(daily_base)} · {date_text}",
            )

            day_path = _day_path(date_text)
            cached = base._load_gzip_json(day_path, None)
            if isinstance(cached, dict) and cached.get("version") == D4_VERSION:
                row = cached
            else:
                try:
                    outputs = {
                        mode: run_kr_d4_loss_router_replay(
                            date_text=date_text,
                            codes=codes,
                            config=frozen_config,
                            mode=mode,
                        )
                        for mode in modes
                    }
                    row = {
                        "version": D4_VERSION,
                        "date": date_text,
                        "cached_D2_KRW": int(base_row.get("D2_KRW", 0) or 0),
                        **{
                            ("CONTROL_D2" if mode == "CONTROL" else mode): _pack(outputs[mode])
                            for mode in modes
                        },
                    }
                    row["parity_delta_KRW"] = int(
                        row["CONTROL_D2"]["pnl_KRW"] - row["cached_D2_KRW"]
                    )
                    for key in modes[1:]:
                        row[key]["delta_vs_control_KRW"] = int(
                            row[key]["pnl_KRW"] - row["CONTROL_D2"]["pnl_KRW"]
                        )
                    base._save_gzip_json(day_path, row)
                except Exception as exc:
                    errors.append(
                        {"date": date_text, "error": f"{type(exc).__name__}: {exc}"}
                    )
                    _state(last_error=errors[-1]["error"][:1000])
                    continue

            rows.append(row)
            delta = int(row.get("parity_delta_KRW", 0) or 0)
            if delta != 0:
                parity_mismatches.append(
                    {
                        "date": date_text,
                        "cached_D2_KRW": int(row.get("cached_D2_KRW", 0) or 0),
                        "control_D2_KRW": int(
                            (row.get("CONTROL_D2") or {}).get("pnl_KRW", 0) or 0
                        ),
                        "delta_KRW": delta,
                    }
                )

            _state(
                status="running",
                phase="D4_LOSS_ROUTER_FULL_ENGINE",
                current_date=date_text,
                completed_days=len(rows),
                error_days=len(errors),
                total_days=len(daily_base),
                progress_pct=round(100.0 * idx / len(daily_base), 1),
                message=f"D4 LOSS ROUTER {idx}/{len(daily_base)} 완료",
            )

        control = _aggregate("D-v2 원본 전체엔진", "CONTROL_D2", rows)
        loss_traps = _aggregate(
            "LOSS_TRAPS · 깊은하락/과열/STOP재진입 선택 차단",
            "LOSS_TRAPS",
            rows,
        )
        pg2 = _aggregate(
            "PG2_REARM · LOSS_TRAPS + PG2 새 셋업 확인",
            "PG2_REARM",
            rows,
        )
        router = _aggregate(
            "LOSS_ROUTER · PG2_REARM + RED장 선택형 라우팅",
            "LOSS_ROUTER",
            rows,
        )
        brake = _aggregate(
            "ROUTER_BRAKE · LOSS_ROUTER + STOP2/-10만원 신규매수 중단",
            "ROUTER_BRAKE",
            rows,
        )
        variants = [control, loss_traps, pg2, router, brake]

        for v in variants[1:]:
            v["delta_vs_control_KRW"] = int(v["total_KRW"] - control["total_KRW"])

        parity_ok = (
            len(rows) == len(daily_base)
            and not errors
            and not parity_mismatches
        )

        keys = ["LOSS_TRAPS", "PG2_REARM", "LOSS_ROUTER", "ROUTER_BRAKE"]
        preservation = {k: _profit_preservation(rows, k) for k in keys}
        monthly = {"CONTROL_D2": _monthly(rows, "CONTROL_D2")}
        month_pres = {}
        for k in keys:
            monthly[k] = _monthly(rows, k)
            month_pres[k] = _profitable_month_preservation(monthly[k])

        guardrail_passed = []
        production_passed = []
        for v in variants[1:]:
            key = v["id"]
            guardrails = (
                parity_ok
                and int(v["delta_vs_control_KRW"]) > 0
                and int(v["max_cumulative_drawdown_KRW"])
                < int(control["max_cumulative_drawdown_KRW"])
                and preservation[key]["passes_positive_day_90pct"]
                and preservation[key]["passes_gross_profit_95pct"]
                and month_pres[key]["passes_85pct_each_profitable_month"]
            )
            v["guardrails_pass"] = bool(guardrails)
            v["production_147_profit_pass"] = bool(guardrails and int(v["total_KRW"]) > 0)
            if guardrails:
                guardrail_passed.append(v)
            if guardrails and int(v["total_KRW"]) > 0:
                production_passed.append(v)

        best_guardrail = (
            max(guardrail_passed, key=lambda x: int(x["total_KRW"]))
            if guardrail_passed
            else None
        )
        production_candidate = (
            max(production_passed, key=lambda x: int(x["total_KRW"]))
            if production_passed
            else None
        )

        def day_delta(r: dict, key: str) -> dict:
            a = int((r.get("CONTROL_D2") or {}).get("pnl_KRW", 0) or 0)
            b = int((r.get(key) or {}).get("pnl_KRW", 0) or 0)
            return {
                "date": r.get("date"),
                "control_KRW": a,
                "variant_KRW": b,
                "delta_KRW": b - a,
            }

        payload = {
            "ok": True,
            "version": D4_VERSION,
            "mode": "PATH_CONSISTENT_D4_LOSS_ROUTER",
            "read_only": True,
            "period": result.get("period", {}),
            "completed_at": datetime.now(KST).isoformat(timespec="seconds"),
            "days_expected": len(daily_base),
            "days_completed": len(rows),
            "errors": errors,
            "parity": {
                "required": True,
                "ok": parity_ok,
                "cached_D2_expected_total_KRW": int(
                    ((result.get("overall") or {}).get("D2_total_KRW", 0)) or 0
                ),
                "control_total_KRW": int(control["total_KRW"]),
                "total_delta_KRW": int(
                    control["total_KRW"]
                    - int(((result.get("overall") or {}).get("D2_total_KRW", 0) or 0))
                ),
                "mismatch_days": parity_mismatches,
                "rule": "CONTROL must match frozen cached D-v2 day-by-day before any D4 conclusion is accepted.",
            },
            "variants": variants,
            "profit_preservation": preservation,
            "profitable_month_preservation": month_pres,
            "monthly": monthly,
            "best_candidate_if_guardrails_pass": best_guardrail,
            "production_candidate_if_147_profit_positive": production_candidate,
            "acceptance": {
                "parity_required": True,
                "must_improve_total_KRW": True,
                "must_reduce_MDD": True,
                "positive_day_profit_preservation_min_pct": 90.0,
                "gross_profit_preservation_min_pct": 95.0,
                "each_original_profitable_month_min_pct": 85.0,
                "production_candidate_requires_147_total_KRW_above_zero": True,
            },
            "top_improved_days": {
                k: sorted(
                    [day_delta(r, k) for r in rows],
                    key=lambda x: x["delta_KRW"],
                    reverse=True,
                )[:12]
                for k in keys
            },
            "top_worsened_days": {
                k: sorted(
                    [day_delta(r, k) for r in rows],
                    key=lambda x: x["delta_KRW"],
                )[:12]
                for k in keys
            },
            "rules": {
                "CONTROL": "frozen D-v2",
                "LOSS_TRAPS": "ordinary BUY1 only: STOP re-entry ban; VWAP<=-1.5 recovery breakout; VWAP>=+3.5 pullback then re-breakout",
                "PG2_REARM": "LOSS_TRAPS + PROFIT_GUARD2 10m/fresh-setup re-arm; TAKE2 re-entry unchanged",
                "LOSS_ROUTER": "PG2_REARM + RED market only non-A+ requires fresh breakout; A+ direct allowed",
                "ROUTER_BRAKE": "LOSS_ROUTER + after 2 STOP_LOSS or realized <= -100,000 KRW stop all new buys for the day",
                "exit_engine": "identical frozen D-v2 STOP/TAKE1/TAKE2/PROFIT_GUARD",
                "generic_downsizing": False,
                "new_early_sell_rules": False,
            },
            "daily": rows,
            "important_limit": (
                "Historical candidate selection uses the same fixed liquidity universe as the frozen D-v2 replay, "
                "not exact historical whole-market KIS TOP5. Fees/taxes remain excluded. "
                "This is strategy-development validation, not a promise of live performance."
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
            message="D4 LOSS ROUTER + BAD DAY BRAKE 전체엔진 검증 완료",
            last_error="",
        )
    except Exception as exc:
        _state(
            status="error",
            phase="ERROR",
            result_ready=_paths()[3].exists(),
            last_error=f"{type(exc).__name__}: {exc}"[:1200],
            message="D4 LOSS ROUTER 검증 오류",
        )


def ensure_d4_loss_router_started(
    result: dict,
    provider,
    codes,
    frozen_config,
    protected_window_fn,
) -> dict:
    global _THREAD
    _, _, _, result_file = _paths()
    existing = _read_json(result_file, {}) or {}
    base_total = int(((result.get("overall") or {}).get("D2_total_KRW", 0)) or 0)

    if (
        existing.get("ok") is True
        and existing.get("version") == D4_VERSION
        and int(
            ((existing.get("parity") or {}).get("cached_D2_expected_total_KRW", base_total))
            or base_total
        )
        == base_total
    ):
        compact = dict(existing)
        compact.pop("daily", None)
        return compact

    with _LOCK:
        if _THREAD and _THREAD.is_alive():
            return _public_state()

        _THREAD = threading.Thread(
            target=_job,
            args=(
                dict(result),
                provider,
                list(codes),
                frozen_config,
                protected_window_fn,
            ),
            daemon=True,
            name="kr-d4-loss-router-full-engine",
        )
        _THREAD.start()
        state = _public_state()
        state["started"] = True
        return state

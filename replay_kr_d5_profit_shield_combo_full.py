from __future__ import annotations

"""D5 PROFIT SHIELD COMBO full-engine replay.

Single-strategy test that combines the strongest lessons from the old
Profit-Preserving Shield and D4 LOSS ROUTER while prioritizing upside preservation.

Principles
- Frozen D-v2 opening and profit engine stay unchanged.
- No blanket RED-market ban and no whole-day BAD DAY BRAKE.
- Same-day re-entry after STOP_LOSS is blocked.
- PROFIT_GUARD2 re-entry is not hard-blocked: fresh breakout gets full size;
  otherwise an eligible signal may probe at 25%.
- Deep-below-VWAP rebound traps and extreme overheat are reduced to 25%, not banned.
- A normal BUY1 is trimmed 50% only after a strict multi-signal failure: loss,
  below VWAP, weak short momentum, lost leadership, and RED market all agree.
- Remaining shares keep the frozen D-v2 -3% STOP, TAKE1, TAKE2, PROFIT_GUARD.

No real orders are sent. Cached KIS 1-minute bars only; no future bars.
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
D5_VERSION = "kr-d5-profit-shield-combo-fast-v1"

_THREAD: threading.Thread | None = None
_LOCK = threading.RLock()

# FAST v2: share repeated price/VWAP lookups across variants for the same day.
# D3 already shares TOP5 and market-regime reconstruction; these caches remove
# another large block of repeated pandas slicing without changing any decision.
_FAST_CACHE_DATE = ""
_FAST_PRICE_CACHE: dict[tuple[str, str], float] = {}
_FAST_VWAP_CACHE: dict[tuple[str, str], float] = {}


def _reset_fast_cache_if_needed(date_text: str) -> None:
    global _FAST_CACHE_DATE, _FAST_PRICE_CACHE, _FAST_VWAP_CACHE
    if _FAST_CACHE_DATE != date_text:
        _FAST_CACHE_DATE = date_text
        _FAST_PRICE_CACHE = {}
        _FAST_VWAP_CACHE = {}


def _price_cached(
    frame: pd.DataFrame,
    date_text: str,
    now: pd.Timestamp,
    symbol: str,
) -> float:
    _reset_fast_cache_if_needed(date_text)
    key = (str(symbol), now.isoformat())
    if key not in _FAST_PRICE_CACHE:
        _FAST_PRICE_CACHE[key] = float(
            replay_kr._price_at(frame, date_text, now) or 0.0
        )
    return float(_FAST_PRICE_CACHE[key])


def _vwap_cached(
    frame: pd.DataFrame,
    date_text: str,
    now: pd.Timestamp,
    symbol: str,
    ref_price: float,
) -> float:
    _reset_fast_cache_if_needed(date_text)
    key = (str(symbol), now.isoformat())
    if key not in _FAST_VWAP_CACHE:
        _FAST_VWAP_CACHE[key] = float(
            d2._vwap_gap_pct(frame, date_text, now, ref_price)
        )
    return float(_FAST_VWAP_CACHE[key])


def _base():
    import replay_kr_long_backtest as base
    return base


def _paths():
    base = _base()
    root = base.ROOT / "d5_profit_shield_combo_full_engine"
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
        cur["version"] = D5_VERSION
        cur["updated_at"] = datetime.now(KST).isoformat(timespec="seconds")
        _atomic_write_json(state_file, cur)
        return cur


def _public_state() -> dict:
    _, _, state_file, _ = _paths()
    state = _read_json(state_file, {}) or {}
    if not state:
        state = {
            "ok": True,
            "version": D5_VERSION,
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



def _pack_frozen_control(day_payload: dict | None, fallback_pnl: int) -> dict:
    """Pack the already-verified frozen D-v2 day instead of replaying it 147 times."""
    p = day_payload or {}
    events = [
        e for e in list(p.get("d_events", []) or [])
        if isinstance(e, dict)
    ]
    sm = p.get("d_summary", {}) or {}
    sell_realized = [
        int(e.get("실현손익KRW", 0) or 0)
        for e in events
        if str(e.get("구분", "")).upper() == "SELL"
    ]
    wins = [v for v in sell_realized if v > 0]
    losses = [v for v in sell_realized if v < 0]
    actions: dict[str, int] = {}
    for e in events:
        a = str(e.get("액션", "") or "")
        if a:
            actions[a] = actions.get(a, 0) + 1

    pnl = int(sm.get("실현손익KRW", fallback_pnl) or fallback_pnl)
    # The long-backtest daily row is the canonical frozen number.
    pnl = int(fallback_pnl)

    return {
        "pnl_KRW": pnl,
        "buy_orders": int(sm.get("매수주문횟수", sum(1 for e in events if str(e.get("구분","")).upper()=="BUY")) or 0),
        "sell_orders": int(sm.get("매도주문횟수", sum(1 for e in events if str(e.get("구분","")).upper()=="SELL")) or 0),
        "total_orders": int(sm.get("총주문횟수", len(events)) or len(events)),
        "traded_symbols": int(sm.get("거래종목수", len({str(e.get("종목코드","")) for e in events if e.get("종목코드")})) or 0),
        "gross_profit_KRW": int(sum(wins)),
        "gross_loss_KRW": int(sum(losses)),
        "positive_sell_events": len(wins),
        "negative_sell_events": len(losses),
        "take1_events": int(actions.get("TAKE1", 0)),
        "take2_events": int(actions.get("TAKE2", 0)),
        "profit_guard1_events": int(actions.get("PROFIT_GUARD1", 0)),
        "profit_guard2_events": int(actions.get("PROFIT_GUARD2", 0)),
        "stop_loss_events": int(actions.get("STOP_LOSS", 0)),
        "diagnostic": p.get("diagnostic", {}) or {},
        "source": "frozen_D2_cached_day_result",
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


def run_kr_d5_profit_shield_combo_replay(
    date_text: str,
    codes: Iterable[str] | None = None,
    config: d2.OpenDefenseConfig | None = None,
    mode: str = "CONTROL",
) -> dict:
    """Run one day of D-v2 with a profit-preserving loss shield.

    CONTROL intentionally follows the proven full-engine D-v2 path exactly.
    GENTLE and FULL change only risk sizing/defense behavior; TAKE1, TAKE2 and the
    original PROFIT_GUARD thresholds remain unchanged.
    """
    cfg = config or d2.OpenDefenseConfig()
    mode = str(mode or "CONTROL").upper().strip()
    if mode not in {"CONTROL", "D5_COMBO"}:
        raise ValueError("mode must be CONTROL or D5_COMBO")
    shield = mode == "D5_COMBO"
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

    normal_amount = d2._normal_entry_amount(cfg)       # current D-v2 50%
    defense_amount = d2._defense_entry_amount(cfg)     # current D-v2 25%
    confirm_amount = max(0, normal_amount - defense_amount)

    positions: dict[str, dict] = {}
    events: list[dict] = []
    latest_top5 = pd.DataFrame()
    last_scan = None
    daily_buy_amount = 0.0
    daily_orders = 0

    # D5 state: preserve participation but remember proven failure/re-entry context.
    pg2_disarmed_at: dict[str, pd.Timestamp] = {}
    stopped_symbols: set[str] = set()
    leader_streaks: dict[str, int] = {}
    last_market_snapshot: dict = {}

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
        "stop_reentry_blocks": 0,
        "pg2_disarm_activations": 0,
        "pg2_full_rearm_entries": 0,
        "pg2_probe_25pct_entries": 0,
        "deep_red_probe_25pct_entries": 0,
        "overheat_probe_25pct_entries": 0,
        "selective_shield_half_exits": 0,
        "selective_shield_small_qty_skips": 0,
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

            if shield:
                last_market_snapshot = d3._market_snapshot_cached(
                    target_frames, date_text, now, latest_top5
                )

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
            ref_price = _price_cached(frame, date_text, now, symbol)
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
            vg = _vwap_cached(frame, date_text, now, symbol, ref_price)
            row = top5_map.get(symbol)
            state = d3._row_state(row)

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
                hold_min = d3._hold_minutes(pos, now)
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
                        fail = d3._fail_signals(cfg, state, vg)
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

            # D5 selective shield: intervene only after multiple independent
            # failure signals agree. No generic -1% trim and no -1.8% full exit.
            is_normal_buy1 = str(pos.get("entry_action", "")) == "BUY1"
            if shield and is_normal_buy1 and not bool(pos.get("d5_shield_reduced", False)):
                hold_n = d3._hold_minutes(pos, now)
                market_red = bool((last_market_snapshot or {}).get("regime") == "RED")
                leader_lost = bool(
                    state["rank"] > 3
                    or state["score"] < 58.0
                    or state["weak"]
                    or not state["signal"]
                )
                strict_fail = bool(
                    hold_n >= 4.0
                    and pnl <= -1.20
                    and vg <= -0.25
                    and state["ret3"] <= -0.15
                    and state["ret5"] <= 0.0
                    and leader_lost
                    and market_red
                )
                if strict_fail:
                    if qty <= 1:
                        diagnostic["selective_shield_small_qty_skips"] += 1
                        pos["d5_shield_reduced"] = True
                    else:
                        sell_qty = max(1, qty // 2)
                        sell_qty = min(sell_qty, qty - 1)
                        fill = replay_kr._fill_price(c_cfg, "SELL", ref_price)
                        realized = (fill - avg) * sell_qty
                        add_event(
                            now, symbol, "D5_SELECTIVE_SHIELD_50", "SELL", sell_qty,
                            ref_price, fill,
                            f"D5 선택형 절반방어 · 보유 {hold_n:.1f}분 · pnl {pnl:.2f}% · "
                            f"VWAP {vg:+.2f}% · 3/5분 {state['ret3']:+.2f}/{state['ret5']:+.2f}% · RED",
                            pnl, realized, vwap_gap=vg,
                        )
                        pos["qty"] = qty - sell_qty
                        pos["d5_shield_reduced"] = True
                        diagnostic["selective_shield_half_exits"] += 1
                        continue

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
                    if shield:
                        stopped_symbols.add(symbol)
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
                positions.pop(symbol, None)
                continue

            suppress_profit_guard = False
            if bool(pos.get("defense_position", False)) and not bool(pos.get("open_confirmed", False)):
                suppress_profit_guard = (
                    d3._hold_minutes(pos, now) < float(cfg.profit_guard_confirm_grace_minutes)
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
                ref_price = _price_cached(frame, date_text, now, symbol)
                if ref_price <= 0:
                    continue
                vg = _vwap_cached(frame, date_text, now, symbol, ref_price)
                state = d3._row_state(row)

                # Opening 09:09~09:20 is frozen D-v2 behavior.
                if defense_now:
                    strong_open, _ = d2._is_strong_open(cfg, row, vg)
                    if strong_open:
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
                        sizing_notes: list[str] = []

                        # The clearest repeated loss shape: same-day STOP_LOSS re-entry.
                        if symbol in stopped_symbols:
                            diagnostic["stop_reentry_blocks"] += 1
                            continue

                        # PG2: preserve participation. Fresh breakout gets normal 50%;
                        # otherwise the next eligible signal may probe at only 25%.
                        disarmed = pg2_disarmed_at.get(symbol)
                        if disarmed is not None:
                            streak = int(leader_streaks.get(symbol, 0))
                            fresh_breakout, _ = d3._breakout_allowed(
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
                            if fresh_breakout and rearm_quality:
                                pg2_disarmed_at.pop(symbol, None)
                                diagnostic["pg2_full_rearm_entries"] += 1
                                sizing_notes.append("PG2_FRESH_BREAKOUT_50")
                            else:
                                target_amount = defense_amount
                                diagnostic["pg2_probe_25pct_entries"] += 1
                                sizing_notes.append("PG2_PROBE_25")

                        snap = last_market_snapshot or d3._market_snapshot_cached(
                            target_frames, date_text, now, latest_top5
                        )
                        market_red = bool(snap.get("regime") == "RED")

                        # Deep rebound trap: only reduce when the broad market is RED.
                        # A positive 3/5m bounce from <=-1.5% VWAP is treated as a probe,
                        # not banned, so a genuine recovery can still participate.
                        deep_red_rebound = bool(
                            market_red
                            and vg <= -1.50
                            and state["ret3"] > 0.0
                            and state["ret5"] > 0.0
                        )
                        if deep_red_rebound:
                            target_amount = defense_amount
                            diagnostic["deep_red_probe_25pct_entries"] += 1
                            sizing_notes.append("DEEP_RED_REBOUND_25")

                        # Extreme chase only: much narrower than old Shield/D4.
                        extreme_overheat = bool(vg >= 4.0 and state["ret5"] >= 3.0)
                        if extreme_overheat:
                            target_amount = defense_amount
                            diagnostic["overheat_probe_25pct_entries"] += 1
                            sizing_notes.append("EXTREME_OVERHEAT_25")

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
                elif target_amount == defense_amount:
                    prefix = "D5 COMBO PROBE 25%"
                else:
                    prefix = "D5 COMBO BUY1 50%"
                if shield and not defense_now and sizing_notes:
                    prefix += " · " + ",".join(sizing_notes)

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
                    "d5_shield_reduced": False,
                }
                daily_buy_amount += cost

                if action == "OPEN_STRONG_BUY":
                    diagnostic["strong_open_entries"] += 1
                elif action == "OPEN_DEFENSE_BUY":
                    diagnostic["defense_entries"] += 1
                else:
                    diagnostic["normal_entries"] += 1

        now += pd.Timedelta(seconds=int(cfg.manage_seconds))

    # Safety liquidation if anything somehow remains after the normal force-exit loop.
    if positions:
        for symbol, pos in list(positions.items()):
            frame = target_frames.get(symbol)
            ref_price = _price_cached(frame, date_text, end, symbol) if frame is not None else 0.0
            if ref_price <= 0:
                continue
            qty = int(pos.get("qty", 0))
            avg = d2._safe_float(pos.get("avg_price", 0))
            fill = replay_kr._fill_price(c_cfg, "SELL", ref_price)
            pnl = (ref_price / avg - 1.0) * 100.0 if avg > 0 else 0.0
            realized = (fill - avg) * qty
            vg = _vwap_cached(frame, date_text, end, symbol, ref_price)
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
        "version": D5_VERSION,
        "date": date_text,
        "strategy": f"D5_PROFIT_SHIELD_COMBO_{mode}",
        "mode": mode,
        "diagnostic": diagnostic,
        "summary": summary,
        "events": events,
        "config": asdict(cfg),
        "rules": {
            "opening": "frozen D-v2 09:09~09:20",
            "profit_engine": "frozen D-v2 STOP -3%, TAKE1 +3%, TAKE2 +5%, PROFIT_GUARD unchanged",
            "stop_reentry": "same-day ordinary BUY1 re-entry after STOP_LOSS blocked",
            "pg2": "no hard lock; fresh breakout+quality => 50%, otherwise eligible probe => 25%",
            "deep_red_rebound": "RED + VWAP<=-1.5 + positive 3/5m bounce => 25% probe, not ban",
            "extreme_overheat": "VWAP>=+4.0 and ret5>=+3.0 => 25% probe, not ban",
            "selective_shield": "normal BUY1 only: hold>=4m, pnl<=-1.2, VWAP<=-0.25, ret3<=-0.15, ret5<=0, leadership lost, RED => sell ~50%; remainder uses frozen exits",
            "red_blanket_entry_ban": False,
            "bad_day_full_brake": False,
            "generic_late_day_downsizing": False,
            "pre_guard": False,
        },
        "assumptions": {
            "real_orders": False,
            "future_data_visible": False,
            "fees_taxes": "별도 미포함",
            "purpose": "D5 single combined strategy: preserve D-v2 upside while selectively reducing proven loss shapes",
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
            r for r in list(result.get("daily", []) or [])
            if isinstance(r, dict) and r.get("date")
        ]
        daily_base.sort(key=lambda r: str(r.get("date")))
        if not daily_base:
            raise RuntimeError("기존 147일 daily 결과가 없습니다.")

        replay_kr._download_intraday = provider
        d2._download_intraday = provider

        _state(
            status="running",
            phase="D5_PROFIT_SHIELD_COMBO",
            started_at=datetime.now(KST).isoformat(timespec="seconds"),
            total_days=len(daily_base),
            completed_days=0,
            progress_pct=0.0,
            result_ready=False,
            error_days=0,
            message="D5 단일 COMBO · cached CONTROL + D5_COMBO 시작",
            last_error="",
        )

        rows: list[dict] = []
        errors: list[dict] = []
        parity_mismatches: list[dict] = []
        available_dates = [str(r.get("date")) for r in daily_base]
        sentinel_candidates = [
            available_dates[0] if available_dates else "",
            "2026-01-30",
            "2026-05-08",
            "2026-06-23",
            "2026-07-28",
            available_dates[-1] if available_dates else "",
        ]
        available_set = set(available_dates)
        sentinel_dates = {d for d in sentinel_candidates if d in available_set}
        control_spot_parity: list[dict] = []

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
                    message="실시간 자동매매 보호를 위해 D5 검증 일시정지",
                )
                time.sleep(30.0)

            _state(
                status="running",
                phase="D5_PROFIT_SHIELD_COMBO",
                current_date=date_text,
                completed_days=len(rows),
                error_days=len(errors),
                total_days=len(daily_base),
                progress_pct=round(100.0 * (idx - 1) / len(daily_base), 1),
                message=f"D5 COMBO {idx}/{len(daily_base)} · {date_text}",
            )

            day_path = _day_path(date_text)
            cached = base._load_gzip_json(day_path, None)
            if isinstance(cached, dict) and cached.get("version") == D5_VERSION:
                row = cached
            else:
                try:
                    frozen_day = base._load_day_result(date_text)
                    cached_d2 = int(base_row.get("D2_KRW", 0) or 0)
                    control_pack = _pack_frozen_control(frozen_day, cached_d2)
                    combo = run_kr_d5_profit_shield_combo_replay(
                        date_text=date_text,
                        codes=codes,
                        config=frozen_config,
                        mode="D5_COMBO",
                    )

                    if date_text in sentinel_dates:
                        audit = run_kr_d5_profit_shield_combo_replay(
                            date_text=date_text,
                            codes=codes,
                            config=frozen_config,
                            mode="CONTROL",
                        )
                        audit_pnl = int((audit.get("summary") or {}).get("실현손익KRW", 0) or 0)
                        control_spot_parity.append({
                            "date": date_text,
                            "cached_D2_KRW": cached_d2,
                            "replayed_CONTROL_KRW": audit_pnl,
                            "delta_KRW": int(audit_pnl - cached_d2),
                        })

                    row = {
                        "version": D5_VERSION,
                        "date": date_text,
                        "cached_D2_KRW": cached_d2,
                        "CONTROL_D2": control_pack,
                        "D5_COMBO": _pack(combo),
                    }
                    row["parity_delta_KRW"] = int(
                        row["CONTROL_D2"]["pnl_KRW"] - row["cached_D2_KRW"]
                    )
                    row["D5_COMBO"]["delta_vs_control_KRW"] = int(
                        row["D5_COMBO"]["pnl_KRW"] - row["CONTROL_D2"]["pnl_KRW"]
                    )
                    base._save_gzip_json(day_path, row)
                except Exception as exc:
                    errors.append({"date": date_text, "error": f"{type(exc).__name__}: {exc}"})
                    _state(last_error=errors[-1]["error"][:1000])
                    continue

            rows.append(row)

            if date_text in sentinel_dates and not any(x.get("date") == date_text for x in control_spot_parity):
                try:
                    audit = run_kr_d5_profit_shield_combo_replay(
                        date_text=date_text,
                        codes=codes,
                        config=frozen_config,
                        mode="CONTROL",
                    )
                    audit_pnl = int((audit.get("summary") or {}).get("실현손익KRW", 0) or 0)
                    cached_d2 = int(base_row.get("D2_KRW", 0) or 0)
                    control_spot_parity.append({
                        "date": date_text,
                        "cached_D2_KRW": cached_d2,
                        "replayed_CONTROL_KRW": audit_pnl,
                        "delta_KRW": int(audit_pnl - cached_d2),
                    })
                except Exception as exc:
                    control_spot_parity.append({
                        "date": date_text,
                        "cached_D2_KRW": int(base_row.get("D2_KRW", 0) or 0),
                        "error": f"{type(exc).__name__}: {exc}",
                        "delta_KRW": None,
                    })

            delta = int(row.get("parity_delta_KRW", 0) or 0)
            if delta != 0:
                parity_mismatches.append({
                    "date": date_text,
                    "cached_D2_KRW": int(row.get("cached_D2_KRW", 0) or 0),
                    "control_D2_KRW": int((row.get("CONTROL_D2") or {}).get("pnl_KRW", 0) or 0),
                    "delta_KRW": delta,
                })

            _state(
                status="running",
                phase="D5_PROFIT_SHIELD_COMBO",
                current_date=date_text,
                completed_days=len(rows),
                error_days=len(errors),
                total_days=len(daily_base),
                progress_pct=round(100.0 * idx / len(daily_base), 1),
                message=f"D5 COMBO {idx}/{len(daily_base)} 완료",
            )

        control = _aggregate("D-v2 원본 · 검증된 캐시 재사용", "CONTROL_D2", rows)
        combo = _aggregate("D5 PROFIT SHIELD COMBO · 단일 결합전략", "D5_COMBO", rows)
        combo["delta_vs_control_KRW"] = int(combo["total_KRW"] - control["total_KRW"])

        spot_parity_ok = bool(control_spot_parity) and all(
            x.get("delta_KRW") == 0 for x in control_spot_parity
        )
        parity_ok = (
            len(rows) == len(daily_base)
            and not errors
            and not parity_mismatches
            and spot_parity_ok
        )

        preservation = {"D5_COMBO": _profit_preservation(rows, "D5_COMBO")}
        monthly = {
            "CONTROL_D2": _monthly(rows, "CONTROL_D2"),
            "D5_COMBO": _monthly(rows, "D5_COMBO"),
        }
        month_pres = {
            "D5_COMBO": _profitable_month_preservation(monthly["D5_COMBO"])
        }
        guardrails = bool(
            parity_ok
            and int(combo["delta_vs_control_KRW"]) > 0
            and int(combo["max_cumulative_drawdown_KRW"]) < int(control["max_cumulative_drawdown_KRW"])
            and preservation["D5_COMBO"]["passes_positive_day_90pct"]
            and preservation["D5_COMBO"]["passes_gross_profit_95pct"]
            and month_pres["D5_COMBO"]["passes_85pct_each_profitable_month"]
        )
        combo["guardrails_pass"] = guardrails
        combo["target_loss_under_1m_pass"] = bool(int(combo["total_KRW"]) >= -1_000_000)
        combo["production_147_profit_pass"] = bool(guardrails and int(combo["total_KRW"]) > 0)

        def day_delta(r: dict) -> dict:
            a = int((r.get("CONTROL_D2") or {}).get("pnl_KRW", 0) or 0)
            b = int((r.get("D5_COMBO") or {}).get("pnl_KRW", 0) or 0)
            return {"date": r.get("date"), "control_KRW": a, "variant_KRW": b, "delta_KRW": b-a}

        payload = {
            "ok": True,
            "version": D5_VERSION,
            "mode": "PATH_CONSISTENT_D5_SINGLE_COMBO_FAST",
            "read_only": True,
            "executed_variants": ["D5_COMBO"],
            "period": result.get("period", {}),
            "completed_at": datetime.now(KST).isoformat(timespec="seconds"),
            "days_expected": len(daily_base),
            "days_completed": len(rows),
            "errors": errors,
            "parity": {
                "required": True,
                "ok": parity_ok,
                "cached_D2_expected_total_KRW": int(((result.get("overall") or {}).get("D2_total_KRW", 0)) or 0),
                "control_total_KRW": int(control["total_KRW"]),
                "total_delta_KRW": int(
                    control["total_KRW"]
                    - int(((result.get("overall") or {}).get("D2_total_KRW", 0)) or 0)
                ),
                "mismatch_days": parity_mismatches,
                "control_spot_audit": control_spot_parity,
                "spot_audit_ok": spot_parity_ok,
            },
            "variants": [control, combo],
            "profit_preservation": preservation,
            "profitable_month_preservation": month_pres,
            "monthly": monthly,
            "acceptance": {
                "must_improve_total_KRW": True,
                "must_reduce_MDD": True,
                "positive_day_profit_preservation_min_pct": 90.0,
                "gross_profit_preservation_min_pct": 95.0,
                "each_original_profitable_month_min_pct": 85.0,
                "first_target_total_KRW": -1_000_000,
                "production_candidate_requires_147_total_KRW_above_zero": True,
            },
            "top_improved_days": sorted([day_delta(r) for r in rows], key=lambda x: x["delta_KRW"], reverse=True)[:15],
            "top_worsened_days": sorted([day_delta(r) for r in rows], key=lambda x: x["delta_KRW"])[:15],
            "rules": {
                "opening": "frozen D-v2",
                "profit_engine": "frozen D-v2 STOP/TAKE1/TAKE2/PROFIT_GUARD",
                "stop_reentry": "blocked same day",
                "pg2": "fresh breakout 50%; otherwise 25% probe; no hard lock",
                "deep_red_rebound": "25% probe only",
                "extreme_overheat": "25% probe only",
                "selective_shield": "strict RED+VWAP+momentum+leadership failure => half trim only",
                "blanket_red_ban": False,
                "whole_day_brake": False,
            },
            "daily": rows,
            "important_limit": (
                "Historical candidate selection uses the same fixed liquidity universe as frozen D-v2, not exact historical whole-market KIS TOP5. "
                "Fees/taxes excluded. This is strategy-development validation, not a promise of live performance."
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
            message="D5 단일 PROFIT SHIELD COMBO 검증 완료",
            last_error="",
        )
    except Exception as exc:
        _state(
            status="error",
            phase="ERROR",
            result_ready=_paths()[3].exists(),
            last_error=f"{type(exc).__name__}: {exc}"[:1200],
            message="D5 PROFIT SHIELD COMBO 검증 오류",
        )


def ensure_d5_profit_shield_combo_started(
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
        and existing.get("version") == D5_VERSION
        and int(((existing.get("parity") or {}).get("cached_D2_expected_total_KRW", base_total)) or base_total) == base_total
    ):
        compact = dict(existing)
        compact.pop("daily", None)
        return compact

    with _LOCK:
        if _THREAD and _THREAD.is_alive():
            return _public_state()

        _THREAD = threading.Thread(
            target=_job,
            args=(dict(result), provider, list(codes), frozen_config, protected_window_fn),
            daemon=True,
            name="kr-d5-profit-shield-combo",
        )
        _THREAD.start()
        state = _public_state()
        state["started"] = True
        return state

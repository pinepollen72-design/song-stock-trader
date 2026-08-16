from __future__ import annotations

"""
D3-v2 HYBRID full-engine replay.

Goal
- Keep the frozen D-v2 opening engine unchanged.
- Preserve ordinary winning trades as much as possible.
- Apply 25% sizing only to clearly risky ordinary BUY1 entries.
- Add a deep structural-failure exit with winner immunity.
- After PROFIT_GUARD2, require a 5-minute lock + TOP3/VWAP/3m/5m re-arm.
- Read only the already cached KIS 1-minute bars supplied by replay_kr_long_backtest.
- Never send real/paper orders.

Acceptance gate
1) frozen D-v2 parity must match day-by-day
2) total PnL must improve
3) positive-day profit preservation >= 90%
4) gross-profit preservation >= 90%
5) every originally profitable month retains >= 85%
"""

import json
import os
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

import pandas as pd

import replay_kr
import replay_kr_open_defense_v2 as d2_replay

KST = ZoneInfo("Asia/Seoul")
D3_HYBRID_VERSION = "kr-d3-v2-hybrid-full-engine-v1"


def _resolve_state_dir() -> Path:
    explicit = os.getenv("SONG_TRADER_STATE_DIR", "").strip()
    if explicit:
        return Path(explicit)
    railway_volume = os.getenv("RAILWAY_VOLUME_MOUNT_PATH", "").strip()
    if railway_volume:
        return Path(railway_volume) / "song_trader_v2"
    return Path("/tmp/song_trader_v2")


ROOT = _resolve_state_dir() / "replays" / "kr_d3_hybrid"
DAY_DIR = ROOT / "daily"
STATE_FILE = ROOT / "state.json"
RESULT_FILE = ROOT / "result.json"
for _p in (ROOT, DAY_DIR):
    _p.mkdir(parents=True, exist_ok=True)

_LOCK = threading.RLock()
_THREAD: threading.Thread | None = None


@dataclass(frozen=True)
class HybridRules:
    # Ordinary entry size. Opening D-v2 keeps its own frozen rules.
    normal_entry_pct: float = 0.50
    risky_entry_pct: float = 0.25

    # SURGICAL95-style selective downsizing.
    vwap_risk_cutoff_pct: float = -0.50
    overheat_vwap_gap_pct: float = 3.00

    # Risk-sized entries can restore to normal size only after real recovery.
    risk_confirm_min_hold_minutes: float = 4.0
    risk_confirm_min_score: float = 65.0
    risk_confirm_max_rank: int = 3
    risk_confirm_require_vwap: bool = True
    risk_confirm_require_ret3_nonnegative: bool = True
    risk_confirm_require_ret5_positive: bool = True
    # Do not chase a recovered 25% probe after it has already entered the
    # original D-v2 profit-protection zone.
    risk_confirm_max_pnl_pct: float = 1.20

    # SURGICAL95 deep structural-failure exit.
    structural_min_hold_minutes: float = 5.0
    structural_loss_pct: float = -2.40
    structural_min_failures: int = 3
    winner_immunity_peak_pct: float = 0.80
    winner_failure_loss_pct: float = -2.70
    winner_failure_min_failures: int = 4
    leader_fail_score: float = 58.0

    # PROFIT_GUARD2 re-arm.
    pg2_lock_minutes: float = 5.0
    pg2_rearm_max_rank: int = 3
    pg2_rearm_min_score: float = 50.0

    # Acceptance.
    profit_preservation_min_pct: float = 90.0
    gross_profit_preservation_min_pct: float = 90.0
    profitable_month_floor_pct: float = 85.0


RULES = HybridRules()


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _state(**updates) -> dict:
    with _LOCK:
        cur = _read_json(STATE_FILE, {}) or {}
        cur.update(updates)
        cur.setdefault("version", D3_HYBRID_VERSION)
        cur["updated_at"] = datetime.now(KST).isoformat(timespec="seconds")
        _atomic_write_json(STATE_FILE, cur)
        return cur


def _public_state() -> dict:
    state = _read_json(STATE_FILE, {}) or {}
    if not state:
        return {
            "ok": True,
            "version": D3_HYBRID_VERSION,
            "status": "not_started",
            "result_ready": False,
        }
    out = dict(state)
    out["ok"] = True
    out["thread_alive"] = bool(_THREAD and _THREAD.is_alive())
    return out


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _rank_number(row: Any) -> int:
    if row is None:
        return 999
    raw = str(row.get("순위", "") or "")
    digits = "".join(ch for ch in raw if ch.isdigit())
    try:
        return int(digits) if digits else 999
    except Exception:
        return 999


def _vwap_gap_pct(frame: pd.DataFrame, date_text: str, now: pd.Timestamp, price: float) -> float:
    # Use the exact frozen D-v2 VWAP helper so no future bar is visible.
    return float(d2_replay._vwap_gap_pct(frame, date_text, now, price))


def _max_drawdown(values: list[int]) -> int:
    equity = 0
    peak = 0
    max_dd = 0
    for pnl in values:
        equity += int(pnl)
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return int(max_dd)


def _entry_metrics(row: Any) -> dict[str, Any]:
    return {
        "score": _safe_float(row.get("종합점수", 0)) if row is not None else 0.0,
        "rank": _rank_number(row),
        "ret3": _safe_float(row.get("최근3분수익률", 0)) if row is not None else -999.0,
        "ret5": _safe_float(row.get("최근5분수익률", 0)) if row is not None else -999.0,
        "vol": _safe_float(row.get("거래량배수", 0)) if row is not None else 0.0,
        "weak": bool(row.get("모멘텀약화", False)) if row is not None else True,
        "signal_ok": "매수 후보" in str(row.get("판정", "")) if row is not None else False,
    }


def _ordinary_risky(row: Any, vwap_gap: float) -> tuple[bool, str]:
    m = _entry_metrics(row)
    weak_short = bool(m["weak"] or m["ret3"] < 0.0 or m["ret5"] <= 0.0)
    below_vwap_risk = vwap_gap <= RULES.vwap_risk_cutoff_pct and weak_short
    weak_overheat = vwap_gap >= RULES.overheat_vwap_gap_pct and bool(
        m["weak"] or m["ret3"] <= 0.0 or m["ret5"] <= 0.0
    )
    if below_vwap_risk:
        return True, (
            f"RISK_SIZE_VWAP · VWAP {vwap_gap:+.2f}% · "
            f"3/5분 {m['ret3']:+.2f}/{m['ret5']:+.2f}%"
        )
    if weak_overheat:
        return True, (
            f"RISK_SIZE_OVERHEAT · VWAP {vwap_gap:+.2f}% · "
            f"3/5분 {m['ret3']:+.2f}/{m['ret5']:+.2f}%"
        )
    return False, ""


def _recovery_confirm_allowed(pos: dict, row: Any, now: pd.Timestamp, pnl: float, vwap_gap: float) -> tuple[bool, str]:
    if row is None:
        return False, "TOP5 이탈"
    try:
        created = pd.Timestamp(pos.get("created_at"))
        if created.tzinfo is None:
            created = created.tz_localize(KST)
        hold_min = max(0.0, (now - created).total_seconds() / 60.0)
    except Exception:
        hold_min = 0.0
    m = _entry_metrics(row)
    checks = [
        hold_min >= RULES.risk_confirm_min_hold_minutes,
        m["signal_ok"],
        not m["weak"],
        m["score"] >= RULES.risk_confirm_min_score,
        m["rank"] <= RULES.risk_confirm_max_rank,
        pnl >= -0.10,
        pnl <= RULES.risk_confirm_max_pnl_pct,
    ]
    if RULES.risk_confirm_require_vwap:
        checks.append(vwap_gap >= 0.0)
    if RULES.risk_confirm_require_ret3_nonnegative:
        checks.append(m["ret3"] >= 0.0)
    if RULES.risk_confirm_require_ret5_positive:
        checks.append(m["ret5"] > 0.0)
    reason = (
        f"RISK_CONFIRM · 보유 {hold_min:.1f}분 · TOP{m['rank']} · 점수 {m['score']:.1f} · "
        f"3/5분 {m['ret3']:+.2f}/{m['ret5']:+.2f}% · VWAP {vwap_gap:+.2f}%"
    )
    return all(checks), reason


def _pg2_rearmed(row: Any, vwap_gap: float) -> tuple[bool, str]:
    m = _entry_metrics(row)
    ok = bool(
        m["signal_ok"]
        and not m["weak"]
        and m["score"] >= RULES.pg2_rearm_min_score
        and m["rank"] <= RULES.pg2_rearm_max_rank
        and vwap_gap >= 0.0
        and m["ret3"] >= 0.0
        and m["ret5"] > 0.0
    )
    return ok, (
        f"PG2_REARM · TOP{m['rank']} · 점수 {m['score']:.1f} · "
        f"3/5분 {m['ret3']:+.2f}/{m['ret5']:+.2f}% · VWAP {vwap_gap:+.2f}%"
    )


def _structural_failures(row: Any, vwap_gap: float) -> list[str]:
    m = _entry_metrics(row)
    failures: list[str] = []
    if vwap_gap < 0.0:
        failures.append("BELOW_VWAP")
    if m["ret3"] < 0.0:
        failures.append("RET3_NEG")
    if m["ret5"] <= 0.0:
        failures.append("RET5_NONPOS")
    if (
        not m["signal_ok"]
        or m["weak"]
        or m["rank"] > 3
        or m["score"] < RULES.leader_fail_score
    ):
        failures.append("LEADER_LOST")
    return failures


def _run_hybrid_day(
    date_text: str,
    codes: Iterable[str],
    frozen_cfg: d2_replay.OpenDefenseConfig,
) -> dict:
    cfg = frozen_cfg
    universe = replay_kr._normalize_universe(codes)
    frames, meta = replay_kr._download_intraday(date_text, universe)
    if not frames:
        raise RuntimeError("cached KIS minute bars unavailable")
    target_frames = {
        code: frame
        for code, frame in frames.items()
        if not frame[frame.index.strftime("%Y-%m-%d") == date_text].empty
    }
    if not target_frames:
        raise RuntimeError(f"{date_text} minute bars unavailable")

    # Fill model is identical to frozen D-v2.
    fill_cfg = replay_kr.KRReplayConfig(
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

    date0 = pd.Timestamp(date_text, tz=KST)
    start = date0 + pd.Timedelta(hours=9, minutes=9)
    end = date0 + pd.Timedelta(hours=15, minutes=16)
    defense_start_sec = replay_kr._clock_seconds(cfg.defense_start_time)
    defense_end_sec = replay_kr._clock_seconds(cfg.defense_end_time)
    last_entry_sec = replay_kr._clock_seconds(cfg.last_entry_time)
    force_exit_sec = replay_kr._clock_seconds(cfg.force_exit_time)

    # The full-size leg is taken from the frozen D-v2 helper rather than
    # re-derived, so opening/normal sizing stays exactly aligned with the
    # cached D-v2 configuration.
    normal_amount = int(d2_replay._normal_entry_amount(cfg))
    risky_amount = int(cfg.per_stock_budget_krw * RULES.risky_entry_pct)
    open_defense_amount = int(cfg.per_stock_budget_krw * cfg.defense_entry_pct_of_stock_budget)
    open_confirm_amount = max(0, normal_amount - open_defense_amount)
    risk_confirm_amount = max(0, normal_amount - risky_amount)

    positions: dict[str, dict] = {}
    events: list[dict] = []
    latest_top5 = pd.DataFrame()
    last_scan = None
    daily_buy_amount = 0.0
    daily_orders = 0
    pg2_block_until: dict[str, pd.Timestamp] = {}
    diag = {
        "strong_open_entries": 0,
        "open_defense_entries": 0,
        "open_confirms": 0,
        "ordinary_full_entries": 0,
        "risky_25pct_entries": 0,
        "risky_vwap_entries": 0,
        "risky_overheat_entries": 0,
        "risk_confirms": 0,
        "structural_exits": 0,
        "winner_immunity_blocks": 0,
        "winner_failure_exits": 0,
        "pg2_locks": 0,
        "pg2_lock_blocks": 0,
        "pg2_rearm_blocks": 0,
        "pg2_rearm_successes": 0,
        "stop_loss_events": 0,
        "take1_events": 0,
        "take2_events": 0,
        "profit_guard1_events": 0,
        "profit_guard2_events": 0,
    }

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
            latest_top5 = replay_kr._build_top5_at(
                target_frames, meta, date_text, now, cfg.scan_count
            )
            last_scan = now

        top5_map: dict[str, Any] = {}
        if latest_top5 is not None and not latest_top5.empty:
            for _, r in latest_top5.iterrows():
                top5_map[str(r.get("종목코드", "")).zfill(6)] = r

        # 1) Manage existing positions.
        for symbol, pos in list(positions.items()):
            frame = target_frames.get(symbol)
            if frame is None:
                continue
            ref_price = replay_kr._price_at(frame, date_text, now)
            if ref_price <= 0:
                continue
            qty = int(pos.get("qty", 0) or 0)
            avg = _safe_float(pos.get("avg_price", 0))
            if qty <= 0 or avg <= 0:
                continue
            pnl = (ref_price / avg - 1.0) * 100.0
            peak = max(_safe_float(pos.get("peak_pnl", pnl)), pnl)
            pos["peak_pnl"] = peak
            dd = max(0.0, peak - pnl)
            vg = _vwap_gap_pct(frame, date_text, now, ref_price)
            row = top5_map.get(symbol)

            if replay_kr._seconds_of_day(now) >= force_exit_sec:
                fill = replay_kr._fill_price(fill_cfg, "SELL", ref_price)
                realized = (fill - avg) * qty
                add_event(now, symbol, "FORCE_SELL", "SELL", qty, ref_price, fill,
                          f"당일 강제청산 {cfg.force_exit_time} KST", pnl, realized,
                          vwap_gap=vg)
                positions.pop(symbol, None)
                continue

            # Frozen D-v2 opening defense logic is kept unchanged.
            if bool(pos.get("opening_defense", False)):
                try:
                    created = pd.Timestamp(pos.get("created_at"))
                    if created.tzinfo is None:
                        created = created.tz_localize(KST)
                    hold_min = max(0.0, (now - created).total_seconds() / 60.0)
                except Exception:
                    hold_min = 0.0
                m = _entry_metrics(row)
                if hold_min <= float(cfg.defense_fail_window_minutes):
                    if pnl <= float(cfg.defense_emergency_fail_pct):
                        fill = replay_kr._fill_price(fill_cfg, "SELL", ref_price)
                        realized = (fill - avg) * qty
                        add_event(now, symbol, "OPEN_EMERGENCY_EXIT", "SELL", qty,
                                  ref_price, fill,
                                  f"장초반 비상청산 · pnl {pnl:.2f}% <= {cfg.defense_emergency_fail_pct:.2f}%",
                                  pnl, realized, vwap_gap=vg)
                        positions.pop(symbol, None)
                        continue
                    if hold_min >= float(cfg.soft_fail_min_hold_minutes) and pnl <= float(cfg.defense_soft_fail_pct):
                        fail_signals = []
                        if vg <= float(cfg.soft_fail_vwap_gap_pct):
                            fail_signals.append("BELOW_VWAP")
                        if m["ret3"] <= float(cfg.soft_fail_ret3_pct):
                            fail_signals.append("RET3_WEAK")
                        if m["ret5"] <= float(cfg.soft_fail_ret5_pct):
                            fail_signals.append("RET5_WEAK")
                        if m["score"] < float(cfg.soft_fail_score) or m["weak"] or not m["signal_ok"]:
                            fail_signals.append("LEADER_STRENGTH_LOST")
                        if len(fail_signals) >= int(cfg.soft_fail_min_signals):
                            fill = replay_kr._fill_price(fill_cfg, "SELL", ref_price)
                            realized = (fill - avg) * qty
                            add_event(now, symbol, "OPEN_SOFT_FAIL_EXIT", "SELL", qty,
                                      ref_price, fill,
                                      f"복합 돌파실패 · pnl {pnl:.2f}% · " + ",".join(fail_signals),
                                      pnl, realized, vwap_gap=vg)
                            positions.pop(symbol, None)
                            continue

            # Risk-sized ordinary entry may recover back to the original 50%.
            if (
                bool(pos.get("ordinary_risky", False))
                and not bool(pos.get("risk_confirmed", False))
                and replay_kr._seconds_of_day(now) < last_entry_sec
                and daily_orders < cfg.max_daily_orders
            ):
                ok, why = _recovery_confirm_allowed(pos, row, now, pnl, vg)
                if ok and risk_confirm_amount > 0:
                    fill = replay_kr._fill_price(fill_cfg, "BUY", ref_price)
                    qty2 = int(risk_confirm_amount // fill)
                    cost = fill * qty2
                    if qty2 > 0 and daily_buy_amount + cost <= cfg.daily_budget_krw:
                        old_qty = int(pos["qty"])
                        old_avg = _safe_float(pos["avg_price"])
                        new_qty = old_qty + qty2
                        new_avg = (old_avg * old_qty + fill * qty2) / new_qty
                        m = _entry_metrics(row)
                        add_event(now, symbol, "RISK_CONFIRM", "BUY", qty2,
                                  ref_price, fill, why, pnl, 0.0,
                                  m["score"], f"TOP{m['rank']}", vg)
                        pos["qty"] = new_qty
                        pos["avg_price"] = new_avg
                        # A size add changes the cost basis. Reset the profit peak to
                        # the new basis so PROFIT_GUARD is not triggered by an
                        # artificial pre-add peak on the next 45-second cycle.
                        pos["peak_pnl"] = max(0.0, (ref_price / new_avg - 1.0) * 100.0)
                        pos["risk_confirmed"] = True
                        pos["ordinary_risky"] = False
                        daily_buy_amount += cost
                        diag["risk_confirms"] += 1
                        continue

            # Hybrid deep structural failure applies only after opening window.
            if bool(pos.get("ordinary_position", False)):
                try:
                    created = pd.Timestamp(pos.get("created_at"))
                    if created.tzinfo is None:
                        created = created.tz_localize(KST)
                    hold_min = max(0.0, (now - created).total_seconds() / 60.0)
                except Exception:
                    hold_min = 0.0
                failures = _structural_failures(row, vg)
                winner = peak >= RULES.winner_immunity_peak_pct
                should_exit = False
                action = ""
                if hold_min >= RULES.structural_min_hold_minutes:
                    if not winner and pnl <= RULES.structural_loss_pct and len(failures) >= RULES.structural_min_failures:
                        should_exit = True
                        action = "HYBRID_STRUCT_EXIT"
                    elif winner:
                        if pnl <= RULES.winner_failure_loss_pct and len(failures) >= RULES.winner_failure_min_failures:
                            should_exit = True
                            action = "HYBRID_WINNER_FAIL_EXIT"
                        elif pnl <= RULES.structural_loss_pct and len(failures) >= RULES.structural_min_failures:
                            diag["winner_immunity_blocks"] += 1
                if should_exit:
                    fill = replay_kr._fill_price(fill_cfg, "SELL", ref_price)
                    realized = (fill - avg) * qty
                    add_event(now, symbol, action, "SELL", qty, ref_price, fill,
                              f"구조실패 · peak {peak:+.2f}% · pnl {pnl:+.2f}% · " + ",".join(failures),
                              pnl, realized, vwap_gap=vg)
                    if action == "HYBRID_WINNER_FAIL_EXIT":
                        diag["winner_failure_exits"] += 1
                    else:
                        diag["structural_exits"] += 1
                    positions.pop(symbol, None)
                    continue

            # Frozen D-v2 stop-loss remains final hard stop.
            if pnl <= -abs(cfg.stop_loss_pct):
                fill = replay_kr._fill_price(fill_cfg, "SELL", ref_price)
                realized = (fill - avg) * qty
                add_event(now, symbol, "STOP_LOSS", "SELL", qty, ref_price, fill,
                          f"손절 {pnl:.2f}%", pnl, realized, vwap_gap=vg)
                diag["stop_loss_events"] += 1
                positions.pop(symbol, None)
                continue

            # Frozen D-v2 opening confirm stays before profit guard.
            if (
                bool(pos.get("opening_defense", False))
                and not bool(pos.get("open_confirmed", False))
                and replay_kr._seconds_of_day(now) < defense_end_sec
                and daily_orders < cfg.max_daily_orders
            ):
                ok, why = d2_replay._confirm_allowed(cfg, pos, row, now, pnl, vg)
                if ok and open_confirm_amount > 0:
                    fill = replay_kr._fill_price(fill_cfg, "BUY", ref_price)
                    qty2 = int(open_confirm_amount // fill)
                    cost = fill * qty2
                    if qty2 > 0 and daily_buy_amount + cost <= cfg.daily_budget_krw:
                        old_qty = int(pos["qty"])
                        old_avg = _safe_float(pos["avg_price"])
                        new_qty = old_qty + qty2
                        new_avg = (old_avg * old_qty + fill * qty2) / new_qty
                        m = _entry_metrics(row)
                        add_event(now, symbol, "OPEN_CONFIRM", "BUY", qty2,
                                  ref_price, fill, why, pnl, 0.0,
                                  m["score"], f"TOP{m['rank']}", vg)
                        pos["qty"] = new_qty
                        pos["avg_price"] = new_avg
                        pos["open_confirmed"] = True
                        daily_buy_amount += cost
                        diag["open_confirms"] += 1
                        continue

            if pnl >= cfg.take1_pct and not bool(pos.get("take1_sent")):
                sell_qty = max(1, qty // 2)
                fill = replay_kr._fill_price(fill_cfg, "SELL", ref_price)
                realized = (fill - avg) * sell_qty
                add_event(now, symbol, "TAKE1", "SELL", sell_qty, ref_price, fill,
                          f"1차 익절 {pnl:.2f}% · 약 50%", pnl, realized, vwap_gap=vg)
                pos["qty"] = qty - sell_qty
                pos["take1_sent"] = True
                diag["take1_events"] += 1
                if pos["qty"] <= 0:
                    positions.pop(symbol, None)
                continue

            if pnl >= cfg.take2_pct and bool(pos.get("take1_sent")):
                fill = replay_kr._fill_price(fill_cfg, "SELL", ref_price)
                realized = (fill - avg) * qty
                add_event(now, symbol, "TAKE2", "SELL", qty, ref_price, fill,
                          f"2차 익절 {pnl:.2f}% · 전량", pnl, realized, vwap_gap=vg)
                diag["take2_events"] += 1
                positions.pop(symbol, None)
                continue

            suppress_profit_guard = False
            if bool(pos.get("opening_defense", False)) and not bool(pos.get("open_confirmed", False)):
                try:
                    created_pg = pd.Timestamp(pos.get("created_at"))
                    if created_pg.tzinfo is None:
                        created_pg = created_pg.tz_localize(KST)
                    hold_pg = max(0.0, (now - created_pg).total_seconds() / 60.0)
                except Exception:
                    hold_pg = 0.0
                suppress_profit_guard = hold_pg < float(cfg.profit_guard_confirm_grace_minutes)

            if peak >= cfg.profit_guard_trigger_pct and dd >= cfg.profit_guard_drawdown_pct:
                if suppress_profit_guard:
                    continue
                if not bool(pos.get("take1_sent")):
                    sell_qty = max(1, qty // 2)
                    fill = replay_kr._fill_price(fill_cfg, "SELL", ref_price)
                    realized = (fill - avg) * sell_qty
                    add_event(now, symbol, "PROFIT_GUARD1", "SELL", sell_qty,
                              ref_price, fill,
                              f"수익보호 1차 · 최고 +{peak:.2f}% → 현재 {pnl:.2f}% ({dd:.2f}%p 되밀림)",
                              pnl, realized, vwap_gap=vg)
                    pos["qty"] = qty - sell_qty
                    pos["take1_sent"] = True
                    diag["profit_guard1_events"] += 1
                    if pos["qty"] <= 0:
                        positions.pop(symbol, None)
                    continue
                else:
                    fill = replay_kr._fill_price(fill_cfg, "SELL", ref_price)
                    realized = (fill - avg) * qty
                    add_event(now, symbol, "PROFIT_GUARD2", "SELL", qty,
                              ref_price, fill,
                              f"수익보호 2차 · 최고 +{peak:.2f}% → 현재 {pnl:.2f}% ({dd:.2f}%p 되밀림)",
                              pnl, realized, vwap_gap=vg)
                    diag["profit_guard2_events"] += 1
                    diag["pg2_locks"] += 1
                    pg2_block_until[symbol] = now + pd.Timedelta(minutes=RULES.pg2_lock_minutes)
                    positions.pop(symbol, None)
                    continue

        # 2) New entries.
        if (
            replay_kr._seconds_of_day(now) < last_entry_sec
            and len(positions) < cfg.max_positions
            and daily_orders < cfg.max_daily_orders
            and latest_top5 is not None
            and not latest_top5.empty
        ):
            sec = replay_kr._seconds_of_day(now)
            opening = defense_start_sec <= sec < defense_end_sec
            for _, row in latest_top5.iterrows():
                if len(positions) >= cfg.max_positions or daily_orders >= cfg.max_daily_orders:
                    break
                symbol = str(row.get("종목코드", "")).zfill(6)
                if not symbol or symbol in positions:
                    continue
                m = _entry_metrics(row)
                if not m["signal_ok"] or m["weak"] or m["score"] < cfg.min_score:
                    continue
                frame = target_frames.get(symbol)
                if frame is None:
                    continue
                ref_price = replay_kr._price_at(frame, date_text, now)
                if ref_price <= 0:
                    continue
                vg = _vwap_gap_pct(frame, date_text, now, ref_price)

                # Frozen D-v2 opening behavior exactly.
                if opening:
                    strong_open, _why = d2_replay._is_strong_open(cfg, row, vg)
                    if strong_open:
                        target_amount = normal_amount
                        action = "OPEN_STRONG_BUY"
                        opening_defense = False
                    else:
                        defense_count = sum(
                            1 for p in positions.values()
                            if bool(p.get("opening_defense", False))
                        )
                        if defense_count >= int(cfg.defense_max_positions):
                            continue
                        wait, _wait_reason = d2_replay._is_chase_wait(cfg, row, vg)
                        if wait:
                            continue
                        target_amount = open_defense_amount
                        action = "OPEN_DEFENSE_BUY"
                        opening_defense = True
                    ordinary_position = False
                    ordinary_risky = False
                else:
                    # PG2 lock/re-arm affects only re-entry after PROFIT_GUARD2.
                    block_until = pg2_block_until.get(symbol)
                    if block_until is not None:
                        if now < block_until:
                            diag["pg2_lock_blocks"] += 1
                            continue
                        rearmed, _rearm_reason = _pg2_rearmed(row, vg)
                        if not rearmed:
                            diag["pg2_rearm_blocks"] += 1
                            continue
                        # Re-armed winner enters with original full 50%.
                        target_amount = normal_amount
                        action = "PG2_REARM_BUY1"
                        ordinary_risky = False
                    else:
                        risky, risky_reason = _ordinary_risky(row, vg)
                        if risky:
                            target_amount = risky_amount
                            action = "HYBRID_RISK_BUY1"
                            ordinary_risky = True
                            diag["risky_25pct_entries"] += 1
                            if risky_reason.startswith("RISK_SIZE_VWAP"):
                                diag["risky_vwap_entries"] += 1
                            else:
                                diag["risky_overheat_entries"] += 1
                        else:
                            target_amount = normal_amount
                            action = "BUY1"
                            ordinary_risky = False
                            diag["ordinary_full_entries"] += 1
                    opening_defense = False
                    ordinary_position = True

                fill = replay_kr._fill_price(fill_cfg, "BUY", ref_price)
                qty1 = int(target_amount // fill)
                if qty1 <= 0:
                    continue
                cost = fill * qty1
                if daily_buy_amount + cost > cfg.daily_budget_krw:
                    continue
                prefix = {
                    "OPEN_STRONG_BUY": "OPEN STRONG 50%",
                    "OPEN_DEFENSE_BUY": "OPEN DEFENSE 25%",
                    "HYBRID_RISK_BUY1": "HYBRID RISK 25%",
                    "PG2_REARM_BUY1": "PG2 REARM 50%",
                }.get(action, "D-v2 NORMAL 50%")
                add_event(
                    now, symbol, action, "BUY", qty1, ref_price, fill,
                    f"{prefix} · 점수 {m['score']:.1f} · TOP{m['rank']} · "
                    f"3/5분 {m['ret3']:+.2f}/{m['ret5']:+.2f}% · "
                    f"거래량 {m['vol']:.2f}배 · VWAP {vg:+.2f}%",
                    "", 0.0, m["score"], f"TOP{m['rank']}", vg,
                )
                positions[symbol] = {
                    "qty": qty1,
                    "avg_price": fill,
                    "created_at": now.isoformat(),
                    "take1_sent": False,
                    "peak_pnl": 0.0,
                    "opening_defense": opening_defense,
                    # d2 helper expects this exact key for OPEN_CONFIRM.
                    "defense_position": opening_defense,
                    "open_confirmed": not opening_defense,
                    "ordinary_position": ordinary_position,
                    "ordinary_risky": ordinary_risky,
                    "risk_confirmed": not ordinary_risky,
                }
                daily_buy_amount += cost
                if action == "PG2_REARM_BUY1":
                    pg2_block_until.pop(symbol, None)
                    diag["pg2_rearm_successes"] += 1
                if action == "OPEN_STRONG_BUY":
                    diag["strong_open_entries"] += 1
                elif action == "OPEN_DEFENSE_BUY":
                    diag["open_defense_entries"] += 1

        now += pd.Timedelta(seconds=int(cfg.manage_seconds))

    # End safety exit.
    if positions:
        for symbol, pos in list(positions.items()):
            frame = target_frames.get(symbol)
            ref_price = replay_kr._price_at(frame, date_text, end) if frame is not None else 0.0
            if ref_price <= 0:
                continue
            qty = int(pos.get("qty", 0) or 0)
            avg = _safe_float(pos.get("avg_price", 0))
            if qty <= 0 or avg <= 0:
                continue
            fill = replay_kr._fill_price(fill_cfg, "SELL", ref_price)
            pnl = (ref_price / avg - 1.0) * 100.0
            realized = (fill - avg) * qty
            vg = _vwap_gap_pct(frame, date_text, end, ref_price)
            add_event(end, symbol, "FORCE_SELL_END", "SELL", qty, ref_price, fill,
                      "리플레이 종료 안전청산", pnl, realized, vwap_gap=vg)
            positions.pop(symbol, None)

    events_df = pd.DataFrame(events)
    buy_amount = float(events_df.loc[events_df["구분"] == "BUY", "주문금액KRW"].sum()) if not events_df.empty else 0.0
    sell_amount = float(events_df.loc[events_df["구분"] == "SELL", "주문금액KRW"].sum()) if not events_df.empty else 0.0
    realized = float(events_df["실현손익KRW"].sum()) if not events_df.empty else 0.0
    summary = {
        "총주문횟수": int(len(events_df)),
        "매수주문횟수": int((events_df["구분"] == "BUY").sum()) if not events_df.empty else 0,
        "매도주문횟수": int((events_df["구분"] == "SELL").sum()) if not events_df.empty else 0,
        "거래종목수": int(events_df["종목코드"].nunique()) if not events_df.empty else 0,
        "누적매수금액KRW": int(round(buy_amount)),
        "누적매도금액KRW": int(round(sell_amount)),
        "실현손익KRW": int(round(realized)),
        "일일예산1000만원대비수익률": round(realized / cfg.daily_budget_krw * 100.0, 3) if cfg.daily_budget_krw else 0.0,
    }
    return {
        "ok": True,
        "version": D3_HYBRID_VERSION,
        "date": date_text,
        "strategy": "D3_V2_HYBRID",
        "summary": summary,
        "diagnostic": diag,
        "events": events,
        "rules": asdict(RULES),
    }


def _pack_day_engine(payload: dict, *, control: bool = False) -> dict:
    if control:
        sm = payload.get("d_summary", {}) or {}
        events = list(payload.get("d_events", []) or [])
        diag = payload.get("diagnostic", {}) or {}
    else:
        sm = payload.get("summary", {}) or {}
        events = list(payload.get("events", []) or [])
        diag = payload.get("diagnostic", {}) or {}
    sell_pnls = [
        int(e.get("실현손익KRW", 0) or 0)
        for e in events
        if isinstance(e, dict) and str(e.get("구분", "")).upper() == "SELL"
    ]
    gross_profit = int(sum(x for x in sell_pnls if x > 0))
    gross_loss = int(sum(x for x in sell_pnls if x < 0))
    positive_sell = sum(1 for x in sell_pnls if x > 0)
    negative_sell = sum(1 for x in sell_pnls if x < 0)
    return {
        "pnl_KRW": int(sm.get("실현손익KRW", 0) or 0),
        "buy_orders": int(sm.get("매수주문횟수", 0) or 0),
        "sell_orders": int(sm.get("매도주문횟수", 0) or 0),
        "total_orders": int(sm.get("총주문횟수", 0) or 0),
        "gross_profit_KRW": gross_profit,
        "gross_loss_KRW": gross_loss,
        "positive_sell_events": int(positive_sell),
        "negative_sell_events": int(negative_sell),
        "diagnostic": diag,
    }


def _aggregate_variant(label: str, key: str, rows: list[dict]) -> dict:
    vals = [int((r.get(key) or {}).get("pnl_KRW", 0) or 0) for r in rows]
    total = int(sum(vals))
    gp = int(sum(int((r.get(key) or {}).get("gross_profit_KRW", 0) or 0) for r in rows))
    gl = int(sum(int((r.get(key) or {}).get("gross_loss_KRW", 0) or 0) for r in rows))
    pos_sell = int(sum(int((r.get(key) or {}).get("positive_sell_events", 0) or 0) for r in rows))
    neg_sell = int(sum(int((r.get(key) or {}).get("negative_sell_events", 0) or 0) for r in rows))
    buys = int(sum(int((r.get(key) or {}).get("buy_orders", 0) or 0) for r in rows))
    orders = int(sum(int((r.get(key) or {}).get("total_orders", 0) or 0) for r in rows))
    diagnostics: dict[str, int] = {}
    for r in rows:
        for k, v in ((r.get(key) or {}).get("diagnostic", {}) or {}).items():
            try:
                diagnostics[k] = diagnostics.get(k, 0) + int(v or 0)
            except Exception:
                pass
    return {
        "id": key,
        "label": label,
        "total_KRW": total,
        "average_daily_KRW": round(total / len(vals), 1) if vals else 0.0,
        "positive_days": int(sum(1 for v in vals if v > 0)),
        "negative_days": int(sum(1 for v in vals if v < 0)),
        "max_cumulative_drawdown_KRW": _max_drawdown(vals),
        "buy_orders": buys,
        "total_orders": orders,
        "gross_profit_KRW": gp,
        "gross_loss_KRW": gl,
        "profit_factor": round(gp / abs(gl), 4) if gl < 0 else 0.0,
        "positive_sell_events": pos_sell,
        "negative_sell_events": neg_sell,
        "sell_event_win_rate_pct": round(100.0 * pos_sell / (pos_sell + neg_sell), 2) if (pos_sell + neg_sell) else 0.0,
        "diagnostic_totals": diagnostics,
    }


def _profit_preservation(rows: list[dict], variant_key: str) -> dict:
    control_positive_profit = 0
    sacrificed = 0
    extra = 0
    improved_negative = 0
    positive_days_hurt = 0
    loss_reduction = 0
    loss_worsening = 0
    for r in rows:
        c = int((r.get("CONTROL_D2") or {}).get("pnl_KRW", 0) or 0)
        v = int((r.get(variant_key) or {}).get("pnl_KRW", 0) or 0)
        if c > 0:
            control_positive_profit += c
            if v < c:
                # Match the existing SURGICAL acceptance metric exactly: if a
                # control-positive day turns negative, the negative overshoot is
                # counted too (c - v), not clipped at zero.
                sacrificed += c - v
                positive_days_hurt += 1
            elif v > c:
                extra += v - c
        elif c < 0:
            if v > c:
                improved_negative += 1
                loss_reduction += v - c
            elif v < c:
                loss_worsening += c - v
    control_gp = int(sum(int((r.get("CONTROL_D2") or {}).get("gross_profit_KRW", 0) or 0) for r in rows))
    variant_gp = int(sum(int((r.get(variant_key) or {}).get("gross_profit_KRW", 0) or 0) for r in rows))
    preservation = (
        100.0 * max(0, control_positive_profit - sacrificed) / control_positive_profit
        if control_positive_profit > 0 else 100.0
    )
    gp_preservation = 100.0 * variant_gp / control_gp if control_gp > 0 else 100.0
    return {
        "control_positive_day_profit_KRW": control_positive_profit,
        "profit_sacrificed_on_control_positive_days_KRW": int(sacrificed),
        "extra_profit_on_control_positive_days_KRW": int(extra),
        "profit_preservation_pct": round(preservation, 2),
        "control_gross_profit_KRW": control_gp,
        "variant_gross_profit_KRW": variant_gp,
        "gross_profit_preservation_pct": round(gp_preservation, 2),
        "control_positive_days_hurt": int(positive_days_hurt),
        "control_negative_days_improved": int(improved_negative),
        "loss_reduction_on_control_negative_days_KRW": int(loss_reduction),
        "loss_worsening_on_control_negative_days_KRW": int(loss_worsening),
        "passes_90pct_profit_preservation": preservation >= RULES.profit_preservation_min_pct,
        "passes_90pct_gross_profit_preservation": gp_preservation >= RULES.gross_profit_preservation_min_pct,
    }


def _monthly(rows: list[dict], key: str) -> list[dict]:
    monthly: dict[str, dict] = {}
    for r in rows:
        date_text = str(r.get("date", ""))
        month = date_text[:7]
        m = monthly.setdefault(month, {"month": month, "days": 0, "control_KRW": 0, "variant_KRW": 0})
        m["days"] += 1
        m["control_KRW"] += int((r.get("CONTROL_D2") or {}).get("pnl_KRW", 0) or 0)
        m["variant_KRW"] += int((r.get(key) or {}).get("pnl_KRW", 0) or 0)
    out = []
    for month in sorted(monthly):
        m = monthly[month]
        m["delta_KRW"] = int(m["variant_KRW"] - m["control_KRW"])
        out.append(m)
    return out


def _profitable_month_preservation(monthly_rows: list[dict]) -> dict:
    months = []
    for r in monthly_rows:
        c = int(r.get("control_KRW", 0) or 0)
        if c <= 0:
            continue
        v = int(r.get("variant_KRW", 0) or 0)
        pct = max(0.0, 100.0 * v / c)
        months.append({
            "month": r.get("month"),
            "control_KRW": c,
            "variant_KRW": v,
            "preservation_pct": round(pct, 2),
        })
    min_pct = min((float(x["preservation_pct"]) for x in months), default=100.0)
    return {
        "months": months,
        "min_preservation_pct": round(min_pct, 2),
        "passes_85pct_each_profitable_month": min_pct >= RULES.profitable_month_floor_pct,
    }


def _day_delta(row: dict, key: str) -> dict:
    c = int((row.get("CONTROL_D2") or {}).get("pnl_KRW", 0) or 0)
    v = int((row.get(key) or {}).get("pnl_KRW", 0) or 0)
    return {"date": row.get("date"), "control_KRW": c, "variant_KRW": v, "delta_KRW": v - c}


def _run_job(
    result: dict,
    provider: Callable,
    codes: list[str],
    frozen_config: d2_replay.OpenDefenseConfig,
    protected_window_fn: Callable[[], tuple[bool, str]] | None,
) -> None:
    try:
        daily_base = [r for r in list(result.get("daily", []) or []) if isinstance(r, dict) and r.get("date")]
        daily_base.sort(key=lambda r: str(r.get("date")))
        if not daily_base:
            raise RuntimeError("frozen D-v2 daily rows missing")

        # Both control and hybrid use the same cached KIS provider.
        replay_kr._download_intraday = provider
        d2_replay._download_intraday = provider

        _state(
            status="running",
            phase="D3_V2_HYBRID_FULL_ENGINE",
            started_at=datetime.now(KST).isoformat(timespec="seconds"),
            total_days=len(daily_base),
            completed_days=0,
            progress_pct=0.0,
            result_ready=False,
            message="D-v2 CONTROL vs D3-v2 HYBRID 전체엔진 비교 시작",
            last_error="",
        )

        rows: list[dict] = []
        errors: list[dict] = []
        mismatches: list[dict] = []

        for idx, base_row in enumerate(daily_base, start=1):
            date_text = str(base_row.get("date"))
            if protected_window_fn is not None:
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
                        message="실시간 자동매매 보호를 위해 D3-v2 HYBRID 검증 일시정지",
                    )
                    time.sleep(30.0)

            day_path = DAY_DIR / f"{date_text}.json"
            cached = _read_json(day_path, None)
            if isinstance(cached, dict) and cached.get("version") == D3_HYBRID_VERSION:
                row = cached
            else:
                try:
                    control_payload = d2_replay.run_kr_open_defense_replay(
                        date_text=date_text,
                        codes=codes,
                        config=frozen_config,
                        refresh=True,
                    )
                    hybrid_payload = _run_hybrid_day(date_text, codes, frozen_config)
                    row = {
                        "version": D3_HYBRID_VERSION,
                        "date": date_text,
                        "cached_D2_KRW": int(base_row.get("D2_KRW", 0) or 0),
                        "CONTROL_D2": _pack_day_engine(control_payload, control=True),
                        "D3_V2_HYBRID": _pack_day_engine(hybrid_payload, control=False),
                    }
                    row["parity_delta_KRW"] = int(row["CONTROL_D2"]["pnl_KRW"] - row["cached_D2_KRW"])
                    row["D3_V2_HYBRID"]["delta_vs_control_KRW"] = int(
                        row["D3_V2_HYBRID"]["pnl_KRW"] - row["CONTROL_D2"]["pnl_KRW"]
                    )
                    _atomic_write_json(day_path, row)
                except Exception as exc:
                    errors.append({"date": date_text, "error": f"{type(exc).__name__}: {exc}"})
                    _state(last_error=errors[-1]["error"][:1000])
                    continue

            rows.append(row)
            parity_delta = int(row.get("parity_delta_KRW", 0) or 0)
            if parity_delta != 0:
                mismatches.append({
                    "date": date_text,
                    "cached_D2_KRW": int(row.get("cached_D2_KRW", 0) or 0),
                    "control_D2_KRW": int((row.get("CONTROL_D2") or {}).get("pnl_KRW", 0) or 0),
                    "delta_KRW": parity_delta,
                })
            _state(
                status="running",
                phase="D3_V2_HYBRID_FULL_ENGINE",
                current_date=date_text,
                completed_days=len(rows),
                error_days=len(errors),
                total_days=len(daily_base),
                progress_pct=round(100.0 * idx / len(daily_base), 1),
                message=f"D3-v2 HYBRID {idx}/{len(daily_base)} 완료",
            )

        control = _aggregate_variant("D-v2 원본 전체엔진", "CONTROL_D2", rows)
        hybrid = _aggregate_variant("D3-v2 HYBRID · 선택형 25% + 수익보존 방어", "D3_V2_HYBRID", rows)
        hybrid["delta_vs_control_KRW"] = int(hybrid["total_KRW"] - control["total_KRW"])
        parity_ok = (
            not mismatches
            and not errors
            and len(rows) == len(daily_base)
            and int(control["total_KRW"]) == int(((result.get("overall") or {}).get("D2_total_KRW", 0)) or 0)
        )

        preservation = _profit_preservation(rows, "D3_V2_HYBRID")
        monthly = _monthly(rows, "D3_V2_HYBRID")
        month_pres = _profitable_month_preservation(monthly)
        accepted = bool(
            parity_ok
            and hybrid["total_KRW"] > control["total_KRW"]
            and preservation["profit_preservation_pct"] >= RULES.profit_preservation_min_pct
            and preservation["gross_profit_preservation_pct"] >= RULES.gross_profit_preservation_min_pct
            and month_pres["min_preservation_pct"] >= RULES.profitable_month_floor_pct
        )

        diffs = [_day_delta(r, "D3_V2_HYBRID") for r in rows]
        payload = {
            "ok": True,
            "version": D3_HYBRID_VERSION,
            "mode": "PATH_CONSISTENT_D3_V2_HYBRID",
            "read_only": True,
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
                "total_delta_KRW": int(control["total_KRW"]) - int(((result.get("overall") or {}).get("D2_total_KRW", 0)) or 0),
                "mismatch_days": mismatches,
                "rule": "CONTROL must match frozen cached D-v2 day-by-day before D3-v2 HYBRID is accepted.",
            },
            "variants": [control, hybrid],
            "profit_preservation": {"D3_V2_HYBRID": preservation},
            "profitable_month_preservation": {"D3_V2_HYBRID": month_pres},
            "monthly": {"D3_V2_HYBRID": monthly},
            "acceptance": {
                "profit_preservation_min_pct": RULES.profit_preservation_min_pct,
                "gross_profit_preservation_min_pct": RULES.gross_profit_preservation_min_pct,
                "each_original_profitable_month_min_pct": RULES.profitable_month_floor_pct,
                "must_improve_total_KRW": True,
                "parity_required": True,
            },
            "accepted": accepted,
            "best_candidate_if_acceptance_passes": hybrid if accepted else None,
            "top_hybrid_improved_days": sorted(diffs, key=lambda x: x["delta_KRW"], reverse=True)[:10],
            "top_hybrid_worsened_days": sorted(diffs, key=lambda x: x["delta_KRW"])[:10],
            "rules": {
                "opening": "frozen D-v2 unchanged",
                "ordinary_entry": "50% normally; 25% only on SURGICAL95-style VWAP weakness or weak overheat; recovered risk entry may confirm back to 50%",
                "structural_exit": "ordinary post-open only: >=5m, pnl<=-2.4%, >=3 strict failures",
                "winner_immunity": "peak>=+0.8% requires pnl<=-2.7% and all 4 failures",
                "pg2_reentry": "5m hard lock then TOP3 + VWAP>=0 + ret3>=0 + ret5>0 + valid buy signal; full 50%",
                "take2_reentry": "unchanged",
                "profit_engine": "TAKE1 +3%, TAKE2 +5%, original PROFIT_GUARD unchanged",
            },
            "daily": rows,
            "important_limit": (
                "Historical candidate selection uses the same fixed liquidity universe as frozen D-v2, not exact historical whole-market KIS TOP5. "
                "Fees/taxes remain excluded. This is strategy-development validation only and is not connected to live/paper orders."
            ),
        }
        _atomic_write_json(RESULT_FILE, payload)
        _state(
            status="completed",
            phase="DONE",
            completed_days=len(rows),
            error_days=len(errors),
            total_days=len(daily_base),
            progress_pct=100.0,
            result_ready=True,
            parity_ok=parity_ok,
            accepted=accepted,
            finished_at=datetime.now(KST).isoformat(timespec="seconds"),
            message="D3-v2 HYBRID 전체엔진 검증 완료",
            last_error="",
        )
    except Exception as exc:
        _state(
            status="error",
            phase="ERROR",
            result_ready=RESULT_FILE.exists(),
            last_error=f"{type(exc).__name__}: {exc}"[:1200],
            message="D3-v2 HYBRID 전체엔진 검증 오류",
        )


def ensure_d3_hybrid_started(
    result: dict,
    *,
    provider: Callable,
    codes: Iterable[str],
    frozen_config: d2_replay.OpenDefenseConfig,
    protected_window_fn: Callable[[], tuple[bool, str]] | None = None,
) -> dict:
    """Return cached result or start a read-only background full-engine validation."""
    global _THREAD
    existing = _read_json(RESULT_FILE, {}) or {}
    base_total = int(((result.get("overall") or {}).get("D2_total_KRW", 0)) or 0)
    if (
        existing.get("ok") is True
        and existing.get("version") == D3_HYBRID_VERSION
        and int(((existing.get("parity") or {}).get("cached_D2_expected_total_KRW", base_total)) or base_total) == base_total
    ):
        compact = dict(existing)
        compact.pop("daily", None)
        return compact

    with _LOCK:
        if _THREAD and _THREAD.is_alive():
            return _public_state()
        _THREAD = threading.Thread(
            target=_run_job,
            args=(
                dict(result),
                provider,
                [str(x).zfill(6) for x in codes],
                frozen_config,
                protected_window_fn,
            ),
            daemon=True,
            name="kr-d3-v2-hybrid-full-engine",
        )
        _THREAD.start()
        state = _public_state()
        state["started"] = True
        return state

from __future__ import annotations

"""Read-only full-engine Runner replay for KR D-v2.

This module keeps the D-v2 entry/defense/profit-guard engine intact and changes only
what happens after TAKE2 (+5%) when runner_trailing_pct is provided.

It never sends real orders. The long-backtest module monkey-patches replay_kr's
intraday provider to the already-cached Railway KIS 1-minute bars before calling it.
"""

import math
from dataclasses import asdict
from typing import Iterable

import pandas as pd

import replay_kr
import replay_kr_open_defense_v2 as d2

KST = d2.KST
RUNNER_FULL_VERSION = "kr-d2-runner-full-engine-v1"


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


def run_kr_open_defense_runner_replay(
    date_text: str,
    codes: Iterable[str] | None = None,
    config: d2.OpenDefenseConfig | None = None,
    runner_trailing_pct: float | None = None,
    runner_fraction: float = 0.30,
) -> dict:
    """Run the whole D-v2 engine for one day with an optional TAKE2 runner.

    runner_trailing_pct=None reproduces the original D-v2 TAKE2 full exit and is used
    as a parity control. With a positive percentage, TAKE2 retains floor(original
    episode bought quantity * runner_fraction), capped by the remaining quantity.
    The retained runner occupies a real position slot, blocks same-symbol re-entry,
    counts its later sell as a real order, and exits on closed-bar Close trailing or
    the normal 15:15 force exit.
    """
    cfg = config or d2.OpenDefenseConfig()
    trail_pct = None if runner_trailing_pct is None else float(runner_trailing_pct)
    if trail_pct is not None and trail_pct <= 0:
        raise ValueError("runner_trailing_pct must be positive or None")
    runner_fraction = max(0.0, min(1.0, float(runner_fraction)))
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

    normal_amount = d2._normal_entry_amount(cfg)
    defense_amount = d2._defense_entry_amount(cfg)
    confirm_amount = max(0, normal_amount - defense_amount)

    positions: dict[str, dict] = {}
    events: list[dict] = []
    latest_top5 = pd.DataFrame()
    last_scan = None
    daily_buy_amount = 0.0
    daily_orders = 0
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
        "runner_take2_activations": 0,
        "runner_trailing_exits": 0,
        "runner_force_exits": 0,
        "runner_no_whole_share": 0,
        "runner_same_symbol_reentry_blocks": 0,
        "runner_position_cap_block_ticks": 0,
    }
    chase_wait_seen: set[tuple[str, str]] = set()
    runner_same_symbol_seen: set[tuple[str, str]] = set()

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
            latest_top5 = replay_kr._build_top5_at(target_frames, meta, date_text, now, cfg.scan_count)
            last_scan = now

        top5_map = {}
        if latest_top5 is not None and not latest_top5.empty:
            for _, r in latest_top5.iterrows():
                top5_map[str(r.get("종목코드", "")).zfill(6)] = r

        # 1) Existing-position management.
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
            vg = d2._vwap_gap_pct(frame, date_text, now, ref_price)

            # Runner is managed independently after TAKE2. It still occupies a real slot.
            if bool(pos.get("runner_active", False)):
                peak_close = max(d2._safe_float(pos.get("runner_peak_close", ref_price)), ref_price)
                pos["runner_peak_close"] = peak_close
                if replay_kr._seconds_of_day(now) >= force_exit_sec:
                    fill = replay_kr._fill_price(c_cfg, "SELL", ref_price)
                    realized = (fill - avg) * qty
                    add_event(
                        now, symbol, "RUNNER_FORCE_SELL", "SELL", qty,
                        ref_price, fill,
                        f"Runner 당일 강제청산 {cfg.force_exit_time} KST",
                        pnl, realized, vwap_gap=vg,
                    )
                    diagnostic["runner_force_exits"] += 1
                    positions.pop(symbol, None)
                    continue
                trigger = peak_close * (1.0 - float(trail_pct or 0.0) / 100.0)
                if trail_pct is not None and ref_price <= trigger:
                    fill = replay_kr._fill_price(c_cfg, "SELL", ref_price)
                    realized = (fill - avg) * qty
                    drawdown = (ref_price / peak_close - 1.0) * 100.0 if peak_close > 0 else 0.0
                    add_event(
                        now, symbol, f"RUNNER_TRAIL_{trail_pct:.1f}", "SELL", qty,
                        ref_price, fill,
                        f"Runner 최고종가 {peak_close:.2f} 대비 {drawdown:.2f}% 되밀림",
                        pnl, realized, vwap_gap=vg,
                    )
                    diagnostic["runner_trailing_exits"] += 1
                    positions.pop(symbol, None)
                    continue
                # No D-v2 stop/profit-guard is applied to the protected runner tail.
                continue

            peak = max(d2._safe_float(pos.get("peak_pnl", pnl)), pnl)
            pos["peak_pnl"] = peak
            dd = max(0.0, peak - pnl)

            if replay_kr._seconds_of_day(now) >= force_exit_sec:
                fill = replay_kr._fill_price(c_cfg, "SELL", ref_price)
                realized = (fill - avg) * qty
                add_event(now, symbol, "FORCE_SELL", "SELL", qty, ref_price, fill,
                          f"당일 강제청산 {cfg.force_exit_time} KST", pnl, realized,
                          vwap_gap=vg)
                positions.pop(symbol, None)
                continue

            if bool(pos.get("defense_position", False)):
                try:
                    created = pd.Timestamp(pos.get("created_at"))
                    if created.tzinfo is None:
                        created = created.tz_localize(KST)
                    hold_min = max(0.0, (now - created).total_seconds() / 60.0)
                except Exception:
                    hold_min = 0.0

                row = top5_map.get(symbol)
                ret3 = d2._safe_float(row.get("최근3분수익률", 0)) if row is not None else -999.0
                ret5 = d2._safe_float(row.get("최근5분수익률", 0)) if row is not None else -999.0
                score_now = d2._safe_float(row.get("종합점수", 0)) if row is not None else 0.0
                weak_now = bool(row.get("모멘텀약화", False)) if row is not None else True
                signal_now = "매수 후보" in str(row.get("판정", "")) if row is not None else False

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

                    if hold_min >= float(cfg.soft_fail_min_hold_minutes) and pnl <= float(cfg.defense_soft_fail_pct):
                        fail_signals = []
                        if vg <= float(cfg.soft_fail_vwap_gap_pct):
                            fail_signals.append("BELOW_VWAP")
                        if ret3 <= float(cfg.soft_fail_ret3_pct):
                            fail_signals.append("RET3_WEAK")
                        if ret5 <= float(cfg.soft_fail_ret5_pct):
                            fail_signals.append("RET5_WEAK")
                        if score_now < float(cfg.soft_fail_score) or weak_now or not signal_now:
                            fail_signals.append("LEADER_STRENGTH_LOST")
                        if len(fail_signals) >= int(cfg.soft_fail_min_signals):
                            fill = replay_kr._fill_price(c_cfg, "SELL", ref_price)
                            realized = (fill - avg) * qty
                            add_event(
                                now, symbol, "OPEN_SOFT_FAIL_EXIT", "SELL", qty,
                                ref_price, fill,
                                f"복합 돌파실패 · pnl {pnl:.2f}% · " + ",".join(fail_signals),
                                pnl, realized, vwap_gap=vg,
                            )
                            diagnostic["open_soft_fail_exits"] += 1
                            positions.pop(symbol, None)
                            continue

            if pnl <= -abs(cfg.stop_loss_pct):
                fill = replay_kr._fill_price(c_cfg, "SELL", ref_price)
                realized = (fill - avg) * qty
                add_event(now, symbol, "STOP_LOSS", "SELL", qty, ref_price, fill,
                          f"손절 {pnl:.2f}%", pnl, realized, vwap_gap=vg)
                positions.pop(symbol, None)
                continue

            if (
                bool(pos.get("defense_position", False))
                and not bool(pos.get("open_confirmed", False))
                and replay_kr._seconds_of_day(now) < defense_end_sec
                and daily_orders < cfg.max_daily_orders
            ):
                row = top5_map.get(symbol)
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
                        score = d2._safe_float(row.get("종합점수", 0)) if row is not None else ""
                        rank = str(row.get("순위", "")) if row is not None else ""
                        add_event(
                            now, symbol, "OPEN_CONFIRM", "BUY", qty2,
                            ref_price, fill, why, pnl, 0.0, score, rank, vg,
                        )
                        pos["qty"] = new_qty
                        pos["avg_price"] = new_avg
                        pos["episode_bought_qty"] = int(pos.get("episode_bought_qty", old_qty)) + qty2
                        pos["open_confirmed"] = True
                        daily_buy_amount += cost
                        diagnostic["open_confirms"] += 1
                        continue

            if pnl >= cfg.take1_pct and not bool(pos.get("take1_sent")):
                sell_qty = max(1, qty // 2)
                fill = replay_kr._fill_price(c_cfg, "SELL", ref_price)
                realized = (fill - avg) * sell_qty
                add_event(now, symbol, "TAKE1", "SELL", sell_qty, ref_price, fill,
                          f"1차 익절 {pnl:.2f}% · 약 50%", pnl, realized, vwap_gap=vg)
                pos["qty"] = qty - sell_qty
                pos["take1_sent"] = True
                if pos["qty"] <= 0:
                    positions.pop(symbol, None)
                continue

            if pnl >= cfg.take2_pct and bool(pos.get("take1_sent")):
                if trail_pct is None:
                    fill = replay_kr._fill_price(c_cfg, "SELL", ref_price)
                    realized = (fill - avg) * qty
                    add_event(now, symbol, "TAKE2", "SELL", qty, ref_price, fill,
                              f"2차 익절 {pnl:.2f}% · 전량", pnl, realized, vwap_gap=vg)
                    positions.pop(symbol, None)
                    continue

                original_qty = max(qty, int(pos.get("episode_bought_qty", qty) or qty))
                runner_qty = min(qty, int(math.floor(original_qty * runner_fraction)))
                if runner_qty <= 0:
                    diagnostic["runner_no_whole_share"] += 1
                    fill = replay_kr._fill_price(c_cfg, "SELL", ref_price)
                    realized = (fill - avg) * qty
                    add_event(now, symbol, "TAKE2", "SELL", qty, ref_price, fill,
                              f"2차 익절 {pnl:.2f}% · Runner 1주 미만이라 전량", pnl, realized, vwap_gap=vg)
                    positions.pop(symbol, None)
                    continue

                sell_qty = qty - runner_qty
                if sell_qty > 0:
                    fill = replay_kr._fill_price(c_cfg, "SELL", ref_price)
                    realized = (fill - avg) * sell_qty
                    add_event(
                        now, symbol, "TAKE2_RUNNER", "SELL", sell_qty,
                        ref_price, fill,
                        f"2차 익절 {pnl:.2f}% · 원래 포지션의 {runner_fraction*100:.0f}% Runner {runner_qty}주 유지",
                        pnl, realized, vwap_gap=vg,
                    )
                pos["qty"] = runner_qty
                pos["runner_active"] = True
                pos["runner_trailing_pct"] = trail_pct
                pos["runner_peak_close"] = ref_price
                pos["runner_started_at"] = now.isoformat()
                diagnostic["runner_take2_activations"] += 1
                continue

            suppress_profit_guard = False
            if bool(pos.get("defense_position", False)) and not bool(pos.get("open_confirmed", False)):
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
                    positions.pop(symbol, None)
                    continue

        # 2) New entries. A runner is a real position and can consume a slot.
        if len(positions) >= cfg.max_positions and any(bool(p.get("runner_active")) for p in positions.values()):
            diagnostic["runner_position_cap_block_ticks"] += 1

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
                signal = str(row.get("판정", ""))
                score = d2._safe_float(row.get("종합점수", 0))
                weak = bool(row.get("모멘텀약화", False))
                eligible_signal = "매수 후보" in signal and not weak and score >= cfg.min_score
                if symbol in positions:
                    if eligible_signal and bool(positions[symbol].get("runner_active", False)):
                        key = (symbol, now.isoformat())
                        if key not in runner_same_symbol_seen:
                            runner_same_symbol_seen.add(key)
                            diagnostic["runner_same_symbol_reentry_blocks"] += 1
                    continue
                if not eligible_signal:
                    continue

                frame = target_frames.get(symbol)
                if frame is None:
                    continue
                ref_price = replay_kr._price_at(frame, date_text, now)
                if ref_price <= 0:
                    continue
                vg = d2._vwap_gap_pct(frame, date_text, now, ref_price)

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
                    target_amount = normal_amount
                    action = "BUY1"
                    defense_position = False

                fill = replay_kr._fill_price(c_cfg, "BUY", ref_price)
                qty1 = int(target_amount // fill)
                if qty1 <= 0:
                    continue
                cost = fill * qty1
                if daily_buy_amount + cost > cfg.daily_budget_krw:
                    continue

                rank = str(row.get("순위", ""))
                r3 = d2._safe_float(row.get("최근3분수익률", 0))
                r5 = d2._safe_float(row.get("최근5분수익률", 0))
                vr = d2._safe_float(row.get("거래량배수", 0))
                if action == "OPEN_STRONG_BUY":
                    prefix = "OPEN STRONG 50%"
                elif action == "OPEN_DEFENSE_BUY":
                    prefix = "OPEN DEFENSE 25%"
                else:
                    prefix = "C기준 50%"
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
                    "episode_bought_qty": qty1,
                    "runner_active": False,
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
            ref_price = replay_kr._price_at(frame, date_text, end) if frame is not None else 0.0
            if ref_price <= 0:
                continue
            qty = int(pos.get("qty", 0))
            avg = d2._safe_float(pos.get("avg_price", 0))
            fill = replay_kr._fill_price(c_cfg, "SELL", ref_price)
            pnl = (ref_price / avg - 1.0) * 100.0 if avg > 0 else 0.0
            realized = (fill - avg) * qty
            vg = d2._vwap_gap_pct(frame, date_text, end, ref_price)
            action = "RUNNER_FORCE_SELL_END" if bool(pos.get("runner_active")) else "FORCE_SELL_END"
            add_event(end, symbol, action, "SELL", qty, ref_price, fill,
                      "리플레이 종료 안전청산", pnl, realized, vwap_gap=vg)
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
        "일일예산1000만원대비수익률": round((realized / cfg.daily_budget_krw * 100.0), 3) if cfg.daily_budget_krw > 0 else 0.0,
    }

    return {
        "ok": True,
        "version": RUNNER_FULL_VERSION,
        "date": date_text,
        "strategy": "D2_FULL_ENGINE" if trail_pct is None else f"D2_FULL_ENGINE_RUNNER_{trail_pct:.1f}",
        "runner": {
            "enabled": trail_pct is not None,
            "trailing_pct": trail_pct,
            "fraction_of_original_position": runner_fraction,
            "whole_share_rule": "floor(original episode bought quantity * fraction), capped by remaining TAKE2 qty",
            "position_slot_is_real": True,
            "same_symbol_reentry_blocked_while_held": True,
            "extra_runner_exit_counts_as_order": True,
            "force_exit_time": cfg.force_exit_time,
            "price_rule": "same D-v2 management clock / latest completed 1-minute Close via _price_at",
        },
        "diagnostic": diagnostic,
        "summary": summary,
        "events": events,
        "config": asdict(cfg),
        "assumptions": {
            "real_orders": False,
            "future_data_visible": False,
            "fees_taxes": "별도 미포함",
            "purpose": "Runner full-engine path-consistent validation",
        },
    }

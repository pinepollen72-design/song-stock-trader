from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from ai_committee import (
    CFG as AI_CFG,
    _call_openai,
    _candidate_snapshot,
    _chair,
    _market_context,
    _safe_float,
    OUTPUT_SCHEMA,
    _extract_output_text,
)
from replay_kr import (
    KRReplayConfig,
    _build_top5_at,
    _clock_seconds,
    _download_intraday,
    _fill_price,
    _normalize_universe,
    _price_at,
    _seconds_of_day,
    _extract_yf_frame,
    _bars_until,
    _previous_close,
    run_kr_trade_replay,
)

KST = ZoneInfo("Asia/Seoul")


def _resolve_state_dir() -> Path:
    explicit = os.getenv("SONG_TRADER_STATE_DIR", "").strip()
    if explicit:
        return Path(explicit)
    railway_volume = os.getenv("RAILWAY_VOLUME_MOUNT_PATH", "").strip()
    if railway_volume:
        return Path(railway_volume) / "song_trader_v2"
    return Path("/tmp/song_trader_v2")


STATE_DIR = _resolve_state_dir()
REPLAY_DIR = STATE_DIR / "replays" / "ai_committee" / "kr"
REPLAY_DIR.mkdir(parents=True, exist_ok=True)

AI_REPLAY_VERSION = "kr-ai-committee-replay-v1"
DEFAULT_DATES = [
    "2026-08-10",
    "2026-08-11",
    "2026-08-12",
    "2026-08-13",
    "2026-08-14",
]


def _cache_path(date_text: str) -> Path:
    return REPLAY_DIR / f"kr_ai_committee_replay_{date_text}.json"


def _decision_cache_path(date_text: str) -> Path:
    return REPLAY_DIR / f"kr_ai_decisions_{date_text}.json"


def _load_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


def _atomic_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    tmp.replace(path)


def _rank_number(row: dict | pd.Series) -> int:
    raw = str(row.get("순위", "") or "")
    digits = "".join(ch for ch in raw if ch.isdigit())
    try:
        return int(digits) if digits else 999
    except Exception:
        return 999


def _is_eligible(row: dict | pd.Series, cfg: KRReplayConfig) -> bool:
    signal = str(row.get("판정", ""))
    score = _safe_float(row.get("종합점수", 0))
    weak = bool(row.get("모멘텀약화", False))
    return bool(
        "매수 후보" in signal
        and not weak
        and score >= float(cfg.min_score)
    )


def _decision_key(
    date_text: str,
    now: pd.Timestamp,
    snapshot: dict,
    market_context: dict,
    portfolio_context: dict,
) -> str:
    payload = {
        "date": date_text,
        "at": now.isoformat(),
        "model": AI_CFG.model,
        "snapshot": snapshot,
        "market_context": market_context,
        "portfolio_context": {
            "current_positions": portfolio_context.get("current_positions", 0),
            "max_positions": portfolio_context.get("max_positions", 3),
            "buy2_enabled": False,
        },
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _evaluate_batch(
    date_text: str,
    now: pd.Timestamp,
    latest_top5: pd.DataFrame,
    candidate_rows: list[dict],
    portfolio_context: dict,
    decision_cache: dict,
) -> tuple[dict[str, dict], int]:
    """Evaluate only currently visible candidates. No future data is supplied."""
    if not AI_CFG.api_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")

    all_items = [
        _candidate_snapshot(row, i + 1)
        for i, row in enumerate(latest_top5.to_dict("records"))
    ]
    market_context = _market_context(all_items)

    snapshots = []
    key_by_symbol: dict[str, str] = {}
    result_by_symbol: dict[str, dict] = {}

    for row in candidate_rows:
        snap = _candidate_snapshot(row, _rank_number(row))
        symbol = str(snap.get("symbol", "")).zfill(6)
        key = _decision_key(
            date_text,
            now,
            snap,
            market_context,
            portfolio_context,
        )
        key_by_symbol[symbol] = key
        cached = decision_cache.get(key)
        if isinstance(cached, dict):
            out = dict(cached)
            out["cached"] = True
            result_by_symbol[symbol] = out
        else:
            snapshots.append(snap)

    api_calls = 0
    if snapshots:
        parsed, latency_ms = _call_openai(
            market="KR",
            now_iso=now.isoformat(),
            candidates=snapshots,
            market_context=market_context,
            portfolio_context=portfolio_context,
        )
        api_calls += 1
        evaluations = parsed.get("evaluations", []) or []
        eval_by_symbol = {
            str(x.get("symbol", "")).strip().zfill(6): x
            for x in evaluations
            if isinstance(x, dict)
        }

        for snap in snapshots:
            symbol = str(snap.get("symbol", "")).zfill(6)
            ev = eval_by_symbol.get(symbol)
            if not ev:
                raise RuntimeError(f"AI 응답에 {symbol} 평가가 없습니다.")

            tech = max(0.0, min(100.0, _safe_float(ev.get("technical_score"))))
            mkt = max(0.0, min(100.0, _safe_float(ev.get("market_score"))))
            risk = max(0.0, min(100.0, _safe_float(ev.get("risk_score"))))
            score, decision, chair_code = _chair(tech, mkt, risk)
            flags = [
                str(x).strip()
                for x in (ev.get("flags", []) or [])
                if str(x).strip()
            ]
            if chair_code not in flags:
                flags.append(chair_code)

            item = {
                "date": date_text,
                "time": now.isoformat(),
                "symbol": symbol,
                "name": snap.get("name", symbol),
                "rank": snap.get("rank", 999),
                "strategy_score": snap.get("strategy_score", 0),
                "technical_score": round(tech, 1),
                "technical_vote": ev.get("technical_vote", ""),
                "technical_reason": str(ev.get("technical_reason", ""))[:260],
                "market_score": round(mkt, 1),
                "market_vote": ev.get("market_vote", ""),
                "market_reason": str(ev.get("market_reason", ""))[:260],
                "risk_score": round(risk, 1),
                "risk_vote": ev.get("risk_vote", ""),
                "risk_reason": str(ev.get("risk_reason", ""))[:260],
                "committee_score": round(score, 1),
                "decision": decision,
                "confidence": max(0.0, min(100.0, _safe_float(ev.get("confidence")))),
                "flags": flags[:12],
                "model": AI_CFG.model,
                "latency_ms": int(latency_ms),
                "snapshot": snap,
                "market_context": market_context,
                "portfolio_context": dict(portfolio_context),
                "cached": False,
            }
            key = key_by_symbol[symbol]
            decision_cache[key] = item
            result_by_symbol[symbol] = item

    return result_by_symbol, api_calls


def _episode_summary(events: list[dict]) -> list[dict]:
    """Turn partial exits into one trade episode per BUY1 -> flat."""
    active: dict[str, dict] = {}
    episodes: list[dict] = []

    for ev in events:
        symbol = str(ev.get("종목코드", "")).zfill(6)
        action = str(ev.get("액션", ""))
        side = str(ev.get("구분", ""))
        qty = int(ev.get("수량", 0) or 0)
        amount = int(ev.get("주문금액KRW", 0) or 0)
        pnl = int(ev.get("실현손익KRW", 0) or 0)

        if action == "BUY1" and side == "BUY":
            active[symbol] = {
                "key": f"{ev.get('시간KST','')}|{symbol}",
                "매수시간": ev.get("시간KST", ""),
                "종목코드": symbol,
                "종목명": ev.get("종목명", symbol),
                "매수수량": qty,
                "잔여수량": qty,
                "매수금액KRW": amount,
                "실현손익KRW": 0,
                "종료사유": "",
                "매도시간": "",
            }
            continue

        if side == "SELL" and symbol in active:
            ep = active[symbol]
            ep["잔여수량"] = max(0, int(ep.get("잔여수량", 0)) - qty)
            ep["실현손익KRW"] = int(ep.get("실현손익KRW", 0)) + pnl
            ep["종료사유"] = action
            ep["매도시간"] = ev.get("시간KST", "")
            if ep["잔여수량"] <= 0:
                buy_amount = max(1, int(ep.get("매수금액KRW", 0)))
                ep["수익률"] = round(
                    int(ep.get("실현손익KRW", 0)) / buy_amount * 100.0,
                    3,
                )
                episodes.append(ep)
                active.pop(symbol, None)

    return episodes


def _simulate_ai_gate(
    date_text: str,
    codes: Iterable[str] | None,
    refresh_ai: bool,
) -> dict:
    cfg = KRReplayConfig(buy2_mode="NONE")
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

    decision_path = _decision_cache_path(date_text)
    decision_cache = {} if refresh_ai else _load_json(decision_path, {})
    if not isinstance(decision_cache, dict):
        decision_cache = {}

    date0 = pd.Timestamp(date_text, tz=KST)
    start = date0 + pd.Timedelta(hours=9, minutes=9)
    end = date0 + pd.Timedelta(hours=15, minutes=16)
    last_entry_sec = _clock_seconds(cfg.last_entry_time)
    force_exit_sec = _clock_seconds(cfg.force_exit_time)

    # Current domestic C strategy: 50% first entry, no BUY2.
    total_pct = max(1, cfg.buy1_pct + cfg.buy2_pct)
    buy1_amount = int(cfg.per_stock_budget_krw * cfg.buy1_pct / total_pct)

    positions: dict[str, dict] = {}
    events: list[dict] = []
    decisions: list[dict] = []
    latest_top5 = pd.DataFrame()
    last_scan = None
    daily_buy_amount = 0.0
    daily_orders = 0
    api_calls = 0
    decision_seen: set[str] = set()

    def add_event(
        ts, symbol, action, side, qty, ref_price, fill_price, reason,
        pnl="", realized=0.0, score="", rank="", ai_decision="", ai_score="",
    ):
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
            "AI판정": ai_decision,
            "AI점수": ai_score,
            "이유": reason,
        })
        daily_orders += 1

    now = start
    while now <= end:
        if last_scan is None or (now - last_scan).total_seconds() >= int(cfg.scan_seconds):
            latest_top5 = _build_top5_at(
                target_frames, meta, date_text, now, cfg.scan_count
            )
            last_scan = now

        top5_map: dict[str, pd.Series] = {}
        if latest_top5 is not None and not latest_top5.empty:
            for _, r in latest_top5.iterrows():
                top5_map[str(r.get("종목코드", "")).zfill(6)] = r

        # 1) Existing positions use exactly the same deterministic exits.
        for symbol, pos in list(positions.items()):
            frame = target_frames.get(symbol)
            if frame is None:
                continue
            ref_price = _price_at(frame, date_text, now)
            if ref_price <= 0:
                continue
            qty = int(pos.get("qty", 0))
            avg = float(pos.get("avg_price", 0) or 0)
            if qty <= 0 or avg <= 0:
                continue

            pnl = (ref_price / avg - 1.0) * 100.0
            peak = max(float(pos.get("peak_pnl", pnl)), pnl)
            pos["peak_pnl"] = peak
            dd = max(0.0, peak - pnl)

            if _seconds_of_day(now) >= force_exit_sec:
                fill = _fill_price(cfg, "SELL", ref_price)
                realized = (fill - avg) * qty
                add_event(
                    now, symbol, "FORCE_SELL", "SELL", qty, ref_price, fill,
                    f"당일 강제청산 {cfg.force_exit_time} KST", pnl, realized,
                    ai_decision=pos.get("ai_decision", ""),
                    ai_score=pos.get("ai_score", ""),
                )
                positions.pop(symbol, None)
                continue

            if pnl <= -abs(cfg.stop_loss_pct):
                fill = _fill_price(cfg, "SELL", ref_price)
                realized = (fill - avg) * qty
                add_event(
                    now, symbol, "STOP_LOSS", "SELL", qty, ref_price, fill,
                    f"손절 {pnl:.2f}%", pnl, realized,
                    ai_decision=pos.get("ai_decision", ""),
                    ai_score=pos.get("ai_score", ""),
                )
                positions.pop(symbol, None)
                continue

            if pnl >= cfg.take1_pct and not bool(pos.get("take1_sent")):
                sell_qty = max(1, qty // 2)
                fill = _fill_price(cfg, "SELL", ref_price)
                realized = (fill - avg) * sell_qty
                add_event(
                    now, symbol, "TAKE1", "SELL", sell_qty, ref_price, fill,
                    f"1차 익절 {pnl:.2f}% · 약 50%", pnl, realized,
                    ai_decision=pos.get("ai_decision", ""),
                    ai_score=pos.get("ai_score", ""),
                )
                pos["qty"] = qty - sell_qty
                pos["take1_sent"] = True
                if pos["qty"] <= 0:
                    positions.pop(symbol, None)
                continue

            if pnl >= cfg.take2_pct and bool(pos.get("take1_sent")):
                fill = _fill_price(cfg, "SELL", ref_price)
                realized = (fill - avg) * qty
                add_event(
                    now, symbol, "TAKE2", "SELL", qty, ref_price, fill,
                    f"2차 익절 {pnl:.2f}% · 전량", pnl, realized,
                    ai_decision=pos.get("ai_decision", ""),
                    ai_score=pos.get("ai_score", ""),
                )
                positions.pop(symbol, None)
                continue

            if peak >= cfg.profit_guard_trigger_pct and dd >= cfg.profit_guard_drawdown_pct:
                if not bool(pos.get("take1_sent")):
                    sell_qty = max(1, qty // 2)
                    fill = _fill_price(cfg, "SELL", ref_price)
                    realized = (fill - avg) * sell_qty
                    add_event(
                        now, symbol, "PROFIT_GUARD1", "SELL", sell_qty,
                        ref_price, fill,
                        f"수익보호 1차 · 최고 +{peak:.2f}% → 현재 {pnl:.2f}% ({dd:.2f}%p 되밀림)",
                        pnl, realized,
                        ai_decision=pos.get("ai_decision", ""),
                        ai_score=pos.get("ai_score", ""),
                    )
                    pos["qty"] = qty - sell_qty
                    pos["take1_sent"] = True
                    if pos["qty"] <= 0:
                        positions.pop(symbol, None)
                    continue
                else:
                    fill = _fill_price(cfg, "SELL", ref_price)
                    realized = (fill - avg) * qty
                    add_event(
                        now, symbol, "PROFIT_GUARD2", "SELL", qty,
                        ref_price, fill,
                        f"수익보호 2차 · 최고 +{peak:.2f}% → 현재 {pnl:.2f}% ({dd:.2f}%p 되밀림)",
                        pnl, realized,
                        ai_decision=pos.get("ai_decision", ""),
                        ai_score=pos.get("ai_score", ""),
                    )
                    positions.pop(symbol, None)
                    continue

        # 2) New entries: current strategy must first say BUY, then AI APPROVE only.
        if (
            _seconds_of_day(now) < last_entry_sec
            and len(positions) < cfg.max_positions
            and daily_orders < cfg.max_daily_orders
            and latest_top5 is not None
            and not latest_top5.empty
        ):
            eligible_rows: list[dict] = []
            for _, row in latest_top5.iterrows():
                symbol = str(row.get("종목코드", "")).zfill(6)
                if symbol in positions:
                    continue
                if not _is_eligible(row, cfg):
                    continue
                eligible_rows.append(row.to_dict())

            if eligible_rows:
                free_slots = max(0, cfg.max_positions - len(positions))
                # Evaluate enough rows to fill free slots, plus one alternate.
                batch_rows = eligible_rows[: max(1, min(5, free_slots + 1))]
                portfolio_context = {
                    "mode": "HISTORICAL_REPLAY",
                    "current_positions": len(positions),
                    "max_positions": int(cfg.max_positions),
                    "daily_budget": int(cfg.daily_budget_krw),
                    "remaining_daily_budget": int(
                        max(0, cfg.daily_budget_krw - daily_buy_amount)
                    ),
                    "buy2_enabled": False,
                    "future_data_visible": False,
                }
                evals, calls = _evaluate_batch(
                    date_text=date_text,
                    now=last_scan if last_scan is not None else now,
                    latest_top5=latest_top5,
                    candidate_rows=batch_rows,
                    portfolio_context=portfolio_context,
                    decision_cache=decision_cache,
                )
                api_calls += calls

                for row in eligible_rows:
                    if len(positions) >= cfg.max_positions or daily_orders >= cfg.max_daily_orders:
                        break
                    symbol = str(row.get("종목코드", "")).zfill(6)
                    decision = evals.get(symbol)
                    if decision is None:
                        # If an alternate beyond the first batch is actually needed,
                        # evaluate just that candidate at the same historical snapshot.
                        extra, calls = _evaluate_batch(
                            date_text=date_text,
                            now=last_scan if last_scan is not None else now,
                            latest_top5=latest_top5,
                            candidate_rows=[row],
                            portfolio_context={
                                **portfolio_context,
                                "current_positions": len(positions),
                                "remaining_daily_budget": int(
                                    max(0, cfg.daily_budget_krw - daily_buy_amount)
                                ),
                            },
                            decision_cache=decision_cache,
                        )
                        api_calls += calls
                        decision = extra.get(symbol)
                    if not decision:
                        continue

                    decision_key = f"{decision.get('time','')}|{symbol}|{decision.get('decision','')}"
                    if decision_key not in decision_seen:
                        decisions.append(decision)
                        decision_seen.add(decision_key)

                    if str(decision.get("decision", "")) != "APPROVE":
                        continue

                    frame = target_frames.get(symbol)
                    if frame is None:
                        continue
                    ref_price = _price_at(frame, date_text, now)
                    if ref_price <= 0:
                        continue
                    fill = _fill_price(cfg, "BUY", ref_price)
                    qty1 = int(buy1_amount // fill)
                    if qty1 <= 0:
                        continue
                    cost = fill * qty1
                    if daily_buy_amount + cost > cfg.daily_budget_krw:
                        continue

                    score = _safe_float(row.get("종합점수", 0))
                    rank = str(row.get("순위", ""))
                    r3 = _safe_float(row.get("최근3분수익률", 0))
                    r5 = _safe_float(row.get("최근5분수익률", 0))
                    vr = _safe_float(row.get("거래량배수", 0))
                    add_event(
                        now, symbol, "BUY1", "BUY", qty1, ref_price, fill,
                        (
                            f"AI 승인 BUY · 위원회 {decision.get('committee_score', 0):.1f}점 · "
                            f"전략 {score:.1f}점 · 3분 {r3:+.2f}% · "
                            f"5분 {r5:+.2f}% · 거래량 {vr:.2f}배"
                        ),
                        "", 0.0, score, rank,
                        ai_decision="APPROVE",
                        ai_score=decision.get("committee_score", ""),
                    )
                    positions[symbol] = {
                        "qty": qty1,
                        "avg_price": fill,
                        "stage": 1,
                        "created_at": now.isoformat(),
                        "take1_sent": False,
                        "peak_pnl": 0.0,
                        "opened_at": now.isoformat(),
                        "ai_decision": "APPROVE",
                        "ai_score": decision.get("committee_score", ""),
                    }
                    daily_buy_amount += cost

        now += pd.Timedelta(seconds=int(cfg.manage_seconds))

    # Safety close if anything remains after the replay window.
    if positions:
        for symbol, pos in list(positions.items()):
            frame = target_frames.get(symbol)
            ref_price = _price_at(frame, date_text, end) if frame is not None else 0.0
            if ref_price <= 0:
                continue
            qty = int(pos.get("qty", 0))
            avg = float(pos.get("avg_price", 0) or 0)
            fill = _fill_price(cfg, "SELL", ref_price)
            pnl = (ref_price / avg - 1.0) * 100.0 if avg > 0 else 0.0
            realized = (fill - avg) * qty
            add_event(
                end, symbol, "FORCE_SELL_END", "SELL", qty, ref_price, fill,
                "리플레이 종료 안전청산", pnl, realized,
                ai_decision=pos.get("ai_decision", ""),
                ai_score=pos.get("ai_score", ""),
            )
            positions.pop(symbol, None)

    _atomic_json(decision_path, decision_cache)

    df = pd.DataFrame(events)
    buy_amount = float(df.loc[df["구분"] == "BUY", "주문금액KRW"].sum()) if not df.empty else 0.0
    sell_amount = float(df.loc[df["구분"] == "SELL", "주문금액KRW"].sum()) if not df.empty else 0.0
    realized = float(df["실현손익KRW"].sum()) if not df.empty else 0.0
    episodes = _episode_summary(events)

    return {
        "summary": {
            "총주문횟수": int(len(df)),
            "매수주문횟수": int((df["구분"] == "BUY").sum()) if not df.empty else 0,
            "매도주문횟수": int((df["구분"] == "SELL").sum()) if not df.empty else 0,
            "거래횟수": len(episodes),
            "수익거래수": sum(1 for x in episodes if int(x.get("실현손익KRW", 0)) > 0),
            "손실거래수": sum(1 for x in episodes if int(x.get("실현손익KRW", 0)) < 0),
            "누적매수금액KRW": int(round(buy_amount)),
            "누적매도금액KRW": int(round(sell_amount)),
            "실현손익KRW": int(round(realized)),
            "일일예산1000만원대비수익률": round(
                realized / cfg.daily_budget_krw * 100.0, 3
            ),
        },
        "events": events,
        "episodes": episodes,
        "decisions": decisions,
        "api_calls": api_calls,
        "data_available_count": len(target_frames),
        "universe_count": len(universe),
    }


def run_kr_ai_committee_replay(
    date_text: str = "2026-08-10",
    codes: Iterable[str] | None = None,
    refresh: bool = False,
) -> dict:
    """
    Compare current domestic C strategy (BUY2 off) vs AI committee APPROVE-only gate.
    The AI only sees information available at the historical decision time.
    """
    cache = _cache_path(date_text)
    if not refresh and not codes and cache.exists():
        cached = _load_json(cache, None)
        if (
            isinstance(cached, dict)
            and cached.get("ok") is True
            and cached.get("version") == AI_REPLAY_VERSION
        ):
            cached["cached"] = True
            return cached

    if not AI_CFG.api_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않아 AI 리플레이를 실행할 수 없습니다.")

    baseline = run_kr_trade_replay(
        date_text=date_text,
        codes=codes,
        config=KRReplayConfig(buy2_mode="NONE"),
        use_cache=False,
    )
    ai = _simulate_ai_gate(date_text, codes, refresh_ai=refresh)

    baseline_events = list(baseline.get("events", []) or [])
    baseline_episodes = _episode_summary(baseline_events)
    base_map = {
        f"{x.get('매수시간','')}|{x.get('종목코드','')}": x
        for x in baseline_episodes
    }

    blocked_loss_avoided = 0
    missed_profit = 0
    matched_blocked = []
    for d in ai.get("decisions", []):
        if str(d.get("decision", "")) == "APPROVE":
            continue
        key = f"{d.get('time','')}|{str(d.get('symbol','')).zfill(6)}"
        # Decision time is scan time; BUY attempt may be 45s later on same snapshot.
        # First try exact; then same symbol within 60s.
        ep = base_map.get(key)
        if ep is None:
            dts = pd.Timestamp(d.get("time"))
            candidates = [
                x for x in baseline_episodes
                if str(x.get("종목코드", "")).zfill(6) == str(d.get("symbol", "")).zfill(6)
            ]
            for x in candidates:
                try:
                    ets = pd.Timestamp(x.get("매수시간"))
                    if abs((ets - dts).total_seconds()) <= 60:
                        ep = x
                        break
                except Exception:
                    pass
        if ep is None:
            continue
        pnl = int(ep.get("실현손익KRW", 0) or 0)
        if pnl < 0:
            blocked_loss_avoided += -pnl
        elif pnl > 0:
            missed_profit += pnl
        matched_blocked.append({
            "종목코드": d.get("symbol", ""),
            "종목명": d.get("name", ""),
            "시간": d.get("time", ""),
            "AI판정": d.get("decision", ""),
            "AI점수": d.get("committee_score", 0),
            "기술점수": d.get("technical_score", 0),
            "시장점수": d.get("market_score", 0),
            "리스크점수": d.get("risk_score", 0),
            "기존전략해당거래손익KRW": pnl,
            "기존전략종료사유": ep.get("종료사유", ""),
        })

    baseline_pnl = int((baseline.get("summary", {}) or {}).get("실현손익KRW", 0) or 0)
    ai_pnl = int((ai.get("summary", {}) or {}).get("실현손익KRW", 0) or 0)
    decisions = list(ai.get("decisions", []) or [])

    counts = {
        "APPROVE": sum(1 for x in decisions if x.get("decision") == "APPROVE"),
        "HOLD": sum(1 for x in decisions if x.get("decision") == "HOLD"),
        "REJECT": sum(1 for x in decisions if x.get("decision") == "REJECT"),
    }

    payload = {
        "ok": True,
        "version": AI_REPLAY_VERSION,
        "date": date_text,
        "model": AI_CFG.model,
        "policy": {
            "baseline": "KR_C_NO_BUY2",
            "ai_gate": "기존 매수조건 충족 AND AI_COMMITTEE=APPROVE",
            "hold_reject_action": "NO_BUY",
            "buy1": "종목당 예산의 50%",
            "buy2": "OFF",
            "future_data_visible_to_ai": False,
            "same_exit_rules": True,
        },
        "comparison": {
            "기존C전략실현손익KRW": baseline_pnl,
            "AI승인전략실현손익KRW": ai_pnl,
            "AI순효과KRW": ai_pnl - baseline_pnl,
            "기존C전략수익률": (baseline.get("summary", {}) or {}).get(
                "일일예산1000만원대비수익률", 0
            ),
            "AI승인전략수익률": (ai.get("summary", {}) or {}).get(
                "일일예산1000만원대비수익률", 0
            ),
            "AI가막은기존손실KRW": int(blocked_loss_avoided),
            "AI가막아놓친기존수익KRW": int(missed_profit),
        },
        "ai_decision_counts": counts,
        "api_calls_this_run": int(ai.get("api_calls", 0) or 0),
        "baseline_summary": baseline.get("summary", {}),
        "ai_summary": ai.get("summary", {}),
        "blocked_baseline_matches": matched_blocked,
        "ai_decisions": decisions,
        "ai_events": ai.get("events", []),
        "assumptions": {
            "real_orders": False,
            "data": "yfinance 1-minute historical bars",
            "candidate_reconstruction": "65개 고정 유동성 종목군 안에서 과거 후보 근사 복원",
            "ai_information_rule": "각 판단시점까지의 데이터만 입력하며 이후 가격은 AI에 제공하지 않음",
            "slippage": "매수 +0.10%, 매도 -0.10% 가정",
            "fees_taxes": "별도 미포함",
            "important_limit": "과거 KIS 전체시장 실시간 거래량랭킹 원본이 없어 당시 전체시장 TOP5를 100% 복원한 것은 아님",
            "interpretation": "전략 비교 실험이며 실전 수익을 보장하지 않음",
        },
        "cached": False,
    }

    if not codes:
        _atomic_json(cache, payload)
    return payload


def summarize_kr_ai_committee_replays(
    dates: Iterable[str] | None = None,
) -> dict:
    dates = list(dates or DEFAULT_DATES)
    daily = []
    missing = []

    for date_text in dates:
        payload = _load_json(_cache_path(date_text), None)
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            missing.append(date_text)
            continue
        c = payload.get("comparison", {}) or {}
        daily.append({
            "date": date_text,
            "기존C전략실현손익KRW": int(c.get("기존C전략실현손익KRW", 0) or 0),
            "AI승인전략실현손익KRW": int(c.get("AI승인전략실현손익KRW", 0) or 0),
            "AI순효과KRW": int(c.get("AI순효과KRW", 0) or 0),
            "AI가막은기존손실KRW": int(c.get("AI가막은기존손실KRW", 0) or 0),
            "AI가막아놓친기존수익KRW": int(c.get("AI가막아놓친기존수익KRW", 0) or 0),
            "APPROVE": int((payload.get("ai_decision_counts", {}) or {}).get("APPROVE", 0) or 0),
            "HOLD": int((payload.get("ai_decision_counts", {}) or {}).get("HOLD", 0) or 0),
            "REJECT": int((payload.get("ai_decision_counts", {}) or {}).get("REJECT", 0) or 0),
        })

    baseline_total = sum(x["기존C전략실현손익KRW"] for x in daily)
    ai_total = sum(x["AI승인전략실현손익KRW"] for x in daily)
    net = ai_total - baseline_total

    return {
        "ok": len(daily) > 0,
        "version": "kr-ai-committee-replay-summary-v1",
        "requested_dates": dates,
        "completed_dates": [x["date"] for x in daily],
        "missing_dates": missing,
        "daily": daily,
        "total": {
            "기존C전략누적손익KRW": baseline_total,
            "AI승인전략누적손익KRW": ai_total,
            "AI누적순효과KRW": net,
            "AI가막은기존손실합계KRW": sum(x["AI가막은기존손실KRW"] for x in daily),
            "AI가막아놓친기존수익합계KRW": sum(x["AI가막아놓친기존수익KRW"] for x in daily),
        },
        "status": (
            "5일 결과 준비 완료"
            if not missing
            else "아직 실행하지 않은 날짜가 있습니다. 일별 AI 리플레이를 먼저 실행하세요."
        ),
        "warning": "국내 과거 후보군은 근사 복원이며 수수료·세금은 별도 미포함입니다.",
    }

# ============================================================================
# AI 투자위원회 V2 리플레이
# - 실전/모의 주문 Worker의 현재 C 전략은 건드리지 않는다.
# - C 전략이 운전하고, AI는 위험 감사관(auditor) 역할만 한다.
# - ALLOW: 기존 BUY1 크기(종목당 예산의 50%)
# - CAUTION: 절반 크기(종목당 예산의 25%)
# - VETO: 명백한 복합 위험일 때만 매수 차단
# ============================================================================

AI_REPLAY_V2_VERSION = "kr-ai-committee-replay-v2"
V2_REPLAY_DIR = STATE_DIR / "replays" / "ai_committee_v2" / "kr"
V2_REPLAY_DIR.mkdir(parents=True, exist_ok=True)

V2_SYSTEM_PROMPT = """\
너는 단기 모멘텀 자동매매 시스템의 'AI 위험 감사위원회 V2'다.
실제 주문 권한은 없고, 기존 C 전략이 이미 뽑은 매수 후보의 위험을 감사한다.

핵심 원칙:
- 기존 C 전략은 당일 강한 대장주/돌파/거래량 가속을 찾는 모멘텀 전략이다.
- '당일 상승률이 높다', '장중 고점에 가깝다', '돌파 중이다'라는 사실만으로
  위험하다고 과도하게 감점하지 마라. 강한 대장주에서는 오히려 정상적인 특성일 수 있다.
- 대신 실제 시장 대비 상대강도(relative_strength), VWAP 괴리(vwap_gap),
  거래량 가속, 3/5/10/20분 모멘텀의 정렬, 돌파 지속성을 함께 본다.
- 특히 당일 약세인데 짧은 반등만 강한 REBOUND_TRAP, 시장 대비 현저한 약세,
  VWAP 아래에서의 반등, 거래량이 꺼진 돌파, 단기 모멘텀 불일치는 강하게 경계한다.
- 보유 포지션 수가 많다는 이유 하나만으로 후보를 거부하지 말고, 실제 후보 위험과 함께 본다.

세 위원:
1) 기술·대장주 위원: 진짜 주도력/돌파 지속성/거래량/모멘텀/VWAP/상대강도 평가.
2) 시장환경 위원: 제공된 KOSPI/KOSDAQ 수익률과 후보군 폭을 평가. 없는 사실은 만들지 않는다.
3) 리스크 위원: 지금 진입했을 때 실패 가능성이 명백히 커지는 복합 위험만 강하게 감점.

risk_score는 '안전도'다. 높을수록 현재 진입 위험이 상대적으로 낮다.
reason은 짧고 구체적으로 쓰고, 미래 가격이나 수익을 단정하지 않는다.
flags는 STRONG_MOMENTUM, NEAR_HIGH, BREAKOUT, CHASE_RISK,
REBOUND_TRAP_RISK, WEAK_VOLUME, BELOW_VWAP, WEAK_RELATIVE_STRENGTH,
MARKET_UNCERTAIN 같은 짧은 코드형 문자열을 사용한다.
"""


def _v2_cache_path(date_text: str) -> Path:
    return V2_REPLAY_DIR / f"kr_ai_committee_v2_replay_{date_text}.json"


def _v2_decision_cache_path(date_text: str) -> Path:
    return V2_REPLAY_DIR / f"kr_ai_v2_decisions_{date_text}.json"


def _download_kr_benchmarks(date_text: str) -> dict[str, pd.DataFrame]:
    """Best-effort KOSPI/KOSDAQ 1-minute benchmark download."""
    try:
        import yfinance as yf
    except Exception:
        return {}

    target = pd.Timestamp(date_text)
    start = (target - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    end = (target + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    tickers = ["^KS11", "^KQ11"]
    try:
        raw = yf.download(
            tickers=" ".join(tickers),
            start=start,
            end=end,
            interval="1m",
            group_by="ticker",
            auto_adjust=False,
            prepost=False,
            threads=True,
            progress=False,
            timeout=20,
        )
    except Exception:
        return {}

    out: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            d = _extract_yf_frame(raw, ticker)
        except Exception:
            d = pd.DataFrame()
        if d is not None and not d.empty:
            day = d[d.index.strftime("%Y-%m-%d") == date_text]
            if not day.empty:
                out[ticker] = d
    return out


def _benchmark_return_at(
    benchmarks: dict[str, pd.DataFrame],
    exchange: str,
    date_text: str,
    cutoff: pd.Timestamp,
) -> float | None:
    ticker = "^KQ11" if str(exchange).upper() == "KQ" else "^KS11"
    frame = benchmarks.get(ticker)
    if frame is None or frame.empty:
        return None
    try:
        d = _bars_until(frame, cutoff, date_text)
        prev = _previous_close(frame, date_text)
        if d.empty or prev <= 0:
            return None
        last = float(d["Close"].iloc[-1])
        return (last / prev - 1.0) * 100.0
    except Exception:
        return None


def _vwap_gap_at(
    frame: pd.DataFrame,
    date_text: str,
    cutoff: pd.Timestamp,
) -> float | None:
    try:
        d = _bars_until(frame, cutoff, date_text)
        if d.empty:
            return None
        close = pd.to_numeric(d["Close"], errors="coerce").fillna(0.0)
        vol = pd.to_numeric(d.get("Volume", 0), errors="coerce").fillna(0.0)
        total_vol = float(vol.sum())
        last = float(close.iloc[-1])
        if total_vol <= 0 or last <= 0:
            return None
        vwap = float((close * vol).sum() / total_vol)
        if vwap <= 0:
            return None
        return (last / vwap - 1.0) * 100.0
    except Exception:
        return None


def _enrich_top5_v2(
    top5: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    meta: dict[str, dict],
    benchmarks: dict[str, pd.DataFrame],
    date_text: str,
    cutoff: pd.Timestamp,
) -> tuple[pd.DataFrame, dict]:
    if top5 is None or top5.empty:
        return pd.DataFrame(), {
            "kospi_return": None,
            "kosdaq_return": None,
            "benchmark_available": False,
        }

    out = top5.copy()
    vwap_values = []
    rs_values = []
    market_returns = []

    for _, row in out.iterrows():
        symbol = str(row.get("종목코드", "")).zfill(6)
        frame = frames.get(symbol)
        exch = str((meta.get(symbol, {}) or {}).get("exchange", "KS"))
        market_ret = _benchmark_return_at(
            benchmarks, exch, date_text, cutoff
        )
        vwap_gap = (
            _vwap_gap_at(frame, date_text, cutoff)
            if frame is not None else None
        )
        day_ret = _safe_float(
            row.get("등락률", row.get("당일등락률", 0))
        )
        relative_strength = (
            day_ret - market_ret if market_ret is not None else 0.0
        )
        vwap_values.append(0.0 if vwap_gap is None else round(vwap_gap, 3))
        rs_values.append(round(relative_strength, 3))
        market_returns.append(
            "" if market_ret is None else round(float(market_ret), 3)
        )

    out["VWAP괴리율"] = vwap_values
    out["상대강도"] = rs_values
    out["시장등락률"] = market_returns

    ks = _benchmark_return_at(benchmarks, "KS", date_text, cutoff)
    kq = _benchmark_return_at(benchmarks, "KQ", date_text, cutoff)
    return out, {
        "kospi_return": None if ks is None else round(float(ks), 3),
        "kosdaq_return": None if kq is None else round(float(kq), 3),
        "benchmark_available": bool(ks is not None or kq is not None),
        "benchmark_source": "yfinance:^KS11/^KQ11",
    }


def _v2_weighted_score(tech: float, market: float, risk: float) -> float:
    # 점수는 표시/분석용. 최종 V2 행동은 아래 audit policy가 결정한다.
    return round(tech * 0.45 + market * 0.20 + risk * 0.35, 1)


def _v2_audit_action(item: dict) -> tuple[str, list[str]]:
    """Map AI scores + raw features to ALLOW / CAUTION / VETO.

    VETO는 단순 추격/고점 근접만으로 발생하지 않는다.
    명백한 복합 약세/반등함정/모멘텀 붕괴에만 사용한다.
    """
    snap = item.get("snapshot", {}) or {}
    flags = {str(x).strip().upper() for x in (item.get("flags", []) or [])}
    risk = _safe_float(item.get("risk_score", 50))
    market = _safe_float(item.get("market_score", 50))
    day = _safe_float(snap.get("day_return", 0))
    r3 = _safe_float(snap.get("ret3", 0))
    r5 = _safe_float(snap.get("ret5", 0))
    r10 = _safe_float(snap.get("ret10", 0))
    vr = _safe_float(snap.get("volume_ratio", 0))
    rs = _safe_float(snap.get("relative_strength", 0))
    vwap = _safe_float(snap.get("vwap_gap", 0))
    breakout = bool(snap.get("breakout", False))

    veto_reasons: list[str] = []
    if risk < 30:
        veto_reasons.append("RISK_SCORE_LT30")
    if r3 < -0.25 or r5 < -0.40:
        veto_reasons.append("SHORT_MOMENTUM_BREAKDOWN")
    if (
        ("REBOUND_TRAP_RISK" in flags or day < 0)
        and day < 0
        and rs < -0.35
        and (vwap < 0 or vr < 1.05)
    ):
        veto_reasons.append("REBOUND_TRAP_CONFIRMED")
    if rs < -1.0 and day <= 0 and not breakout:
        veto_reasons.append("WEAK_RS_NO_BREAKOUT")
    if vwap < -0.60 and rs < 0:
        veto_reasons.append("BELOW_VWAP_AND_WEAK_RS")
    if "WEAK_VOLUME" in flags and vr < 0.70 and not breakout:
        veto_reasons.append("VERY_WEAK_VOLUME")

    if veto_reasons:
        return "VETO", veto_reasons

    caution: list[str] = []
    if risk < 55:
        caution.append("RISK_SCORE_LT55")
    if market < 48:
        caution.append("MARKET_SCORE_LT48")
    if day >= 10.0 and (vr < 1.20 or r3 < 0.50):
        caution.append("EXTENDED_WITHOUT_FRESH_ACCELERATION")
    if day < 0 and rs < 0:
        caution.append("NEGATIVE_DAY_WEAK_RS")
    if vwap < 0 and rs < 0:
        caution.append("BELOW_VWAP_WEAK_RS")
    if "WEAK_VOLUME" in flags and vr < 1.0:
        caution.append("WEAK_VOLUME")
    if r10 > 0 and r3 < max(0.10, r10 * 0.20):
        caution.append("MOMENTUM_DECELERATION")

    if caution:
        return "CAUTION", caution
    return "ALLOW", []


def _v2_decision_key(
    date_text: str,
    now: pd.Timestamp,
    snapshot: dict,
    market_context: dict,
    portfolio_context: dict,
) -> str:
    payload = {
        "version": AI_REPLAY_V2_VERSION,
        "date": date_text,
        "at": now.isoformat(),
        "model": AI_CFG.model,
        "snapshot": snapshot,
        "market_context": market_context,
        "portfolio_context": {
            "current_positions": portfolio_context.get("current_positions", 0),
            "max_positions": portfolio_context.get("max_positions", 3),
            "buy2_enabled": False,
        },
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _call_openai_v2(
    now_iso: str,
    candidates: list[dict],
    market_context: dict,
    portfolio_context: dict,
) -> tuple[dict, int]:
    if not AI_CFG.api_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")

    user_payload = {
        "market": "KR",
        "time": now_iso,
        "mode": "HISTORICAL_REPLAY_RISK_AUDITOR_V2",
        "candidates": candidates,
        "market_context": market_context,
        "portfolio_context": portfolio_context,
        "scoring_contract": {
            "technical_score": "0~100, 높을수록 현재 대장주/돌파 기술품질 우수",
            "market_score": "0~100, 높을수록 실제 지수와 후보군 환경이 우호적",
            "risk_score": "0~100, 높을수록 지금 진입 안전도가 높음",
            "important": "강한 당일 상승/고점 근접 자체만으로 낮은 risk_score를 주지 말 것",
        },
    }
    body = {
        "model": AI_CFG.model,
        "input": [
            {"role": "system", "content": V2_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    user_payload, ensure_ascii=False, default=str
                ),
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "ai_investment_committee_v2_auditor",
                "strict": True,
                "schema": OUTPUT_SCHEMA,
            }
        },
        "max_output_tokens": 1400,
    }
    started = time.perf_counter()
    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {AI_CFG.api_key}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=max(2.0, AI_CFG.timeout_seconds),
    )
    latency_ms = int(round((time.perf_counter() - started) * 1000))
    if not response.ok:
        raise RuntimeError(
            f"OpenAI API HTTP {response.status_code}: {response.text[:500]}"
        )
    data = response.json()
    text = _extract_output_text(data)
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise RuntimeError("AI V2 응답이 JSON 객체가 아닙니다.")
    return parsed, latency_ms


def _evaluate_batch_v2(
    date_text: str,
    now: pd.Timestamp,
    latest_top5: pd.DataFrame,
    candidate_rows: list[dict],
    portfolio_context: dict,
    market_extra: dict,
    decision_cache: dict,
) -> tuple[dict[str, dict], int]:
    all_items = [
        _candidate_snapshot(row, i + 1)
        for i, row in enumerate(latest_top5.to_dict("records"))
    ]
    market_context = _market_context(all_items)
    market_context.update(market_extra or {})

    snapshots: list[dict] = []
    key_by_symbol: dict[str, str] = {}
    result_by_symbol: dict[str, dict] = {}
    for row in candidate_rows:
        snap = _candidate_snapshot(row, _rank_number(row))
        symbol = str(snap.get("symbol", "")).zfill(6)
        key = _v2_decision_key(
            date_text, now, snap, market_context, portfolio_context
        )
        key_by_symbol[symbol] = key
        cached = decision_cache.get(key)
        if isinstance(cached, dict):
            out = dict(cached)
            out["cached"] = True
            result_by_symbol[symbol] = out
        else:
            snapshots.append(snap)

    api_calls = 0
    if snapshots:
        parsed, latency_ms = _call_openai_v2(
            now_iso=now.isoformat(),
            candidates=snapshots,
            market_context=market_context,
            portfolio_context=portfolio_context,
        )
        api_calls += 1
        evaluations = parsed.get("evaluations", []) or []
        eval_by_symbol = {
            str(x.get("symbol", "")).strip().zfill(6): x
            for x in evaluations if isinstance(x, dict)
        }
        for snap in snapshots:
            symbol = str(snap.get("symbol", "")).zfill(6)
            ev = eval_by_symbol.get(symbol)
            if not ev:
                raise RuntimeError(f"AI V2 응답에 {symbol} 평가가 없습니다.")
            tech = max(0.0, min(100.0, _safe_float(ev.get("technical_score"))))
            mkt = max(0.0, min(100.0, _safe_float(ev.get("market_score"))))
            risk = max(0.0, min(100.0, _safe_float(ev.get("risk_score"))))
            flags = [
                str(x).strip()
                for x in (ev.get("flags", []) or []) if str(x).strip()
            ][:12]
            item = {
                "date": date_text,
                "time": now.isoformat(),
                "symbol": symbol,
                "name": snap.get("name", symbol),
                "rank": snap.get("rank", 999),
                "strategy_score": snap.get("strategy_score", 0),
                "technical_score": round(tech, 1),
                "technical_vote": ev.get("technical_vote", ""),
                "technical_reason": str(ev.get("technical_reason", ""))[:260],
                "market_score": round(mkt, 1),
                "market_vote": ev.get("market_vote", ""),
                "market_reason": str(ev.get("market_reason", ""))[:260],
                "risk_score": round(risk, 1),
                "risk_vote": ev.get("risk_vote", ""),
                "risk_reason": str(ev.get("risk_reason", ""))[:260],
                "committee_score": _v2_weighted_score(tech, mkt, risk),
                "confidence": max(0.0, min(100.0, _safe_float(ev.get("confidence")))),
                "flags": flags,
                "model": AI_CFG.model,
                "latency_ms": int(latency_ms),
                "snapshot": snap,
                "market_context": market_context,
                "portfolio_context": dict(portfolio_context),
                "cached": False,
            }
            action, reasons = _v2_audit_action(item)
            item["audit_action"] = action
            item["audit_reasons"] = reasons
            key = key_by_symbol[symbol]
            decision_cache[key] = item
            result_by_symbol[symbol] = item

    return result_by_symbol, api_calls


def _simulate_ai_v2_auditor(
    date_text: str,
    codes: Iterable[str] | None,
    refresh_ai: bool,
) -> dict:
    cfg = KRReplayConfig(buy2_mode="NONE")
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

    benchmarks = _download_kr_benchmarks(date_text)
    decision_path = _v2_decision_cache_path(date_text)
    decision_cache = {} if refresh_ai else _load_json(decision_path, {})
    if not isinstance(decision_cache, dict):
        decision_cache = {}

    date0 = pd.Timestamp(date_text, tz=KST)
    start = date0 + pd.Timedelta(hours=9, minutes=9)
    end = date0 + pd.Timedelta(hours=15, minutes=16)
    last_entry_sec = _clock_seconds(cfg.last_entry_time)
    force_exit_sec = _clock_seconds(cfg.force_exit_time)

    total_pct = max(1, cfg.buy1_pct + cfg.buy2_pct)
    full_buy1_amount = int(cfg.per_stock_budget_krw * cfg.buy1_pct / total_pct)
    caution_buy1_amount = max(1, int(full_buy1_amount * 0.50))

    positions: dict[str, dict] = {}
    events: list[dict] = []
    decisions: list[dict] = []
    latest_top5 = pd.DataFrame()
    market_extra: dict = {}
    last_scan = None
    daily_buy_amount = 0.0
    daily_orders = 0
    api_calls = 0
    ai_errors = 0
    decision_seen: set[str] = set()
    veto_until: dict[str, pd.Timestamp] = {}

    def add_event(
        ts, symbol, action, side, qty, ref_price, fill_price, reason,
        pnl="", realized=0.0, score="", rank="", audit_action="", ai_score="",
    ):
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
            "AI_V2판정": audit_action,
            "AI점수": ai_score,
            "이유": reason,
        })
        daily_orders += 1

    now = start
    while now <= end:
        if last_scan is None or (now - last_scan).total_seconds() >= int(cfg.scan_seconds):
            raw_top5 = _build_top5_at(
                target_frames, meta, date_text, now, cfg.scan_count
            )
            latest_top5, market_extra = _enrich_top5_v2(
                raw_top5, target_frames, meta, benchmarks, date_text, now
            )
            last_scan = now

        # Existing positions: identical exits to C/V1.
        for symbol, pos in list(positions.items()):
            frame = target_frames.get(symbol)
            if frame is None:
                continue
            ref_price = _price_at(frame, date_text, now)
            if ref_price <= 0:
                continue
            qty = int(pos.get("qty", 0))
            avg = float(pos.get("avg_price", 0) or 0)
            if qty <= 0 or avg <= 0:
                continue
            pnl = (ref_price / avg - 1.0) * 100.0
            peak = max(float(pos.get("peak_pnl", pnl)), pnl)
            pos["peak_pnl"] = peak
            dd = max(0.0, peak - pnl)

            if _seconds_of_day(now) >= force_exit_sec:
                fill = _fill_price(cfg, "SELL", ref_price)
                realized = (fill - avg) * qty
                add_event(now, symbol, "FORCE_SELL", "SELL", qty, ref_price, fill,
                          f"당일 강제청산 {cfg.force_exit_time} KST", pnl, realized,
                          audit_action=pos.get("audit_action", ""), ai_score=pos.get("ai_score", ""))
                positions.pop(symbol, None)
                continue
            if pnl <= -abs(cfg.stop_loss_pct):
                fill = _fill_price(cfg, "SELL", ref_price)
                realized = (fill - avg) * qty
                add_event(now, symbol, "STOP_LOSS", "SELL", qty, ref_price, fill,
                          f"손절 {pnl:.2f}%", pnl, realized,
                          audit_action=pos.get("audit_action", ""), ai_score=pos.get("ai_score", ""))
                positions.pop(symbol, None)
                continue
            if pnl >= cfg.take1_pct and not bool(pos.get("take1_sent")):
                sell_qty = max(1, qty // 2)
                fill = _fill_price(cfg, "SELL", ref_price)
                realized = (fill - avg) * sell_qty
                add_event(now, symbol, "TAKE1", "SELL", sell_qty, ref_price, fill,
                          f"1차 익절 {pnl:.2f}% · 약 50%", pnl, realized,
                          audit_action=pos.get("audit_action", ""), ai_score=pos.get("ai_score", ""))
                pos["qty"] = qty - sell_qty
                pos["take1_sent"] = True
                if pos["qty"] <= 0:
                    positions.pop(symbol, None)
                continue
            if pnl >= cfg.take2_pct and bool(pos.get("take1_sent")):
                fill = _fill_price(cfg, "SELL", ref_price)
                realized = (fill - avg) * qty
                add_event(now, symbol, "TAKE2", "SELL", qty, ref_price, fill,
                          f"2차 익절 {pnl:.2f}% · 전량", pnl, realized,
                          audit_action=pos.get("audit_action", ""), ai_score=pos.get("ai_score", ""))
                positions.pop(symbol, None)
                continue
            if peak >= cfg.profit_guard_trigger_pct and dd >= cfg.profit_guard_drawdown_pct:
                if not bool(pos.get("take1_sent")):
                    sell_qty = max(1, qty // 2)
                    fill = _fill_price(cfg, "SELL", ref_price)
                    realized = (fill - avg) * sell_qty
                    add_event(now, symbol, "PROFIT_GUARD1", "SELL", sell_qty, ref_price, fill,
                              f"수익보호 1차 · 최고 +{peak:.2f}% → 현재 {pnl:.2f}% ({dd:.2f}%p 되밀림)",
                              pnl, realized, audit_action=pos.get("audit_action", ""), ai_score=pos.get("ai_score", ""))
                    pos["qty"] = qty - sell_qty
                    pos["take1_sent"] = True
                    if pos["qty"] <= 0:
                        positions.pop(symbol, None)
                    continue
                fill = _fill_price(cfg, "SELL", ref_price)
                realized = (fill - avg) * qty
                add_event(now, symbol, "PROFIT_GUARD2", "SELL", qty, ref_price, fill,
                          f"수익보호 2차 · 최고 +{peak:.2f}% → 현재 {pnl:.2f}% ({dd:.2f}%p 되밀림)",
                          pnl, realized, audit_action=pos.get("audit_action", ""), ai_score=pos.get("ai_score", ""))
                positions.pop(symbol, None)
                continue

        # C strategy remains driver. AI only audits eligible entries.
        if (
            _seconds_of_day(now) < last_entry_sec
            and len(positions) < cfg.max_positions
            and daily_orders < cfg.max_daily_orders
            and latest_top5 is not None
            and not latest_top5.empty
        ):
            eligible_rows: list[dict] = []
            for _, row in latest_top5.iterrows():
                symbol = str(row.get("종목코드", "")).zfill(6)
                if symbol in positions or not _is_eligible(row, cfg):
                    continue
                until = veto_until.get(symbol)
                if until is not None and now < until:
                    continue
                eligible_rows.append(row.to_dict())

            if eligible_rows:
                free_slots = max(0, cfg.max_positions - len(positions))
                batch_rows = eligible_rows[: max(1, min(5, free_slots + 1))]
                portfolio_context = {
                    "mode": "HISTORICAL_REPLAY_AUDITOR_V2",
                    "current_positions": len(positions),
                    "max_positions": int(cfg.max_positions),
                    "daily_budget": int(cfg.daily_budget_krw),
                    "remaining_daily_budget": int(max(0, cfg.daily_budget_krw - daily_buy_amount)),
                    "buy2_enabled": False,
                    "future_data_visible": False,
                }
                try:
                    evals, calls = _evaluate_batch_v2(
                        date_text=date_text,
                        now=last_scan if last_scan is not None else now,
                        latest_top5=latest_top5,
                        candidate_rows=batch_rows,
                        portfolio_context=portfolio_context,
                        market_extra=market_extra,
                        decision_cache=decision_cache,
                    )
                    api_calls += calls
                except Exception as e:
                    ai_errors += 1
                    evals = {}
                    # Auditor failure is fail-open: do not block C strategy.

                for row in eligible_rows:
                    if len(positions) >= cfg.max_positions or daily_orders >= cfg.max_daily_orders:
                        break
                    symbol = str(row.get("종목코드", "")).zfill(6)
                    decision = evals.get(symbol)
                    if decision is None and evals:
                        try:
                            extra, calls = _evaluate_batch_v2(
                                date_text=date_text,
                                now=last_scan if last_scan is not None else now,
                                latest_top5=latest_top5,
                                candidate_rows=[row],
                                portfolio_context={
                                    **portfolio_context,
                                    "current_positions": len(positions),
                                    "remaining_daily_budget": int(max(0, cfg.daily_budget_krw - daily_buy_amount)),
                                },
                                market_extra=market_extra,
                                decision_cache=decision_cache,
                            )
                            api_calls += calls
                            decision = extra.get(symbol)
                        except Exception:
                            ai_errors += 1
                            decision = None

                    if decision:
                        dkey = f"{decision.get('time','')}|{symbol}|{decision.get('audit_action','')}"
                        if dkey not in decision_seen:
                            decisions.append(decision)
                            decision_seen.add(dkey)
                        action = str(decision.get("audit_action", "ALLOW"))
                        ai_score = decision.get("committee_score", "")
                        audit_reasons = decision.get("audit_reasons", []) or []
                    else:
                        action = "FAIL_OPEN_ALLOW"
                        ai_score = ""
                        audit_reasons = ["AI_ERROR_FAIL_OPEN"]

                    if action == "VETO":
                        veto_until[symbol] = now + pd.Timedelta(minutes=3)
                        continue

                    frame = target_frames.get(symbol)
                    if frame is None:
                        continue
                    ref_price = _price_at(frame, date_text, now)
                    if ref_price <= 0:
                        continue
                    fill = _fill_price(cfg, "BUY", ref_price)
                    target_amount = (
                        caution_buy1_amount if action == "CAUTION" else full_buy1_amount
                    )
                    qty1 = int(target_amount // fill)
                    if qty1 <= 0:
                        continue
                    cost = fill * qty1
                    if daily_buy_amount + cost > cfg.daily_budget_krw:
                        continue

                    score = _safe_float(row.get("종합점수", 0))
                    rank = str(row.get("순위", ""))
                    r3 = _safe_float(row.get("최근3분수익률", 0))
                    r5 = _safe_float(row.get("최근5분수익률", 0))
                    vr = _safe_float(row.get("거래량배수", 0))
                    rs = _safe_float(row.get("상대강도", 0))
                    vg = _safe_float(row.get("VWAP괴리율", 0))
                    add_event(
                        now, symbol, "BUY1", "BUY", qty1, ref_price, fill,
                        (
                            f"AI V2 {action} · 전략 {score:.1f}점 · 3분 {r3:+.2f}% · "
                            f"5분 {r5:+.2f}% · 거래량 {vr:.2f}배 · RS {rs:+.2f}%p · "
                            f"VWAP {vg:+.2f}% · {','.join(audit_reasons[:3]) or 'NO_VETO'}"
                        ),
                        "", 0.0, score, rank, audit_action=action, ai_score=ai_score,
                    )
                    positions[symbol] = {
                        "qty": qty1,
                        "avg_price": fill,
                        "stage": 1,
                        "created_at": now.isoformat(),
                        "take1_sent": False,
                        "peak_pnl": 0.0,
                        "opened_at": now.isoformat(),
                        "audit_action": action,
                        "ai_score": ai_score,
                    }
                    daily_buy_amount += cost

        now += pd.Timedelta(seconds=int(cfg.manage_seconds))

    if positions:
        for symbol, pos in list(positions.items()):
            frame = target_frames.get(symbol)
            ref_price = _price_at(frame, date_text, end) if frame is not None else 0.0
            if ref_price <= 0:
                continue
            qty = int(pos.get("qty", 0))
            avg = float(pos.get("avg_price", 0) or 0)
            fill = _fill_price(cfg, "SELL", ref_price)
            pnl = (ref_price / avg - 1.0) * 100.0 if avg > 0 else 0.0
            realized = (fill - avg) * qty
            add_event(end, symbol, "FORCE_SELL_END", "SELL", qty, ref_price, fill,
                      "리플레이 종료 안전청산", pnl, realized,
                      audit_action=pos.get("audit_action", ""), ai_score=pos.get("ai_score", ""))
            positions.pop(symbol, None)

    _atomic_json(decision_path, decision_cache)
    df = pd.DataFrame(events)
    buy_amount = float(df.loc[df["구분"] == "BUY", "주문금액KRW"].sum()) if not df.empty else 0.0
    sell_amount = float(df.loc[df["구분"] == "SELL", "주문금액KRW"].sum()) if not df.empty else 0.0
    realized = float(df["실현손익KRW"].sum()) if not df.empty else 0.0
    episodes = _episode_summary(events)

    return {
        "summary": {
            "총주문횟수": int(len(df)),
            "매수주문횟수": int((df["구분"] == "BUY").sum()) if not df.empty else 0,
            "매도주문횟수": int((df["구분"] == "SELL").sum()) if not df.empty else 0,
            "거래횟수": len(episodes),
            "수익거래수": sum(1 for x in episodes if int(x.get("실현손익KRW", 0)) > 0),
            "손실거래수": sum(1 for x in episodes if int(x.get("실현손익KRW", 0)) < 0),
            "누적매수금액KRW": int(round(buy_amount)),
            "누적매도금액KRW": int(round(sell_amount)),
            "실현손익KRW": int(round(realized)),
            "일일예산1000만원대비수익률": round(realized / cfg.daily_budget_krw * 100.0, 3),
        },
        "events": events,
        "episodes": episodes,
        "decisions": decisions,
        "api_calls": api_calls,
        "ai_errors": ai_errors,
        "benchmark_available": bool(benchmarks),
        "data_available_count": len(target_frames),
        "universe_count": len(universe),
    }


def run_kr_ai_committee_replay_v2(
    date_text: str = "2026-08-10",
    codes: Iterable[str] | None = None,
    refresh: bool = False,
) -> dict:
    cache = _v2_cache_path(date_text)
    if not refresh and not codes and cache.exists():
        cached = _load_json(cache, None)
        if (
            isinstance(cached, dict)
            and cached.get("ok") is True
            and cached.get("version") == AI_REPLAY_V2_VERSION
        ):
            cached["cached"] = True
            return cached

    if not AI_CFG.api_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않아 AI V2 리플레이를 실행할 수 없습니다.")

    baseline = run_kr_trade_replay(
        date_text=date_text,
        codes=codes,
        config=KRReplayConfig(buy2_mode="NONE"),
        use_cache=False,
    )
    v2 = _simulate_ai_v2_auditor(date_text, codes, refresh_ai=refresh)

    v1_payload = _load_json(_cache_path(date_text), None)
    v1_pnl = None
    if isinstance(v1_payload, dict) and v1_payload.get("ok") is True:
        v1_pnl = int(
            ((v1_payload.get("comparison", {}) or {}).get("AI승인전략실현손익KRW", 0)) or 0
        )

    baseline_pnl = int((baseline.get("summary", {}) or {}).get("실현손익KRW", 0) or 0)
    v2_pnl = int((v2.get("summary", {}) or {}).get("실현손익KRW", 0) or 0)
    decisions = list(v2.get("decisions", []) or [])
    counts = {
        "ALLOW": sum(1 for x in decisions if x.get("audit_action") == "ALLOW"),
        "CAUTION": sum(1 for x in decisions if x.get("audit_action") == "CAUTION"),
        "VETO": sum(1 for x in decisions if x.get("audit_action") == "VETO"),
    }

    payload = {
        "ok": True,
        "version": AI_REPLAY_V2_VERSION,
        "date": date_text,
        "model": AI_CFG.model,
        "policy": {
            "baseline": "KR_C_NO_BUY2",
            "ai_role": "RISK_AUDITOR_NOT_GATEKEEPER",
            "ALLOW": "기존 C BUY1 그대로: 종목당 예산의 50%",
            "CAUTION": "매수 취소 대신 절반 크기: 종목당 예산의 25%",
            "VETO": "복합 약세/반등함정 등 명백한 위험일 때만 NO_BUY + 3분 재평가 제한",
            "buy2": "OFF",
            "future_data_visible_to_ai": False,
            "same_exit_rules": True,
            "actual_vwap_and_index_relative_strength": True,
            "fail_open": "AI 오류 시 C전략 매수 차단 안 함",
        },
        "comparison": {
            "기존C전략실현손익KRW": baseline_pnl,
            "AI_V1실현손익KRW": v1_pnl,
            "AI_V2실현손익KRW": v2_pnl,
            "V2대비C순효과KRW": v2_pnl - baseline_pnl,
            "V2대비V1순효과KRW": None if v1_pnl is None else v2_pnl - v1_pnl,
            "기존C전략수익률": (baseline.get("summary", {}) or {}).get("일일예산1000만원대비수익률", 0),
            "AI_V2수익률": (v2.get("summary", {}) or {}).get("일일예산1000만원대비수익률", 0),
        },
        "v2_decision_counts": counts,
        "api_calls_this_run": int(v2.get("api_calls", 0) or 0),
        "ai_errors": int(v2.get("ai_errors", 0) or 0),
        "benchmark_available": bool(v2.get("benchmark_available")),
        "baseline_summary": baseline.get("summary", {}),
        "v1_summary": (v1_payload.get("ai_summary", {}) if isinstance(v1_payload, dict) else {}),
        "v2_summary": v2.get("summary", {}),
        "v2_decisions": decisions,
        "v2_events": v2.get("events", []),
        "assumptions": {
            "real_orders": False,
            "data": "yfinance 1-minute historical bars",
            "candidate_reconstruction": "65개 고정 유동성 종목군 안에서 과거 후보 근사 복원",
            "benchmark": "KOSPI ^KS11 / KOSDAQ ^KQ11 1분봉 best-effort",
            "ai_information_rule": "각 판단시점까지의 데이터만 입력하며 이후 가격은 AI에 제공하지 않음",
            "slippage": "매수 +0.10%, 매도 -0.10% 가정",
            "fees_taxes": "별도 미포함",
            "important_limit": "과거 KIS 전체시장 실시간 거래량랭킹 원본이 없어 당시 전체시장 TOP5를 100% 복원한 것은 아님",
            "interpretation": "V2 설계 검증용 인샘플 실험이며 실전 수익을 보장하지 않음",
        },
        "cached": False,
    }
    if not codes:
        _atomic_json(cache, payload)
    return payload


def summarize_kr_ai_committee_replays_v2(
    dates: Iterable[str] | None = None,
) -> dict:
    dates = list(dates or DEFAULT_DATES)
    daily = []
    missing_v2 = []
    for date_text in dates:
        v2p = _load_json(_v2_cache_path(date_text), None)
        v1p = _load_json(_cache_path(date_text), None)
        if not isinstance(v2p, dict) or v2p.get("ok") is not True:
            missing_v2.append(date_text)
            continue
        c2 = v2p.get("comparison", {}) or {}
        v1_pnl = None
        if isinstance(v1p, dict) and v1p.get("ok") is True:
            v1_pnl = int(((v1p.get("comparison", {}) or {}).get("AI승인전략실현손익KRW", 0)) or 0)
        daily.append({
            "date": date_text,
            "기존C전략실현손익KRW": int(c2.get("기존C전략실현손익KRW", 0) or 0),
            "AI_V1실현손익KRW": v1_pnl,
            "AI_V2실현손익KRW": int(c2.get("AI_V2실현손익KRW", 0) or 0),
            "V2대비C순효과KRW": int(c2.get("V2대비C순효과KRW", 0) or 0),
            "V2대비V1순효과KRW": c2.get("V2대비V1순효과KRW"),
            "ALLOW": int((v2p.get("v2_decision_counts", {}) or {}).get("ALLOW", 0) or 0),
            "CAUTION": int((v2p.get("v2_decision_counts", {}) or {}).get("CAUTION", 0) or 0),
            "VETO": int((v2p.get("v2_decision_counts", {}) or {}).get("VETO", 0) or 0),
            "API호출": int(v2p.get("api_calls_this_run", 0) or 0),
        })

    c_total = sum(x["기존C전략실현손익KRW"] for x in daily)
    v2_total = sum(x["AI_V2실현손익KRW"] for x in daily)
    v1_values = [x["AI_V1실현손익KRW"] for x in daily if x["AI_V1실현손익KRW"] is not None]
    v1_total = sum(v1_values) if len(v1_values) == len(daily) and daily else None
    totals = {
        "기존C전략누적손익KRW": c_total,
        "AI_V1누적손익KRW": v1_total,
        "AI_V2누적손익KRW": v2_total,
        "V2대비C누적순효과KRW": v2_total - c_total,
        "V2대비V1누적순효과KRW": None if v1_total is None else v2_total - v1_total,
    }
    candidates = [("C", c_total), ("V2", v2_total)]
    if v1_total is not None:
        candidates.append(("V1", v1_total))
    winner = max(candidates, key=lambda x: x[1])[0] if candidates else ""

    return {
        "ok": len(daily) > 0,
        "version": "kr-ai-committee-replay-summary-v2",
        "requested_dates": dates,
        "completed_dates": [x["date"] for x in daily],
        "missing_v2_dates": missing_v2,
        "daily": daily,
        "total": totals,
        "winner_so_far": winner,
        "status": "V2 5일 결과 준비 완료" if not missing_v2 else "아직 실행하지 않은 V2 날짜가 있습니다.",
        "warning": "이번 5일은 V2 설계에 사용한 인샘플 기간입니다. 최종 판단은 다음 거래일부터 수정 없이 전진 모의테스트로 확인해야 합니다.",
    }

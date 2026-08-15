from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from ai_committee import (
    CFG as AI_CFG,
    _call_openai,
    _candidate_snapshot,
    _chair,
    _market_context,
    _safe_float,
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

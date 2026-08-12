from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from zoneinfo import ZoneInfo

from trader_core import parse_domestic_holdings, append_trade_log

STATE_DIR = Path(os.getenv("SONG_TRADER_STATE_DIR", "/tmp/song_trader"))
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "domestic_auto_state.json"

KST = ZoneInfo("Asia/Seoul")


@dataclass
class AutoConfig:
    daily_budget: int = 300_000
    per_stock_budget: int = 100_000
    max_positions: int = 3
    max_daily_orders: int = 12

    buy1_pct: int = 40
    buy2_pct: int = 30
    buy3_pct: int = 30

    add2_trigger_pct: float = 0.5
    add3_trigger_pct: float = 1.0

    stop_loss_pct: float = 3.0
    take1_pct: float = 3.0
    take2_pct: float = 5.0

    min_combined_score: float = 65.0
    require_green_signal: bool = True

    # 대장주 예외 진입:
    # TOP1 + 주도주점수 75 이상 + 종합점수 60 이상이면
    # 녹색 신호가 아니어도 1차 소액 진입만 허용합니다.
    leader_exception_enabled: bool = True
    leader_exception_min_lead_score: float = 75.0
    leader_exception_min_combined_score: float = 60.0

    last_entry_time: str = "14:50"
    force_exit_time: str = "15:15"


def _today() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d")


def _now() -> datetime:
    return datetime.now(KST)


def _clock(hhmm: str) -> dtime:
    h, m = [int(x) for x in hhmm.split(":")]
    return dtime(h, m)


def load_state() -> Dict[str,Any]:
    fresh = {
        "date": _today(),
        "positions": {},
        "daily_buy_amount": 0,
        "daily_orders": 0,
        "events": [],
    }
    if not STATE_FILE.exists():
        return fresh
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return fresh
    if state.get("date") != _today():
        return fresh
    state.setdefault("positions", {})
    state.setdefault("daily_buy_amount", 0)
    state.setdefault("daily_orders", 0)
    state.setdefault("events", [])
    return state


def save_state(state: Dict[str,Any]) -> None:
    STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def reset_today_state() -> None:
    if STATE_FILE.exists():
        STATE_FILE.unlink()


def _event(state, event, symbol="", detail=""):
    row = {
        "time": _now().isoformat(timespec="seconds"),
        "event": event,
        "symbol": symbol,
        "detail": str(detail)[:1000],
    }
    state["events"].append(row)
    state["events"] = state["events"][-200:]
    append_trade_log({
        "time": row["time"],
        "mode": "AUTO",
        "market": "KR",
        "symbol": symbol,
        "event": event,
        "detail": row["detail"],
    })


def _order_ok(res: Optional[Dict[str,Any]]) -> bool:
    return bool(res) and str(res.get("rt_cd", "")) == "0"


def _current_price(client, code: str) -> float:
    raw = client.domestic_price(code)
    out = (raw or {}).get("output", {}) or {}
    try:
        return float(out.get("stck_prpr", 0))
    except Exception:
        return 0.0


def _calc_qty(amount: int, price: float) -> int:
    if price <= 0:
        return 0
    return max(0, int(amount // price))


def _split_amounts(config: AutoConfig):
    total_pct = max(1, config.buy1_pct + config.buy2_pct + config.buy3_pct)
    return [
        int(config.per_stock_budget * config.buy1_pct / total_pct),
        int(config.per_stock_budget * config.buy2_pct / total_pct),
        int(config.per_stock_budget * config.buy3_pct / total_pct),
    ]


def _actual_holdings_map(client):
    raw = client.domestic_balance()
    df = parse_domestic_holdings(raw)
    if df.empty:
        return {}, df
    m = {}
    for _, r in df.iterrows():
        m[str(r["종목코드"]).zfill(6)] = {
            "name": r.get("종목명", ""),
            "qty": int(r.get("보유수량", 0)),
            "avg_price": float(r.get("평균매입가", 0)),
            "current_price": float(r.get("현재가", 0)),
        }
    return m, df


def _place_order(client, state, symbol, side, qty, reason, execute_orders):
    if qty <= 0:
        return {"status": "SKIP", "reason": "주문수량 0"}

    if state["daily_orders"] >= 9999:
        return {"status": "SKIP", "reason": "주문 제한"}

    if not execute_orders:
        _event(state, f"DRY_{side.upper()}", symbol, f"{qty}주 · {reason}")
        return {"status": "DRY", "qty": qty}

    try:
        res = client.domestic_order(symbol, qty, side, market_order=True)
    except Exception as e:
        _event(state, "ORDER_ERROR", symbol, repr(e))
        return {"status": "ERROR", "error": repr(e)}

    if _order_ok(res):
        state["daily_orders"] += 1
        _event(state, f"{side.upper()}_ORDER", symbol, f"{qty}주 · {reason} · {res}")
        return {"status": "ORDERED", "qty": qty, "response": res}

    _event(state, "ORDER_REJECT", symbol, str(res))
    return {"status": "REJECT", "response": res}


def run_domestic_cycle(
    client,
    leader_df: pd.DataFrame,
    config: AutoConfig,
    execute_orders: bool = False,
) -> Dict[str,Any]:
    """
    국내 자동매매 1회 사이클.

    안전 규칙:
    - 이 엔진이 오늘 추적한 종목만 자동매도
    - 기존 보유주식은 건드리지 않음
    - 14:50 이후 신규매수 금지
    - 15:15 이후 추적 포지션 전량 청산 시도
    - 손절/익절 우선 처리
    - 매수는 대장주 TOP5 중 기술판정 녹색 + 종합점수 기준 통과 종목만
    """
    state = load_state()
    now = _now()
    result = {
        "time": now.isoformat(timespec="seconds"),
        "execute_orders": execute_orders,
        "actions": [],
        "state": state,
    }

    if now.weekday() >= 5:
        result["message"] = "주말: 주문 없음"
        save_state(state)
        return result

    market_open = dtime(9, 0) <= now.time() < dtime(15, 30)
    if not market_open:
        result["message"] = "장외: 주문 없음"
        save_state(state)
        return result

    holdings, holdings_df = _actual_holdings_map(client)
    result["holdings"] = holdings_df.to_dict("records")

    # 1) 기존 봇 추적 포지션 관리
    for symbol, pos in list(state["positions"].items()):
        actual = holdings.get(symbol, {})
        actual_qty = int(actual.get("qty", 0))

        # 주문 후 실제 잔고가 확인되면 수량/평단 동기화
        if actual_qty > 0:
            pos["actual_qty"] = actual_qty
            if actual.get("avg_price", 0) > 0:
                pos["avg_price"] = float(actual["avg_price"])

        if actual_qty <= 0:
            # 아직 첫 주문 직후일 수 있으므로 오래된 포지션만 종료 처리
            created = pos.get("created_at", "")
            try:
                age = now - datetime.fromisoformat(created)
            except Exception:
                age = timedelta(minutes=10)
            if age > timedelta(minutes=5):
                pos["closed"] = True
                state["positions"].pop(symbol, None)
            continue

        price = _current_price(client, symbol)
        if price <= 0:
            continue

        avg = float(pos.get("avg_price") or actual.get("avg_price") or price)
        pnl = (price / avg - 1) * 100 if avg > 0 else 0.0

        # 강제 청산
        if now.time() >= _clock(config.force_exit_time):
            act = _place_order(
                client, state, symbol, "sell", actual_qty,
                f"당일 강제청산 {config.force_exit_time}", execute_orders
            )
            result["actions"].append({"symbol": symbol, "action": "FORCE_SELL", "pnl": pnl, **act})
            if act["status"] in ("ORDERED", "DRY"):
                pos["force_exit_sent"] = True
            continue

        # 손절
        if pnl <= -abs(config.stop_loss_pct):
            act = _place_order(
                client, state, symbol, "sell", actual_qty,
                f"손절 {pnl:.2f}%", execute_orders
            )
            result["actions"].append({"symbol": symbol, "action": "STOP_LOSS", "pnl": pnl, **act})
            continue

        # 2차 익절: 남은 수량 전량
        if pnl >= config.take2_pct:
            act = _place_order(
                client, state, symbol, "sell", actual_qty,
                f"2차 익절 {pnl:.2f}%", execute_orders
            )
            result["actions"].append({"symbol": symbol, "action": "TAKE2", "pnl": pnl, **act})
            if act["status"] in ("ORDERED", "DRY"):
                pos["take2_sent"] = True
            continue

        # 1차 익절: 절반
        if pnl >= config.take1_pct and not pos.get("take1_sent"):
            sell_qty = max(1, actual_qty // 2)
            act = _place_order(
                client, state, symbol, "sell", sell_qty,
                f"1차 익절 {pnl:.2f}%", execute_orders
            )
            result["actions"].append({"symbol": symbol, "action": "TAKE1", "pnl": pnl, **act})
            if act["status"] in ("ORDERED", "DRY"):
                pos["take1_sent"] = True
            continue

        # 추가 분할매수: 상승 확인 후 피라미딩
        # 대장주 예외 진입은 위험을 제한하기 위해 1차 매수만 허용
        if pos.get("leader_exception"):
            continue

        parts = _split_amounts(config)
        stage = int(pos.get("buy_stage", 1))
        if stage == 1 and pnl >= config.add2_trigger_pct:
            qty = _calc_qty(parts[1], price)
            if qty > 0 and state["daily_buy_amount"] + qty * price <= config.daily_budget:
                act = _place_order(
                    client, state, symbol, "buy", qty,
                    f"2차 분할매수 +{pnl:.2f}%", execute_orders
                )
                result["actions"].append({"symbol": symbol, "action": "BUY2", "pnl": pnl, **act})
                if act["status"] in ("ORDERED", "DRY"):
                    pos["buy_stage"] = 2
                    state["daily_buy_amount"] += int(qty * price)
                continue

        if stage == 2 and pnl >= config.add3_trigger_pct:
            qty = _calc_qty(parts[2], price)
            if qty > 0 and state["daily_buy_amount"] + qty * price <= config.daily_budget:
                act = _place_order(
                    client, state, symbol, "buy", qty,
                    f"3차 분할매수 +{pnl:.2f}%", execute_orders
                )
                result["actions"].append({"symbol": symbol, "action": "BUY3", "pnl": pnl, **act})
                if act["status"] in ("ORDERED", "DRY"):
                    pos["buy_stage"] = 3
                    state["daily_buy_amount"] += int(qty * price)

    # 마감 전 신규진입 차단
    if now.time() >= _clock(config.last_entry_time):
        result["message"] = f"{config.last_entry_time} 이후 신규매수 금지"
        save_state(state)
        result["state"] = state
        return result

    # 2) 신규 진입
    open_positions = len(state["positions"])
    if open_positions >= config.max_positions:
        result["message"] = "최대 보유종목 수 도달"
        save_state(state)
        result["state"] = state
        return result

    if leader_df is None or leader_df.empty:
        result["message"] = "대장주 TOP5 데이터 없음"
        save_state(state)
        result["state"] = state
        return result

    parts = _split_amounts(config)

    for _, row in leader_df.iterrows():
        if len(state["positions"]) >= config.max_positions:
            break
        symbol = str(row.get("종목코드", "")).zfill(6)
        if not symbol or symbol in state["positions"]:
            continue

        signal = str(row.get("판정", ""))
        combined = float(row.get("종합점수", 0) or 0)
        lead_score = float(row.get("주도주점수", 0) or 0)

        rank_text = str(row.get("순위", "")).strip()
        is_rank1 = ("1위" in rank_text) or (rank_text in ("1", "1.0"))

        normal_entry = (
            (not config.require_green_signal or "매수 후보" in signal)
            and combined >= config.min_combined_score
        )

        leader_exception = (
            bool(getattr(config, "leader_exception_enabled", True))
            and is_rank1
            and lead_score >= float(
                getattr(config, "leader_exception_min_lead_score", 75.0)
            )
            and combined >= float(
                getattr(config, "leader_exception_min_combined_score", 60.0)
            )
        )

        if not (normal_entry or leader_exception):
            continue

        if symbol in holdings:
            # 사용자가 기존에 들고 있던 종목은 자동매매 신규진입에서 제외
            continue

        price = _current_price(client, symbol)
        if price <= 0:
            continue

        qty = _calc_qty(parts[0], price)
        if qty <= 0:
            result["actions"].append({
                "symbol": symbol, "action": "SKIP",
                "reason": f"1차 예산 {parts[0]:,}원보다 주가가 높음"
            })
            continue

        cost = int(qty * price)
        if state["daily_buy_amount"] + cost > config.daily_budget:
            continue
        if state["daily_orders"] >= config.max_daily_orders:
            break

        entry_reason = (
            f"대장주 예외 1차매수 · TOP1 · 주도주점수 {lead_score:.1f} · "
            f"종합점수 {combined:.1f} · {signal}"
            if leader_exception and not normal_entry
            else f"1차 분할매수 · 종합점수 {combined:.1f} · {signal}"
        )

        act = _place_order(
            client, state, symbol, "buy", qty,
            entry_reason,
            execute_orders
        )
        result["actions"].append({
            "symbol": symbol,
            "action": "BUY1_EXCEPTION" if leader_exception and not normal_entry else "BUY1",
            "reason": entry_reason,
            **act
        })

        if act["status"] in ("ORDERED", "DRY"):
            state["positions"][symbol] = {
                "name": str(row.get("종목명", "")),
                "created_at": now.isoformat(),
                "buy_stage": 1,
                "avg_price": price,
                "expected_qty": qty,
                "leader_exception": bool(leader_exception and not normal_entry),
                "take1_sent": False,
                "take2_sent": False,
                "force_exit_sent": False,
            }
            state["daily_buy_amount"] += cost

    if not result["actions"]:
        result["message"] = "조건을 통과한 신규 매수 후보 없음"

    save_state(state)
    result["state"] = state
    return result

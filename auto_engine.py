from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from zoneinfo import ZoneInfo

from trader_core import append_trade_log

STATE_DIR = Path(os.getenv("SONG_TRADER_STATE_DIR", "/tmp/song_trader"))
STATE_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = STATE_DIR / "domestic_auto_state.json"
ORDER_LOCK_FILE = STATE_DIR / "domestic_order_lock.json"
US_STATE_FILE = STATE_DIR / "overseas_auto_state.json"

KST = ZoneInfo("Asia/Seoul")
ET = ZoneInfo("America/New_York")


@dataclass
class AutoConfig:
    daily_budget: int = 10_000_000
    per_stock_budget: int = 10_000_000
    max_positions: int = 3
    max_daily_orders: int = 12

    buy1_pct: int = 50
    buy2_pct: int = 30
    buy3_pct: int = 20

    add2_trigger_pct: float = 0.5
    add3_trigger_pct: float = 1.0

    stop_loss_pct: float = 3.0
    take1_pct: float = 3.0
    take2_pct: float = 5.0

    min_combined_score: float = 65.0
    require_green_signal: bool = True

    demo_relaxed_entry_enabled: bool = True
    demo_min_combined_score: float = 40.0

    leader_exception_enabled: bool = True
    leader_exception_min_lead_score: float = 75.0
    leader_exception_min_combined_score: float = 60.0

    last_entry_time: str = "16:00"
    force_exit_time: str = "15:15"
    duplicate_guard_seconds: int = 90
    buying_power_buffer_pct: float = 5.0
    min_order_amount: int = 10_000

    us_daily_budget_usd: float = 1500.0
    us_per_stock_budget_usd: float = 600.0
    us_last_entry_time: str = "15:30"
    us_force_exit_time: str = "15:50"


def _clock(hhmm: str) -> dtime:
    h, m = [int(x) for x in hhmm.split(":")]
    return dtime(h, m)


def _today_kst() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d")


def _now_kst() -> datetime:
    return datetime.now(KST)


def _order_ok(res: Optional[Dict[str, Any]]) -> bool:
    return bool(res) and str(res.get("rt_cd", "")) == "0"


def load_state() -> Dict[str, Any]:
    fresh = {
        "date": _today_kst(),
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

    if state.get("date") != _today_kst():
        return fresh

    state.setdefault("positions", {})
    state.setdefault("daily_buy_amount", 0)
    state.setdefault("daily_orders", 0)
    state.setdefault("events", [])
    return state


def save_state(state: Dict[str, Any]) -> None:
    STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def reset_today_state() -> None:
    for p in (STATE_FILE, ORDER_LOCK_FILE):
        if p.exists():
            p.unlink()


def _event(state, event, symbol="", detail=""):
    row = {
        "time": _now_kst().isoformat(timespec="seconds"),
        "event": event,
        "symbol": symbol,
        "detail": str(detail)[:1000],
    }

    state["events"].append(row)
    state["events"] = state["events"][-300:]

    append_trade_log({
        "time": row["time"],
        "mode": "AUTO",
        "market": "KR",
        "symbol": symbol,
        "event": event,
        "detail": row["detail"],
    })


def _diag(result, symbol, action, reason, **extra):
    row = {
        "symbol": symbol,
        "action": action,
        "reason": reason,
        **extra,
    }
    result.setdefault("diagnostics", []).append(row)
    return row

def _read_order_lock() -> Dict[str, Any]:
    if not ORDER_LOCK_FILE.exists():
        return {}
    try:
        return json.loads(ORDER_LOCK_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_order_lock(symbol: str, side: str, source: str) -> None:
    try:
        ORDER_LOCK_FILE.write_text(
            json.dumps({
                "time": _now_kst().isoformat(),
                "symbol": symbol,
                "side": side,
                "source": source,
            }, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass


def _duplicate_guard_hit(symbol: str, side: str, seconds: int):
    lock = _read_order_lock()
    if not lock:
        return False, ""
    if str(lock.get("symbol")) != symbol or str(lock.get("side")) != side:
        return False, ""
    try:
        age = (_now_kst() - datetime.fromisoformat(lock.get("time", ""))).total_seconds()
    except Exception:
        return False, ""
    if age < seconds:
        return True, (
            f"중복주문 방지: {age:.0f}초 전에 "
            f"{lock.get('source', 'unknown')}에서 같은 주문 시도"
        )
    return False, ""



def _current_price(client, code: str) -> tuple[float, str]:
    try:
        raw = client.domestic_price(code)
    except Exception as e:
        return 0.0, f"KIS 현재가 조회 예외: {type(e).__name__}: {e}"

    out = (raw or {}).get("output", {}) or {}

    try:
        price = float(out.get("stck_prpr", 0) or 0)
    except Exception:
        price = 0.0

    if price <= 0:
        return 0.0, f"KIS 현재가 0 또는 응답 이상: {raw}"

    return price, ""


def _calc_qty(amount: float, price: float) -> int:
    if price <= 0:
        return 0
    return max(0, int(amount // price))


def _split_amounts(config: AutoConfig) -> list[int]:
    total_pct = max(1, config.buy1_pct + config.buy2_pct + config.buy3_pct)
    return [
        int(config.per_stock_budget * config.buy1_pct / total_pct),
        int(config.per_stock_budget * config.buy2_pct / total_pct),
        int(config.per_stock_budget * config.buy3_pct / total_pct),
    ]


def _parse_domestic_holdings(balance_json: Dict[str, Any]) -> pd.DataFrame:
    rows = (balance_json or {}).get("output1", []) or []

    if not rows:
        return pd.DataFrame(
            columns=["종목코드", "종목명", "보유수량", "평균매입가", "현재가"]
        )

    df = pd.DataFrame(rows)

    def first_existing(*names):
        for n in names:
            if n in df.columns:
                return n
        return None

    code = first_existing("pdno", "mksc_shrn_iscd")
    name = first_existing("prdt_name", "hts_kor_isnm")
    qty = first_existing("hldg_qty", "hold_qty")
    avg = first_existing("pchs_avg_pric", "avg_pric")
    cur = first_existing("prpr", "stck_prpr")

    out = pd.DataFrame(index=df.index)
    out["종목코드"] = df[code].astype(str).str.zfill(6) if code else ""
    out["종목명"] = df[name].astype(str) if name else ""
    out["보유수량"] = pd.to_numeric(df[qty], errors="coerce").fillna(0).astype(int) if qty else 0
    out["평균매입가"] = pd.to_numeric(df[avg], errors="coerce").fillna(0.0) if avg else 0.0
    out["현재가"] = pd.to_numeric(df[cur], errors="coerce").fillna(0.0) if cur else 0.0

    return out[out["보유수량"] > 0].reset_index(drop=True)


def _actual_holdings_map(client):
    try:
        raw = client.domestic_balance()
        df = _parse_domestic_holdings(raw)
    except Exception as e:
        return {}, pd.DataFrame(), f"잔고조회 실패: {type(e).__name__}: {e}"

    result = {}

    for _, r in df.iterrows():
        result[str(r["종목코드"]).zfill(6)] = {
            "name": r.get("종목명", ""),
            "qty": int(r.get("보유수량", 0)),
            "avg_price": float(r.get("평균매입가", 0)),
            "current_price": float(r.get("현재가", 0)),
        }

    return result, df, ""



def _safe_buy_qty_from_buying_power(
    client,
    symbol: str,
    target_amount: int,
    current_price: float,
    config: AutoConfig,
) -> tuple[int, Dict[str, Any], str]:
    """
    KIS 매수가능조회 후 주문수량을 자동 축소합니다.
    미수 없는 매수가능금액/수량을 우선 사용합니다.
    """
    if current_price <= 0:
        return 0, {}, "현재가가 0 이하"

    try:
        raw = client.domestic_buying_power(
            symbol=symbol,
            reference_price=int(current_price),
        )
    except Exception as e:
        return 0, {}, f"매수가능조회 실패: {type(e).__name__}: {e}"

    output = (raw or {}).get("output", {}) or {}

    def _to_int(v):
        try:
            return int(float(str(v).replace(",", "").strip() or "0"))
        except Exception:
            return 0

    nrcvb_amt = _to_int(output.get("nrcvb_buy_amt", 0))
    nrcvb_qty = _to_int(output.get("nrcvb_buy_qty", 0))
    ord_cash = _to_int(output.get("ord_psbl_cash", 0))
    max_amt = _to_int(output.get("max_buy_amt", 0))
    max_qty = _to_int(output.get("max_buy_qty", 0))

    available_amt = nrcvb_amt if nrcvb_amt > 0 else ord_cash
    if available_amt <= 0:
        available_amt = max_amt

    available_qty = nrcvb_qty if nrcvb_qty > 0 else max_qty

    buffer_ratio = max(
        0.0,
        min(0.50, float(config.buying_power_buffer_pct) / 100.0),
    )
    buffered_amt = int(available_amt * (1.0 - buffer_ratio))

    usable_amount = min(
        int(target_amount),
        max(0, buffered_amt),
    )

    qty = int(usable_amount // current_price)

    if available_qty > 0:
        qty = min(qty, available_qty)

    meta = {
        "target_amount": int(target_amount),
        "nrcvb_buy_amt": nrcvb_amt,
        "nrcvb_buy_qty": nrcvb_qty,
        "ord_psbl_cash": ord_cash,
        "max_buy_amt": max_amt,
        "max_buy_qty": max_qty,
        "buffer_pct": float(config.buying_power_buffer_pct),
        "buffered_available_amt": buffered_amt,
        "usable_amount": usable_amount,
    }

    if usable_amount < int(config.min_order_amount):
        return 0, meta, (
            f"주문가능금액 부족: 안전여유 적용 후 {usable_amount:,}원 "
            f"< 최소주문금액 {config.min_order_amount:,}원"
        )

    if qty <= 0:
        return 0, meta, (
            f"주문가능수량 0: 사용가능금액 {usable_amount:,}원 / "
            f"현재가 {current_price:,.0f}원"
        )

    if usable_amount < int(target_amount):
        reason = (
            f"주문금액 자동축소: 목표 {target_amount:,}원 → "
            f"사용가능 {usable_amount:,}원 / {qty}주"
        )
    else:
        reason = (
            f"주문가능금액 확인 완료: {usable_amount:,}원 / {qty}주"
        )

    return qty, meta, reason

def _place_order(client, state, symbol, side, qty, reason, execute_orders, source="APP", duplicate_guard_seconds=90):
    if qty <= 0:
        return {
            "status": "SKIP",
            "reason": "주문수량 0",
        }

    if not execute_orders:
        # 중요: DRY에서는 상태/누적매수금액을 변경하지 않음
        return {
            "status": "DRY",
            "qty": qty,
            "reason": reason,
        }

    hit, why = _duplicate_guard_hit(symbol, side, duplicate_guard_seconds)
    if hit:
        return {
            "status": "DUPLICATE_BLOCKED",
            "qty": qty,
            "reason": why,
        }

    _write_order_lock(symbol, side, source)

    try:
        res = client.domestic_order(
            symbol,
            qty,
            side,
            market_order=True,
        )
    except Exception as e:
        _event(state, "ORDER_ERROR", symbol, repr(e))
        return {
            "status": "ERROR",
            "error": repr(e),
        }

    if _order_ok(res):
        state["daily_orders"] += 1
        _event(
            state,
            f"{side.upper()}_ORDER",
            symbol,
            f"{qty}주 · {reason} · {res}",
        )
        return {
            "status": "ORDERED",
            "qty": qty,
            "response": res,
        }

    msg_cd = res.get("msg_cd", "") if isinstance(res, dict) else ""
    msg1 = res.get("msg1", "") if isinstance(res, dict) else ""

    _event(
        state,
        "ORDER_REJECT",
        symbol,
        f"{msg_cd} {msg1} {res}",
    )

    return {
        "status": "REJECT",
        "qty": qty,
        "msg_cd": msg_cd,
        "msg1": msg1,
        "response": res,
    }


def run_domestic_cycle(
    client,
    leader_df: pd.DataFrame,
    config: AutoConfig,
    execute_orders: bool = False,
    source: str = "APP",
) -> Dict[str, Any]:
    state = load_state()
    now = _now_kst()

    result = {
        "time": now.isoformat(timespec="seconds"),
        "execute_orders": bool(execute_orders),
        "source": source,
        "actions": [],
        "diagnostics": [],
        "state": state,
    }

    if now.weekday() >= 5:
        result["message"] = "주말: 주문 없음"
        save_state(state)
        return result

    if not (dtime(8, 30) <= now.time() < dtime(16, 0)):
        result["message"] = "국내 운영시간 외(08:30~16:00)"
        save_state(state)
        return result

    if execute_orders and not (dtime(9, 0) <= now.time() < dtime(15, 30)):
        execute_orders = False
        result["execute_orders"] = False
        result["order_gate_message"] = (
            "분석은 가능하지만 실제 주문전송은 09:00~15:30에만 허용됩니다."
        )

    holdings, holdings_df, balance_warning = _actual_holdings_map(client)
    result["holdings"] = holdings_df.to_dict("records") if not holdings_df.empty else []

    if balance_warning:
        result["balance_warning"] = balance_warning

    # ---------------------------------------------------------
    # 기존 보유종목 관리
    # ---------------------------------------------------------
    for symbol, pos in list(state["positions"].items()):
        actual = holdings.get(symbol, {})
        actual_qty = int(actual.get("qty", 0))

        # 실제 주문 모드에서는 체결 확인 전 5분 유예
        if actual_qty <= 0 and execute_orders:
            created = pos.get("created_at", "")
            try:
                age = now - datetime.fromisoformat(created)
            except Exception:
                age = timedelta(minutes=10)

            if age > timedelta(minutes=5):
                _diag(
                    result,
                    symbol,
                    "DROP_TRACKING",
                    "실제 잔고에 보유수량이 없어 추적 상태 제거",
                )
                state["positions"].pop(symbol, None)
            continue

        if actual_qty <= 0:
            continue

        pos["actual_qty"] = actual_qty

        if actual.get("avg_price", 0) > 0:
            pos["avg_price"] = float(actual["avg_price"])

        price, err = _current_price(client, symbol)
        if price <= 0:
            _diag(result, symbol, "SKIP_MANAGE_PRICE", err)
            continue

        avg = float(pos.get("avg_price") or actual.get("avg_price") or price)
        pnl = (price / avg - 1) * 100 if avg > 0 else 0.0

        if now.time() >= _clock(config.force_exit_time):
            act = _place_order(
                client,
                state,
                symbol,
                "sell",
                actual_qty,
                f"당일 강제청산 {config.force_exit_time}",
                execute_orders,
                source,
                config.duplicate_guard_seconds,
            )
            result["actions"].append({
                "symbol": symbol,
                "action": "FORCE_SELL",
                "pnl": round(pnl, 3),
                **act,
            })
            continue

        if pnl <= -abs(config.stop_loss_pct):
            act = _place_order(
                client,
                state,
                symbol,
                "sell",
                actual_qty,
                f"손절 {pnl:.2f}%",
                execute_orders,
                source,
                config.duplicate_guard_seconds,
            )
            result["actions"].append({
                "symbol": symbol,
                "action": "STOP_LOSS",
                "pnl": round(pnl, 3),
                **act,
            })
            continue

        if pnl >= config.take2_pct:
            act = _place_order(
                client,
                state,
                symbol,
                "sell",
                actual_qty,
                f"2차 익절 {pnl:.2f}%",
                execute_orders,
                source,
                config.duplicate_guard_seconds,
            )
            result["actions"].append({
                "symbol": symbol,
                "action": "TAKE2",
                "pnl": round(pnl, 3),
                **act,
            })
            continue

        if pnl >= config.take1_pct and not pos.get("take1_sent"):
            sell_qty = max(1, actual_qty // 2)

            act = _place_order(
                client,
                state,
                symbol,
                "sell",
                sell_qty,
                f"1차 익절 {pnl:.2f}%",
                execute_orders,
                source,
                config.duplicate_guard_seconds,
            )

            result["actions"].append({
                "symbol": symbol,
                "action": "TAKE1",
                "pnl": round(pnl, 3),
                **act,
            })

            if act.get("status") == "ORDERED":
                pos["take1_sent"] = True
            continue

    # ---------------------------------------------------------
    # 신규 진입
    # ---------------------------------------------------------
    if now.time() >= _clock(config.last_entry_time):
        result["message"] = f"{config.last_entry_time} 이후 신규매수 금지"
        save_state(state)
        result["state"] = state
        return result

    if leader_df is None or leader_df.empty:
        result["message"] = "대장주 TOP5 데이터 없음"
        save_state(state)
        result["state"] = state
        return result

    parts = _split_amounts(config)
    is_demo = str(getattr(client, "env", "demo")).lower() == "demo"

    for _, row in leader_df.head(5).iterrows():
        symbol = str(row.get("종목코드", "")).zfill(6)

        if not (len(symbol) == 6 and symbol.isdigit()):
            _diag(
                result,
                symbol,
                "SKIP_BAD_SYMBOL",
                "종목코드가 6자리 숫자가 아님",
            )
            continue

        if len(state["positions"]) >= config.max_positions:
            _diag(
                result,
                symbol,
                "SKIP_MAX_POSITIONS",
                f"최대 보유종목 {config.max_positions}개 도달",
            )
            continue

        if symbol in state["positions"]:
            _diag(
                result,
                symbol,
                "SKIP_ALREADY_TRACKED",
                "이미 자동매매 추적중인 종목",
            )
            continue

        if symbol in holdings:
            _diag(
                result,
                symbol,
                "SKIP_ALREADY_HELD",
                f"실제 모의계좌 보유중: {holdings[symbol].get('qty', 0)}주",
            )
            continue

        signal = str(row.get("판정", ""))
        combined = float(row.get("종합점수", 0) or 0)
        lead_score = float(row.get("주도주점수", 0) or 0)

        rank_text = str(row.get("순위", "")).strip()
        is_rank1 = ("1위" in rank_text) or rank_text in ("1", "1.0")

        normal_entry = (
            (not config.require_green_signal or "매수 후보" in signal)
            and combined >= config.min_combined_score
        )

        leader_exception = (
            bool(config.leader_exception_enabled)
            and is_rank1
            and lead_score >= config.leader_exception_min_lead_score
            and combined >= config.leader_exception_min_combined_score
        )

        demo_relaxed_entry = (
            is_demo
            and bool(config.demo_relaxed_entry_enabled)
            and combined >= config.demo_min_combined_score
        )

        if not (normal_entry or leader_exception or demo_relaxed_entry):
            _diag(
                result,
                symbol,
                "SKIP_SCORE",
                (
                    f"점수 미달: 종합 {combined:.1f}, 주도 {lead_score:.1f}, "
                    f"신호 {signal}, demo기준 {config.demo_min_combined_score:.1f}"
                ),
            )
            continue

        price, price_err = _current_price(client, symbol)

        if price <= 0:
            _diag(
                result,
                symbol,
                "SKIP_PRICE_LOOKUP",
                price_err,
                combined=combined,
                lead_score=lead_score,
            )
            continue

        qty, buying_power, buying_power_reason = (
            _safe_buy_qty_from_buying_power(
                client=client,
                symbol=symbol,
                target_amount=parts[0],
                current_price=price,
                config=config,
            )
        )

        if qty <= 0:
            _diag(
                result,
                symbol,
                "SKIP_BUYING_POWER",
                buying_power_reason,
                buying_power=buying_power,
            )
            continue

        cost = int(qty * price)

        if state["daily_buy_amount"] + cost > config.daily_budget:
            _diag(
                result,
                symbol,
                "SKIP_DAILY_BUDGET",
                (
                    f"현재 누적 {state['daily_buy_amount']:,} + "
                    f"예상 {cost:,} > 일일한도 {config.daily_budget:,}"
                ),
            )
            continue

        if state["daily_orders"] >= config.max_daily_orders:
            _diag(
                result,
                symbol,
                "SKIP_DAILY_ORDERS",
                f"일일 주문횟수 {config.max_daily_orders}회 도달",
            )
            continue

        if normal_entry:
            entry_reason = f"정상 1차매수 · 종합 {combined:.1f} · {signal}"
            action_name = "BUY1"
            relaxed_flag = False
            leader_exception_flag = False
        elif leader_exception:
            entry_reason = (
                f"대장주 예외 1차매수 · TOP1 · 주도 {lead_score:.1f} · "
                f"종합 {combined:.1f} · {signal}"
            )
            action_name = "BUY1_EXCEPTION"
            relaxed_flag = False
            leader_exception_flag = True
        else:
            entry_reason = (
                f"모의완화 1차매수 · TOP5 · 종합 {combined:.1f} · {signal}"
            )
            action_name = "BUY1_DEMO_RELAXED"
            relaxed_flag = True
            leader_exception_flag = False

        act = _place_order(
            client,
            state,
            symbol,
            "buy",
            qty,
            entry_reason + " · " + buying_power_reason,
            execute_orders,
            source,
            config.duplicate_guard_seconds,
        )

        result["actions"].append({
            "symbol": symbol,
            "action": action_name,
            "current_price": price,
            "estimated_cost": cost,
            "combined_score": combined,
            "lead_score": lead_score,
            "buying_power_reason": buying_power_reason,
            "buying_power": buying_power,
            **act,
        })

        # 중요: DRY는 상태를 건드리지 않습니다.
        if act.get("status") == "ORDERED":
            state["positions"][symbol] = {
                "name": str(row.get("종목명", "")),
                "created_at": now.isoformat(),
                "buy_stage": 1,
                "avg_price": price,
                "expected_qty": qty,
                "leader_exception": leader_exception_flag,
                "demo_relaxed_entry": relaxed_flag,
                "take1_sent": False,
            }
            state["daily_buy_amount"] += cost

    if not result["actions"]:
        result["message"] = (
            "실제 주문/DRY 액션 없음. 아래 진단표에서 종목별 SKIP 이유를 확인하세요."
        )

    save_state(state)
    result["state"] = state
    return result


# 미국 함수 이름 호환용 - 기존 미국 로직은 이번 수정 대상 아님
def run_overseas_cycle(
    client,
    leader_df: pd.DataFrame,
    config: AutoConfig,
    execute_orders: bool = False,
) -> Dict[str, Any]:
    return {
        "time": datetime.now(ET).isoformat(timespec="seconds"),
        "execute_orders": execute_orders,
        "actions": [],
        "message": "이번 교체본은 국내 오류진단/안정화 우선 버전입니다. 미국 로직은 기존 파일을 유지하세요.",
        "state": {},
    }

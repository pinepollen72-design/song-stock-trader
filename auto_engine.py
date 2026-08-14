from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from zoneinfo import ZoneInfo

from trader_core import (
    append_trade_log,
    merge_overseas_holdings,
)

STATE_DIR = Path(os.getenv("SONG_TRADER_STATE_DIR", "/tmp/song_trader"))
STATE_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = STATE_DIR / "domestic_auto_state.json"
US_STATE_FILE = STATE_DIR / "overseas_auto_state.json"
ORDER_LOCK_FILE = STATE_DIR / "domestic_order_lock.json"

KST = ZoneInfo("Asia/Seoul")
ET = ZoneInfo("America/New_York")


@dataclass
class AutoConfig:
    # Worker에서 사용하는 이름
    kr_daily_budget: int = 10_000_000
    kr_per_stock_budget: int = 3_000_000
    us_daily_budget_usd: float = 5_000.0
    us_per_stock_budget_usd: float = 1_500.0

    max_positions: int = 3
    max_daily_orders: int = 12

    min_score: float = 50.0
    stop_loss_pct: float = 3.0
    take1_pct: float = 3.0
    take2_pct: float = 5.0

    kr_last_entry_time: str = "14:50"
    kr_force_exit_time: str = "15:15"
    us_last_entry_time: str = "15:30"
    us_force_exit_time: str = "15:50"

    buying_power_buffer_pct: float = 5.0
    confirm_wait_seconds: int = 8
    force_exit_all_demo_holdings: bool = True

    # 국내 매수는 2회 50:50
    daily_budget: int = 10_000_000
    per_stock_budget: int = 3_000_000
    buy1_pct: int = 50
    buy2_pct: int = 50

    add2_trigger_pct: float = 0.5

    min_combined_score: float = 50.0
    require_green_signal: bool = True

    demo_relaxed_entry_enabled: bool = True
    demo_min_combined_score: float = 40.0

    leader_exception_enabled: bool = True
    leader_exception_min_lead_score: float = 75.0
    leader_exception_min_combined_score: float = 60.0

    last_entry_time: str = "14:50"
    force_exit_time: str = "15:15"

    duplicate_guard_seconds: int = 90
    pending_timeout_seconds: int = 300
    min_order_amount: int = 10_000
    adopt_existing_demo_top5: bool = True

    allow_single_share_over_stage_budget: bool = True

    def __post_init__(self):
        self.daily_budget = int(self.kr_daily_budget)
        self.per_stock_budget = int(self.kr_per_stock_budget)
        self.min_combined_score = float(self.min_score)
        self.last_entry_time = str(self.kr_last_entry_time)
        self.force_exit_time = str(self.kr_force_exit_time)


def _clock(hhmm: str) -> dtime:
    h, m = [int(x) for x in hhmm.split(":")]
    return dtime(h, m)


def _today_kst() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d")


def _today_et() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d")


def _now_kst() -> datetime:
    return datetime.now(KST)


def _now_et() -> datetime:
    return datetime.now(ET)


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
        json.dumps(state, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def reset_today_state() -> None:
    for p in (STATE_FILE, ORDER_LOCK_FILE):
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass


def _event(state, event, symbol="", detail=""):
    row = {
        "time": _now_kst().isoformat(timespec="seconds"),
        "event": event,
        "symbol": symbol,
        "detail": str(detail)[:1500],
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
    row = {"symbol": symbol, "action": action, "reason": reason, **extra}
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
        return 0.0, f"KIS 현재가 0 또는 응답 이상: {str(raw)[:500]}"
    return price, ""


def _calc_qty(amount: float, price: float) -> int:
    if price <= 0:
        return 0
    return max(0, int(amount // price))


def _split_amounts(config: AutoConfig) -> list[int]:
    total_pct = max(1, int(config.buy1_pct) + int(config.buy2_pct))
    first = int(config.per_stock_budget * int(config.buy1_pct) / total_pct)
    second = int(config.per_stock_budget) - first
    return [first, second]


def _parse_domestic_holdings(balance_json: Dict[str, Any]) -> pd.DataFrame:
    rows = (balance_json or {}).get("output1", []) or []
    if not rows:
        return pd.DataFrame(columns=[
            "종목코드", "종목명", "보유수량", "매도가능수량", "평균매입가", "현재가"
        ])
    df = pd.DataFrame(rows)

    def first_existing(*names):
        for name in names:
            if name in df.columns:
                return name
        return None

    code = first_existing("pdno", "mksc_shrn_iscd")
    name = first_existing("prdt_name", "hts_kor_isnm")
    qty = first_existing("hldg_qty", "hold_qty")
    sellable = first_existing("ord_psbl_qty", "sell_psbl_qty", "sll_psbl_qty")
    avg = first_existing("pchs_avg_pric", "avg_pric")
    cur = first_existing("prpr", "stck_prpr")

    out = pd.DataFrame(index=df.index)
    out["종목코드"] = df[code].astype(str).str.zfill(6) if code else ""
    out["종목명"] = df[name].astype(str) if name else ""
    out["보유수량"] = (
        pd.to_numeric(df[qty], errors="coerce").fillna(0).astype(int) if qty else 0
    )
    out["매도가능수량"] = (
        pd.to_numeric(df[sellable], errors="coerce").fillna(0).astype(int)
        if sellable else out["보유수량"]
    )
    out["평균매입가"] = (
        pd.to_numeric(df[avg], errors="coerce").fillna(0.0) if avg else 0.0
    )
    out["현재가"] = (
        pd.to_numeric(df[cur], errors="coerce").fillna(0.0) if cur else 0.0
    )
    return out[out["보유수량"] > 0].reset_index(drop=True)


def _actual_holdings_map(client):
    try:
        raw = client.domestic_balance()
    except Exception as e:
        return {}, pd.DataFrame(), f"잔고조회 실패: {type(e).__name__}: {e}"

    if not isinstance(raw, dict):
        return {}, pd.DataFrame(), "잔고조회 실패: 응답 형식 이상"

    rt_cd = str(raw.get("rt_cd", ""))
    msg1 = str(raw.get("msg1", "") or "")
    rows = raw.get("output1", []) or []

    # 모의투자에서 보유종목이 0개일 때 오류성 문구가 내려오는 경우를
    # "잔고 0"으로 정상 해석한다.
    no_balance = "잔고내역이 없습니다" in msg1
    if rt_cd and rt_cd != "0" and not no_balance:
        return {}, pd.DataFrame(), f"잔고조회 오류: {raw.get('msg_cd', '')} {msg1}".strip()

    try:
        df = _parse_domestic_holdings(raw)
    except Exception as e:
        return {}, pd.DataFrame(), f"잔고 파싱 실패: {type(e).__name__}: {e}"

    result = {}
    for _, r in df.iterrows():
        result[str(r["종목코드"]).zfill(6)] = {
            "name": r.get("종목명", ""),
            "qty": int(r.get("보유수량", 0)),
            "sellable_qty": int(r.get("매도가능수량", 0)),
            "avg_price": float(r.get("평균매입가", 0)),
            "current_price": float(r.get("현재가", 0)),
        }
    return result, df, ""


def _safe_buy_qty_from_buying_power(
    client, symbol: str, target_amount: int, current_price: float, config: AutoConfig
) -> tuple[int, Dict[str, Any], str]:
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

    buffer_ratio = max(0.0, min(0.50, float(config.buying_power_buffer_pct) / 100.0))
    buffered_amt = int(available_amt * (1.0 - buffer_ratio))
    usable_amount = min(int(target_amount), max(0, buffered_amt))
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
            f"주문가능금액 부족: 안전여유 적용 후 {usable_amount:,}원 < "
            f"최소주문금액 {config.min_order_amount:,}원"
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
        reason = f"주문가능금액 확인 완료: {usable_amount:,}원 / {qty}주"
    return qty, meta, reason


def _place_order(
    client, state, symbol, side, qty, reason, execute_orders,
    source="APP", duplicate_guard_seconds=90,
):
    if qty <= 0:
        return {"status": "SKIP", "qty": 0, "reason": "주문수량 0"}
    if not execute_orders:
        return {"status": "DRY", "qty": int(qty), "reason": reason}

    hit, why = _duplicate_guard_hit(symbol, side, duplicate_guard_seconds)
    if hit:
        return {"status": "DUPLICATE_BLOCKED", "qty": int(qty), "reason": why}

    _write_order_lock(symbol, side, source)
    try:
        res = client.domestic_order(symbol, int(qty), side, market_order=True)
    except Exception as e:
        _event(state, "ORDER_ERROR", symbol, repr(e))
        return {"status": "ERROR", "qty": int(qty), "error": repr(e)}

    if _order_ok(res):
        state["daily_orders"] += 1
        _event(state, f"{str(side).upper()}_ORDER", symbol, f"{qty}주 · {reason} · {res}")
        return {"status": "ORDERED", "qty": int(qty), "response": res}

    msg_cd = res.get("msg_cd", "") if isinstance(res, dict) else ""
    msg1 = res.get("msg1", "") if isinstance(res, dict) else ""
    _event(state, "ORDER_REJECT", symbol, f"{msg_cd} {msg1} {res}")
    return {
        "status": "REJECT",
        "qty": int(qty),
        "msg_cd": msg_cd,
        "msg1": msg1,
        "response": res,
    }


def _leader_symbols(leader_df: pd.DataFrame) -> set[str]:
    if leader_df is None or leader_df.empty:
        return set()
    result = set()
    for raw in leader_df.get("종목코드", pd.Series(dtype=str)).tolist():
        code = str(raw).zfill(6)
        if len(code) == 6 and code.isdigit():
            result.add(code)
    return result


def _adopt_demo_top5_holdings(client, state, holdings, leader_df, config, result) -> None:
    if str(getattr(client, "env", "demo")).lower() != "demo":
        return
    if not bool(config.adopt_existing_demo_top5):
        return

    leaders = _leader_symbols(leader_df)
    for symbol in leaders:
        if symbol in state["positions"]:
            continue
        actual = holdings.get(symbol)
        if not actual:
            continue
        qty = int(actual.get("qty", 0))
        if qty <= 0:
            continue
        state["positions"][symbol] = {
            "name": actual.get("name", ""),
            "created_at": _now_kst().isoformat(),
            "buy_stage": 2,  # 기존 보유는 추가매수 금지
            "avg_price": float(actual.get("avg_price", 0) or 0),
            "actual_qty": qty,
            "adopted_existing": True,
            "take1_sent": False,
            "exit_pending": False,
        }
        _diag(
            result, symbol, "ADOPT_EXISTING_TOP5",
            f"모의계좌 보유 {qty}주를 TOP5 추적 대상으로 복구",
        )


def _set_kr_pending(
    pos: Dict[str, Any],
    now: datetime,
    action: str,
    side: str,
    order_qty: int,
    before_qty: int,
    act: Dict[str, Any],
    stage: int = 0,
) -> None:
    if act.get("status") != "ORDERED":
        return
    expected_after = (
        before_qty + int(order_qty)
        if side == "buy"
        else max(0, before_qty - int(order_qty))
    )
    pos["pending_order"] = {
        "action": action,
        "side": side,
        "qty": int(order_qty),
        "before_qty": int(before_qty),
        "expected_after_qty": int(expected_after),
        "stage": int(stage),
        "sent_at": now.isoformat(),
    }


def _confirm_or_wait_pending(
    result: Dict[str, Any],
    symbol: str,
    pos: Dict[str, Any],
    actual_qty: int,
    now: datetime,
    config: AutoConfig,
) -> bool:
    """True면 이번 사이클에서 이 종목에 추가 주문을 하면 안 된다."""
    pending = pos.get("pending_order")
    if not isinstance(pending, dict) or not pending:
        return False

    side = str(pending.get("side", "")).lower()
    before_qty = int(pending.get("before_qty", 0) or 0)
    expected_after = int(pending.get("expected_after_qty", before_qty) or before_qty)

    if side == "buy":
        confirmed = actual_qty >= expected_after and expected_after > before_qty
    elif side == "sell":
        confirmed = actual_qty <= expected_after
    else:
        confirmed = False

    if confirmed:
        action = str(pending.get("action", ""))
        stage = int(pending.get("stage", 0) or 0)
        if stage in (1, 2):
            pos["buy_stage"] = max(int(pos.get("buy_stage", 0) or 0), stage)
        if action == "TAKE1":
            pos["take1_sent"] = True
        if action in ("TAKE2", "STOP_LOSS", "FORCE_SELL") and actual_qty <= 0:
            pos["exit_confirmed"] = True

        pos.pop("pending_order", None)
        pos["last_confirmed_at"] = now.isoformat()
        _diag(
            result, symbol, "ORDER_CONFIRMED",
            f"한국투자 잔고 반영 확인: {before_qty}주 → {actual_qty}주",
            before_qty=before_qty, after_qty=actual_qty,
        )
        return False

    sent_at = str(pending.get("sent_at", "") or "")
    try:
        age = (now - datetime.fromisoformat(sent_at)).total_seconds()
    except Exception:
        age = 0

    # 타임아웃이 되어도 같은 주문을 자동 재발사하지 않는다.
    # 잔고가 확인될 때까지 사람이 진단 로그를 보고 판단하도록 유지.
    if age >= int(config.pending_timeout_seconds):
        _diag(
            result, symbol, "PENDING_TIMEOUT_LOCKED",
            (
                f"주문 후 {age:.0f}초 동안 잔고 반영 미확인. "
                f"안전을 위해 자동 재주문 금지 유지: "
                f"{pending.get('action', '')} {pending.get('qty', 0)}주 / "
                f"KIS 확인수량 {actual_qty}주"
            ),
            before_qty=before_qty, after_qty=actual_qty,
        )
    else:
        _diag(
            result, symbol, "WAIT_PENDING_ORDER",
            (
                f"체결/잔고반영 대기 중: {pending.get('action', '')} "
                f"{pending.get('qty', 0)}주 · KIS 확인수량 {actual_qty}주 · "
                f"같은 종목 추가 주문 차단"
            ),
            before_qty=before_qty, after_qty=actual_qty,
        )
    return True


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
        "market": "KR",
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
        result["message"] = "국내 분석/관리 시간 외(08:30~16:00)"
        save_state(state)
        return result

    order_window_open = dtime(9, 0) <= now.time() < dtime(15, 30)
    if execute_orders and not order_window_open:
        execute_orders = False
        result["execute_orders"] = False
        result["order_gate_message"] = "현재는 정규장 주문시간(09:00~15:30) 밖이라 분석/진단만 수행합니다."

    holdings, holdings_df, balance_warning = _actual_holdings_map(client)
    result["holdings"] = holdings_df.to_dict("records") if not holdings_df.empty else []
    if balance_warning:
        result["balance_warning"] = balance_warning

    _adopt_demo_top5_holdings(client, state, holdings, leader_df, config, result)

    for symbol, pos in list(state["positions"].items()):
        actual = holdings.get(symbol, {})
        actual_qty = int(actual.get("qty", 0))
        sellable_qty = int(actual.get("sellable_qty", actual_qty) or 0)

        # pending 주문은 잔고 0이어도 먼저 확인한다.
        if _confirm_or_wait_pending(result, symbol, pos, actual_qty, now, config):
            continue

        if actual_qty <= 0:
            created = str(pos.get("created_at", "") or "")
            try:
                age = now - datetime.fromisoformat(created)
            except Exception:
                age = timedelta(minutes=10)

            # 신규매수 직후가 아닌데 실제 잔고 0이면 청산 완료로 제거.
            if age > timedelta(minutes=5):
                state["positions"].pop(symbol, None)
                _diag(result, symbol, "DROP_TRACKING", "KIS 실제 보유수량 0주 → 추적 종료", after_qty=0)
            continue

        pos["actual_qty"] = actual_qty
        pos["sellable_qty"] = sellable_qty
        if float(actual.get("avg_price", 0) or 0) > 0:
            pos["avg_price"] = float(actual["avg_price"])

        # 강제청산: 분할매도 규칙보다 우선, 단 실제 매도가능수량만 주문
        if now.time() >= _clock(config.force_exit_time):
            if now.time() < dtime(15, 30):
                order_qty = min(actual_qty, sellable_qty) if sellable_qty > 0 else 0
                if order_qty <= 0:
                    _diag(
                        result, symbol, "WAIT_SELLABLE_QTY",
                        f"강제청산 대상이나 KIS 매도가능수량 0주 / 보유 {actual_qty}주",
                        before_qty=actual_qty, after_qty=actual_qty,
                    )
                    continue

                act = _place_order(
                    client, state, symbol, "sell", order_qty,
                    f"당일 강제청산 {config.force_exit_time} · KIS 실제 매도가능수량 기준",
                    execute_orders, source, config.duplicate_guard_seconds,
                )
                result["actions"].append({
                    "symbol": symbol,
                    "name": actual.get("name") or pos.get("name", symbol),
                    "action": "FORCE_SELL",
                    "qty": order_qty,
                    "before_qty": actual_qty,
                    "after_qty": max(0, actual_qty - order_qty),
                    **act,
                })
                _set_kr_pending(pos, now, "FORCE_SELL", "sell", order_qty, actual_qty, act)
                continue

            _diag(
                result, symbol, "MISSED_FORCE_EXIT",
                f"{config.force_exit_time} 이후이며 15:30 정규장 종료. 실제 보유 {actual_qty}주",
                before_qty=actual_qty, after_qty=actual_qty,
            )
            continue

        price, price_err = _current_price(client, symbol)
        if price <= 0:
            fallback = float(actual.get("current_price", 0) or 0)
            if fallback > 0:
                price = fallback
                _diag(result, symbol, "PRICE_FALLBACK_BALANCE", f"현재가 API 실패 → 잔고 현재가 {price:,.0f}원 사용 / {price_err}")
            else:
                _diag(result, symbol, "SKIP_MANAGE_PRICE", price_err)
                continue

        avg = float(pos.get("avg_price") or actual.get("avg_price") or price)
        pnl = (price / avg - 1) * 100 if avg > 0 else 0.0

        # 손절은 전량, 실제 매도가능수량 기준
        if pnl <= -abs(float(config.stop_loss_pct)):
            order_qty = min(actual_qty, sellable_qty) if sellable_qty > 0 else 0
            if order_qty <= 0:
                _diag(result, symbol, "WAIT_SELLABLE_QTY", f"손절 조건이나 매도가능수량 0주 / 보유 {actual_qty}주")
                continue
            act = _place_order(
                client, state, symbol, "sell", order_qty,
                f"손절 {pnl:.2f}% · KIS 실제 매도가능수량 기준",
                execute_orders, source, config.duplicate_guard_seconds,
            )
            result["actions"].append({
                "symbol": symbol,
                "name": actual.get("name") or pos.get("name", symbol),
                "action": "STOP_LOSS",
                "pnl": round(pnl, 3),
                "qty": order_qty,
                "before_qty": actual_qty,
                "after_qty": max(0, actual_qty - order_qty),
                **act,
            })
            _set_kr_pending(pos, now, "STOP_LOSS", "sell", order_qty, actual_qty, act)
            continue

        # 중요: 5%를 바로 넘어도 TAKE1을 건너뛰지 않는다.
        if pnl >= float(config.take1_pct) and not pos.get("take1_sent"):
            order_qty = max(1, actual_qty // 2)
            if sellable_qty > 0:
                order_qty = min(order_qty, sellable_qty)
            else:
                order_qty = 0
            if order_qty <= 0:
                _diag(result, symbol, "WAIT_SELLABLE_QTY", f"1차 익절 조건이나 매도가능수량 0주 / 보유 {actual_qty}주")
                continue

            act = _place_order(
                client, state, symbol, "sell", order_qty,
                f"1차 익절 {pnl:.2f}% · 보유수량의 약 50%",
                execute_orders, source, config.duplicate_guard_seconds,
            )
            result["actions"].append({
                "symbol": symbol,
                "name": actual.get("name") or pos.get("name", symbol),
                "action": "TAKE1",
                "pnl": round(pnl, 3),
                "qty": order_qty,
                "before_qty": actual_qty,
                "after_qty": max(0, actual_qty - order_qty),
                **act,
            })
            if act.get("status") == "DRY":
                pos["take1_sent"] = True
            else:
                _set_kr_pending(pos, now, "TAKE1", "sell", order_qty, actual_qty, act)
            continue

        # TAKE1 체결 확인 후에만 TAKE2 전량
        if pnl >= float(config.take2_pct) and pos.get("take1_sent"):
            order_qty = min(actual_qty, sellable_qty) if sellable_qty > 0 else 0
            if order_qty <= 0:
                _diag(result, symbol, "WAIT_SELLABLE_QTY", f"2차 익절 조건이나 매도가능수량 0주 / 보유 {actual_qty}주")
                continue
            act = _place_order(
                client, state, symbol, "sell", order_qty,
                f"2차 익절 {pnl:.2f}% · 남은 수량 전량",
                execute_orders, source, config.duplicate_guard_seconds,
            )
            result["actions"].append({
                "symbol": symbol,
                "name": actual.get("name") or pos.get("name", symbol),
                "action": "TAKE2",
                "pnl": round(pnl, 3),
                "qty": order_qty,
                "before_qty": actual_qty,
                "after_qty": max(0, actual_qty - order_qty),
                **act,
            })
            _set_kr_pending(pos, now, "TAKE2", "sell", order_qty, actual_qty, act)
            continue

        if now.time() >= _clock(config.last_entry_time):
            continue

        if pos.get("leader_exception"):
            continue

        parts = _split_amounts(config)
        stage = int(pos.get("buy_stage", 1))

        # 국내 추가매수는 딱 1번만: BUY2
        if stage == 1 and pnl >= float(config.add2_trigger_pct):
            qty, bp, bp_reason = _safe_buy_qty_from_buying_power(
                client, symbol, parts[1], price, config
            )
            cost = int(qty * price)
            if qty > 0 and state["daily_buy_amount"] + cost <= config.daily_budget:
                act = _place_order(
                    client, state, symbol, "buy", qty,
                    f"2차 분할매수 +{pnl:.2f}% · {bp_reason}",
                    execute_orders, source, config.duplicate_guard_seconds,
                )
                result["actions"].append({
                    "symbol": symbol,
                    "name": actual.get("name") or pos.get("name", symbol),
                    "action": "BUY2",
                    "pnl": round(pnl, 3),
                    "qty": qty,
                    "before_qty": actual_qty,
                    "after_qty": actual_qty + qty,
                    "buying_power": bp,
                    **act,
                })
                if act.get("status") == "ORDERED":
                    pos["expected_qty"] = actual_qty + qty
                    state["daily_buy_amount"] += cost
                    _set_kr_pending(pos, now, "BUY2", "buy", qty, actual_qty, act, stage=2)
                elif act.get("status") == "DRY":
                    pos["buy_stage"] = 2
                continue

    if now.time() >= _clock(config.last_entry_time):
        result["message"] = (
            f"{config.last_entry_time} 이후 신규/추가매수 금지 · "
            f"{config.force_exit_time}부터 남은 실제 보유수량 전량청산"
        )
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
        if len(state["positions"]) >= int(config.max_positions):
            break

        symbol = str(row.get("종목코드", "")).zfill(6)
        if not (len(symbol) == 6 and symbol.isdigit()):
            _diag(result, symbol, "SKIP_BAD_SYMBOL", "종목코드가 6자리 숫자가 아님")
            continue
        if symbol in state["positions"]:
            _diag(result, symbol, "SKIP_ALREADY_TRACKED", "이미 자동매매 추적중")
            continue
        if symbol in holdings:
            _diag(result, symbol, "SKIP_ALREADY_HELD", f"실제 계좌 보유중: {holdings[symbol].get('qty', 0)}주")
            continue

        signal = str(row.get("판정", ""))
        combined = float(row.get("종합점수", 0) or 0)
        lead_score = float(row.get("주도주점수", 0) or 0)
        rank_text = str(row.get("순위", "")).strip()
        is_rank1 = "1위" in rank_text or rank_text in ("1", "1.0")

        normal_entry = (
            (not bool(config.require_green_signal) or "매수 후보" in signal)
            and combined >= float(config.min_combined_score)
        )
        leader_exception = (
            bool(config.leader_exception_enabled)
            and is_rank1
            and lead_score >= float(config.leader_exception_min_lead_score)
            and combined >= float(config.leader_exception_min_combined_score)
        )
        demo_relaxed = (
            is_demo
            and bool(config.demo_relaxed_entry_enabled)
            and combined >= float(config.demo_min_combined_score)
        )

        if not (normal_entry or leader_exception or demo_relaxed):
            _diag(
                result, symbol, "SKIP_SCORE",
                f"점수 미달: 종합 {combined:.1f}, 주도 {lead_score:.1f}, 신호 {signal}",
            )
            continue

        price, price_err = _current_price(client, symbol)
        if price <= 0:
            _diag(result, symbol, "SKIP_PRICE_LOOKUP", price_err, combined=combined, lead_score=lead_score)
            continue

        qty, bp, bp_reason = _safe_buy_qty_from_buying_power(
            client, symbol, parts[0], price, config
        )
        if qty <= 0:
            _diag(result, symbol, "SKIP_BUYING_POWER", bp_reason, buying_power=bp)
            continue

        cost = int(qty * price)
        if state["daily_buy_amount"] + cost > int(config.daily_budget):
            _diag(
                result, symbol, "SKIP_DAILY_BUDGET",
                f"누적 {state['daily_buy_amount']:,} + 예상 {cost:,} > 일일한도 {config.daily_budget:,}",
            )
            continue
        if state["daily_orders"] >= int(config.max_daily_orders):
            _diag(result, symbol, "SKIP_DAILY_ORDERS", f"일일 주문 {config.max_daily_orders}회 도달")
            break

        if normal_entry:
            action_name = "BUY1"
            entry_reason = f"정상 1차매수 · 종합 {combined:.1f} · {signal}"
            leader_exception_flag = False
        elif leader_exception:
            action_name = "BUY1_EXCEPTION"
            entry_reason = (
                f"대장주 예외 1차매수 · TOP1 · 주도 {lead_score:.1f} · "
                f"종합 {combined:.1f} · {signal}"
            )
            leader_exception_flag = True
        else:
            action_name = "BUY1_DEMO_RELAXED"
            entry_reason = f"모의완화 1차매수 · TOP5 · 종합 {combined:.1f} · {signal}"
            leader_exception_flag = False

        act = _place_order(
            client, state, symbol, "buy", qty,
            entry_reason + " · " + bp_reason,
            execute_orders, source, config.duplicate_guard_seconds,
        )

        result["actions"].append({
            "symbol": symbol,
            "name": str(row.get("종목명", symbol)),
            "action": action_name,
            "current_price": price,
            "estimated_cost": cost,
            "combined_score": combined,
            "lead_score": lead_score,
            "qty": qty,
            "before_qty": 0,
            "after_qty": qty,
            "buying_power_reason": bp_reason,
            "buying_power": bp,
            **act,
        })

        if act.get("status") == "ORDERED":
            pos = {
                "name": str(row.get("종목명", "")),
                "created_at": now.isoformat(),
                "buy_stage": 0,
                "avg_price": price,
                "expected_qty": qty,
                "leader_exception": leader_exception_flag,
                "take1_sent": False,
                "exit_pending": False,
            }
            _set_kr_pending(pos, now, action_name, "buy", qty, 0, act, stage=1)
            state["positions"][symbol] = pos
            state["daily_buy_amount"] += cost

    if not result["actions"]:
        result["message"] = "실제 주문/DRY 액션 없음. diagnostics에서 종목별 SKIP 이유를 확인하세요."

    save_state(state)
    result["state"] = state
    return result


# -------------------------------------------------------------------
# 미국 상태/로직
# -------------------------------------------------------------------
# 미국 로직은 기존 동작을 유지하되 국내와 같은 2회 매수 비율을 사용한다.

def load_us_state() -> Dict[str, Any]:
    fresh = {
        "date": _today_et(),
        "positions": {},
        "daily_buy_amount_usd": 0.0,
        "daily_orders": 0,
        "events": [],
    }
    if not US_STATE_FILE.exists():
        return fresh
    try:
        state = json.loads(US_STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return fresh
    if state.get("date") != _today_et():
        return fresh
    state.setdefault("positions", {})
    state.setdefault("daily_buy_amount_usd", 0.0)
    state.setdefault("daily_orders", 0)
    state.setdefault("events", [])
    return state


def save_us_state(state: Dict[str, Any]) -> None:
    US_STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def reset_us_state() -> None:
    if US_STATE_FILE.exists():
        US_STATE_FILE.unlink()


def _us_event(state, event, symbol="", detail=""):
    row = {
        "time": _now_et().isoformat(timespec="seconds"),
        "event": event,
        "symbol": symbol,
        "detail": str(detail)[:1000],
    }
    state["events"].append(row)
    state["events"] = state["events"][-200:]
    append_trade_log({
        "time": row["time"],
        "mode": "AUTO",
        "market": "US",
        "symbol": symbol,
        "event": event,
        "detail": row["detail"],
    })


def _us_quote_exchange(exchange: str) -> str:
    return {"NASD": "NAS", "NYSE": "NYS", "AMEX": "AMS"}.get(str(exchange).upper(), "NAS")


def _overseas_current_price(client, symbol: str, exchange: str = "NASD") -> float:
    try:
        raw = client.get(
            "/uapi/overseas-price/v1/quotations/price",
            "HHDFS00000300",
            {"AUTH": "", "EXCD": _us_quote_exchange(exchange), "SYMB": str(symbol).upper()},
        )
    except Exception:
        return 0.0
    out = (raw or {}).get("output", {}) or {}
    for key in ("last", "last_price", "ovrs_nmix_prpr", "stck_prpr"):
        try:
            value = float(out.get(key, 0) or 0)
            if value > 0:
                return value
        except Exception:
            pass
    return 0.0


def _overseas_all_balances(client):
    responses, errors = [], []
    for exchange in ("NASD", "NYSE", "AMEX"):
        try:
            raw = client.overseas_balance_us(exchange=exchange, currency="USD")
        except Exception as e:
            errors.append(f"{exchange}: {type(e).__name__}: {e}")
            continue
        if not isinstance(raw, dict):
            errors.append(f"{exchange}: 응답 형식 이상")
            continue
        rt_cd = str(raw.get("rt_cd", ""))
        if rt_cd and rt_cd != "0":
            errors.append(f"{exchange}: {raw.get('msg_cd', '')} {raw.get('msg1', '')}".strip())
            continue
        raw = dict(raw)
        raw["_exchange"] = exchange
        responses.append(raw)
    return responses, errors


def _actual_us_holdings_map(client):
    responses, errors = _overseas_all_balances(client)
    if not responses:
        return {}, pd.DataFrame(), " / ".join(errors) or "미국 잔고조회 실패"
    try:
        df = merge_overseas_holdings(responses)
    except Exception as e:
        return {}, pd.DataFrame(), f"미국 잔고 파싱 실패: {type(e).__name__}: {e}"

    result = {}
    for _, r in df.iterrows():
        symbol = str(r.get("종목코드", "")).strip().upper()
        if not symbol:
            continue
        qty = int(float(r.get("보유수량", 0) or 0))
        if qty <= 0:
            continue
        result[symbol] = {
            "qty": qty,
            "sellable_qty": int(float(r.get("매도가능수량", 0) or 0)),
            "avg_price": float(r.get("평균매입가", 0) or 0),
            "current_price": float(r.get("현재가", 0) or 0),
            "name": str(r.get("종목명", "") or symbol),
            "exchange": str(r.get("거래소", "") or "NASD").upper(),
        }
    return result, df, " / ".join(errors)


def _adopt_existing_us_holdings(client, state, holdings, config, result):
    if str(getattr(client, "env", "demo")).lower() != "demo":
        return
    for symbol, actual in holdings.items():
        if symbol in state["positions"]:
            state["positions"][symbol].setdefault("exchange", actual.get("exchange", "NASD"))
            continue
        qty = int(actual.get("qty", 0) or 0)
        if qty <= 0:
            continue
        state["positions"][symbol] = {
            "name": actual.get("name", symbol),
            "created_at": _now_et().isoformat(),
            "buy_stage": 2,
            "avg_price": float(actual.get("avg_price", 0) or 0),
            "actual_qty": qty,
            "expected_qty": qty,
            "exchange": actual.get("exchange", "NASD"),
            "take1_sent": False,
            "exit_pending": False,
            "adopted_existing": True,
        }
        _diag(result, symbol, "ADOPT_EXISTING_US", f"한국투자 실제 보유 {qty}주를 자동 관리 대상으로 복구")


def _us_stage_budget(config: AutoConfig, stage: int) -> float:
    weights = [config.buy1_pct, config.buy2_pct]
    total = max(1, sum(weights))
    return float(config.us_per_stock_budget_usd) * weights[stage - 1] / total


def _us_stage_qty(config: AutoConfig, stage: int, price: float) -> int:
    return _calc_qty(_us_stage_budget(config, stage), price)


def _us_first_entry_qty(config: AutoConfig, price: float) -> int:
    qty = _us_stage_qty(config, 1, price)
    if qty > 0:
        return qty
    if config.allow_single_share_over_stage_budget and price > 0 and price <= float(config.us_per_stock_budget_usd):
        return 1
    return 0


def _place_overseas_order(
    client, state, symbol: str, side: str, qty: int, price: float,
    reason: str, execute_orders: bool, exchange: str = "NASD",
):
    if qty <= 0:
        return {"status": "SKIP", "reason": "주문수량 0"}
    if price <= 0:
        return {"status": "SKIP", "reason": "주문가격 0"}
    exchange = str(exchange or "NASD").upper()
    if not execute_orders:
        _us_event(state, f"DRY_{side.upper()}", symbol, f"{qty}주 @ ${price:.2f} · {exchange} · {reason}")
        return {"status": "DRY", "qty": qty, "price": price, "exchange": exchange}
    try:
        res = client.overseas_order_us(
            symbol=symbol, qty=qty, side=side, limit_price=price, exchange=exchange
        )
    except Exception as e:
        _us_event(state, "ORDER_ERROR", symbol, repr(e))
        return {"status": "ERROR", "error": repr(e), "exchange": exchange}
    if _order_ok(res):
        state["daily_orders"] += 1
        _us_event(state, f"{side.upper()}_ORDER", symbol, f"{qty}주 @ ${price:.2f} · {exchange} · {reason} · {res}")
        return {"status": "ORDERED", "qty": qty, "price": price, "exchange": exchange, "response": res}
    _us_event(state, "ORDER_REJECT", symbol, str(res))
    return {
        "status": "REJECT", "qty": qty, "price": price, "exchange": exchange,
        "msg_cd": res.get("msg_cd", "") if isinstance(res, dict) else "",
        "msg1": res.get("msg1", "") if isinstance(res, dict) else "",
        "response": res,
    }


def run_overseas_cycle(
    client,
    leader_df: pd.DataFrame,
    config: AutoConfig,
    execute_orders: bool = False,
    source: str = "APP",
) -> Dict[str, Any]:
    state = load_us_state()
    now = _now_et()
    result = {
        "time": now.isoformat(timespec="seconds"),
        "market": "US",
        "execute_orders": execute_orders,
        "actions": [],
        "diagnostics": [],
        "state": state,
    }

    if now.weekday() >= 5:
        result["message"] = "미국 주말: 주문 없음"
        save_us_state(state)
        return result
    if not (dtime(9, 30) <= now.time() < dtime(16, 0)):
        result["message"] = "미국 정규장 외: 주문 없음"
        save_us_state(state)
        return result

    holdings, holdings_df, balance_warning = _actual_us_holdings_map(client)
    result["holdings"] = holdings_df.to_dict("records") if not holdings_df.empty else []
    if balance_warning:
        result["balance_warning"] = balance_warning

    _adopt_existing_us_holdings(client, state, holdings, config, result)

    for symbol, pos in list(state["positions"].items()):
        actual = holdings.get(symbol, {})
        actual_qty = int(actual.get("qty", 0))

        if actual_qty > 0:
            pos["actual_qty"] = actual_qty
            pos["exchange"] = actual.get("exchange", pos.get("exchange", "NASD"))
            if float(actual.get("avg_price", 0) or 0) > 0:
                pos["avg_price"] = float(actual["avg_price"])

        pending = pos.get("pending_order")
        if isinstance(pending, dict) and pending:
            side = str(pending.get("side", "")).lower()
            before_qty = int(pending.get("before_qty", 0) or 0)
            expected_after_qty = int(pending.get("expected_after_qty", before_qty) or before_qty)
            if side == "buy":
                confirmed = actual_qty >= expected_after_qty and expected_after_qty > before_qty
            elif side == "sell":
                confirmed = actual_qty <= expected_after_qty
            else:
                confirmed = False

            if confirmed:
                pending_action = str(pending.get("action", ""))
                pending_stage = int(pending.get("stage", 0) or 0)
                if pending_stage in (1, 2):
                    pos["buy_stage"] = max(int(pos.get("buy_stage", 0) or 0), pending_stage)
                if pending_action == "TAKE1":
                    pos["take1_sent"] = True
                pos.pop("pending_order", None)
                pos["last_confirmed_at"] = now.isoformat()
                _diag(result, symbol, "ORDER_CONFIRMED", f"한국투자 잔고 반영 확인: {before_qty}주 → {actual_qty}주")
            else:
                _diag(
                    result, symbol, "WAIT_PENDING_ORDER",
                    f"미체결/잔고반영 대기 중: {pending.get('action', '')} {pending.get('qty', 0)}주 · "
                    f"KIS 현재 확인수량 {actual_qty}주 · 같은 종목 추가 주문 차단",
                )
                continue

        if actual_qty <= 0:
            if execute_orders:
                pos.setdefault("awaiting_balance_confirmation", True)
                _diag(result, symbol, "WAIT_BALANCE_CONFIRMATION", "주문 추적중이나 KIS 보유수량이 아직 0주라 추가 주문 차단")
                continue
            continue

        pos.pop("awaiting_balance_confirmation", None)
        exchange = str(actual.get("exchange") or pos.get("exchange") or "NASD").upper()
        price = _overseas_current_price(client, symbol, exchange=exchange)
        if price <= 0:
            fallback = float(actual.get("current_price", 0) or 0)
            if fallback > 0:
                price = fallback
            else:
                _diag(result, symbol, "SKIP_US_PRICE", "KIS 미국 현재가 조회 실패")
                continue

        avg = float(pos.get("avg_price") or actual.get("avg_price") or price)
        pnl = (price / avg - 1) * 100 if avg > 0 else 0.0

        def _set_pending(action, side, order_qty, act, stage=0):
            if act.get("status") != "ORDERED":
                return
            expected_after = actual_qty + int(order_qty) if side == "buy" else max(0, actual_qty - int(order_qty))
            pos["pending_order"] = {
                "action": action, "side": side, "qty": int(order_qty),
                "before_qty": int(actual_qty), "expected_after_qty": int(expected_after),
                "stage": int(stage), "sent_at": now.isoformat(),
                "price": float(act.get("price", price) or price), "exchange": exchange,
            }

        if now.time() >= _clock(config.us_force_exit_time):
            act = _place_overseas_order(
                client, state, symbol, "sell", actual_qty, price,
                f"당일 강제청산 {config.us_force_exit_time} ET", execute_orders, exchange=exchange,
            )
            _set_pending("FORCE_SELL", "sell", actual_qty, act)
            result["actions"].append({
                "symbol": symbol, "action": "FORCE_SELL", "pnl": pnl,
                "before_qty": actual_qty, "after_qty": 0, **act,
            })
            continue

        if pnl <= -abs(config.stop_loss_pct):
            act = _place_overseas_order(
                client, state, symbol, "sell", actual_qty, price,
                f"손절 {pnl:.2f}%", execute_orders, exchange=exchange,
            )
            _set_pending("STOP_LOSS", "sell", actual_qty, act)
            result["actions"].append({
                "symbol": symbol, "action": "STOP_LOSS", "pnl": pnl,
                "before_qty": actual_qty, "after_qty": 0, **act,
            })
            continue

        if pnl >= config.take1_pct and not pos.get("take1_sent"):
            sell_qty = max(1, actual_qty // 2)
            act = _place_overseas_order(
                client, state, symbol, "sell", sell_qty, price,
                f"1차 익절 {pnl:.2f}%", execute_orders, exchange=exchange,
            )
            _set_pending("TAKE1", "sell", sell_qty, act)
            result["actions"].append({
                "symbol": symbol, "action": "TAKE1", "pnl": pnl,
                "before_qty": actual_qty, "after_qty": max(0, actual_qty - sell_qty), **act,
            })
            if act.get("status") == "DRY":
                pos["take1_sent"] = True
            continue

        if pnl >= config.take2_pct and pos.get("take1_sent"):
            act = _place_overseas_order(
                client, state, symbol, "sell", actual_qty, price,
                f"2차 익절 {pnl:.2f}%", execute_orders, exchange=exchange,
            )
            _set_pending("TAKE2", "sell", actual_qty, act)
            result["actions"].append({
                "symbol": symbol, "action": "TAKE2", "pnl": pnl,
                "before_qty": actual_qty, "after_qty": 0, **act,
            })
            continue

        if now.time() >= _clock(config.us_last_entry_time):
            continue

        stage = int(pos.get("buy_stage", 1))
        if state["daily_orders"] >= config.max_daily_orders:
            continue

        if stage == 1 and pnl >= config.add2_trigger_pct:
            qty = _us_stage_qty(config, 2, price)
            cost = qty * price
            if qty > 0 and state["daily_buy_amount_usd"] + cost <= config.us_daily_budget_usd:
                act = _place_overseas_order(
                    client, state, symbol, "buy", qty, price,
                    f"2차 분할매수 +{pnl:.2f}%", execute_orders, exchange=exchange,
                )
                _set_pending("BUY2", "buy", qty, act, stage=2)
                result["actions"].append({
                    "symbol": symbol, "action": "BUY2", "pnl": pnl,
                    "before_qty": actual_qty, "after_qty": actual_qty + qty, **act,
                })
                if act.get("status") == "ORDERED":
                    pos["expected_qty"] = actual_qty + qty
                    state["daily_buy_amount_usd"] += cost
                elif act.get("status") == "DRY":
                    pos["buy_stage"] = 2
                continue

    if now.time() >= _clock(config.us_last_entry_time):
        result["message"] = f"{config.us_last_entry_time} ET 이후 미국 신규매수 금지"
        save_us_state(state)
        result["state"] = state
        return result

    if len(state["positions"]) >= config.max_positions:
        result["message"] = "미국 최대 보유/주문대기 종목 수 도달"
        save_us_state(state)
        result["state"] = state
        return result

    if leader_df is None or leader_df.empty:
        result["message"] = "미국 TOP5 데이터 없음"
        save_us_state(state)
        result["state"] = state
        return result

    for _, row in leader_df.iterrows():
        if len(state["positions"]) >= config.max_positions:
            break
        if state["daily_orders"] >= config.max_daily_orders:
            break

        symbol = str(row.get("종목코드", row.get("종목", ""))).strip().upper()
        if not symbol or symbol in state["positions"]:
            continue
        signal = str(row.get("판정", row.get("종합신호", "")))
        try:
            combined = float(row.get("종합점수", 0) or 0)
        except Exception:
            combined = 0.0

        if config.require_green_signal and "매수 후보" not in signal:
            continue
        if combined < config.min_combined_score:
            continue
        if symbol in holdings:
            continue

        exchange = "NASD"
        price = _overseas_current_price(client, symbol, exchange=exchange)
        if price <= 0:
            result["actions"].append({"symbol": symbol, "action": "SKIP", "reason": "KIS 미국 현재가 조회 실패"})
            continue

        qty = _us_first_entry_qty(config, price)
        if qty <= 0:
            result["actions"].append({
                "symbol": symbol, "action": "SKIP",
                "reason": f"주가 ${price:.2f}가 미국 종목당 한도 ${config.us_per_stock_budget_usd:.2f}를 초과",
            })
            continue

        cost = qty * price
        if cost > config.us_per_stock_budget_usd:
            continue
        if state["daily_buy_amount_usd"] + cost > config.us_daily_budget_usd:
            continue

        reason = (
            f"TOP5 신규매수 · 종합점수 {combined:.1f} · "
            f"1차예산 ${_us_stage_budget(config, 1):.2f} / {qty}주"
        )
        if signal:
            reason += f" · {signal}"

        act = _place_overseas_order(
            client, state, symbol, "buy", qty, price, reason,
            execute_orders, exchange=exchange,
        )
        result["actions"].append({
            "symbol": symbol, "name": str(row.get("종목명", symbol)),
            "action": "BUY1", "reason": reason, "combined_score": combined,
            "before_qty": 0, "after_qty": qty, **act,
        })

        if act.get("status") == "ORDERED":
            state["positions"][symbol] = {
                "name": str(row.get("종목명", symbol)),
                "created_at": now.isoformat(),
                "buy_stage": 0,
                "avg_price": price,
                "expected_qty": qty,
                "exchange": exchange,
                "take1_sent": False,
                "pending_order": {
                    "action": "BUY1", "side": "buy", "qty": int(qty),
                    "before_qty": 0, "expected_after_qty": int(qty),
                    "stage": 1, "sent_at": now.isoformat(),
                    "price": float(price), "exchange": exchange,
                },
            }
            state["daily_buy_amount_usd"] += cost

    if not result["actions"]:
        if any(isinstance(p, dict) and p.get("pending_order") for p in state["positions"].values()):
            result["message"] = "미국 주문 체결/잔고반영 대기 중 · 중복주문 차단"
        else:
            result["message"] = "매수 기준을 통과한 미국 신규매수 후보 없음"

    save_us_state(state)
    result["state"] = state
    return result


run_kr_cycle = run_domestic_cycle
run_us_cycle = run_overseas_cycle

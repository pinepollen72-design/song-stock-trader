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
US_STATE_FILE = STATE_DIR / "overseas_auto_state.json"
ORDER_LOCK_FILE = STATE_DIR / "domestic_order_lock.json"

KST = ZoneInfo("Asia/Seoul")
ET = ZoneInfo("America/New_York")


@dataclass
class AutoConfig:
    # 국내 모의 테스트 기본 한도
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

    min_combined_score: float = 50.0
    require_green_signal: bool = False

    demo_relaxed_entry_enabled: bool = True
    demo_min_combined_score: float = 50.0

    leader_exception_enabled: bool = True
    leader_exception_min_lead_score: float = 75.0
    leader_exception_min_combined_score: float = 60.0

    # 분석은 08:30~16:00 계속하지만,
    # 신규매수는 장마감 청산 여유를 위해 15:10까지만.
    last_entry_time: str = "15:10"
    force_exit_time: str = "15:20"

    duplicate_guard_seconds: int = 90
    buying_power_buffer_pct: float = 5.0
    min_order_amount: int = 10_000

    # 모의계좌에서 상태파일이 초기화되었더라도
    # 현재 TOP5에 있고 실제 보유중인 종목은 자동 추적 대상으로 복구.
    adopt_existing_demo_top5: bool = True

    # 미국
    us_daily_budget_usd: float = 1500.0
    us_per_stock_budget_usd: float = 600.0
    us_last_entry_time: str = "15:30"
    us_force_exit_time: str = "15:50"


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
            json.dumps(
                {
                    "time": _now_kst().isoformat(),
                    "symbol": symbol,
                    "side": side,
                    "source": source,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass


def _duplicate_guard_hit(symbol: str, side: str, seconds: int):
    lock = _read_order_lock()
    if not lock:
        return False, ""

    if (
        str(lock.get("symbol")) != symbol
        or str(lock.get("side")) != side
    ):
        return False, ""

    try:
        age = (
            _now_kst()
            - datetime.fromisoformat(lock.get("time", ""))
        ).total_seconds()
    except Exception:
        return False, ""

    if age < seconds:
        return (
            True,
            f"중복주문 방지: {age:.0f}초 전에 "
            f"{lock.get('source', 'unknown')}에서 같은 주문 시도",
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
    total_pct = max(
        1,
        int(config.buy1_pct)
        + int(config.buy2_pct)
        + int(config.buy3_pct),
    )

    parts = [
        int(config.per_stock_budget * int(config.buy1_pct) / total_pct),
        int(config.per_stock_budget * int(config.buy2_pct) / total_pct),
        int(config.per_stock_budget * int(config.buy3_pct) / total_pct),
    ]
    parts[-1] += int(config.per_stock_budget) - sum(parts)
    return parts


def _parse_domestic_holdings(balance_json: Dict[str, Any]) -> pd.DataFrame:
    rows = (balance_json or {}).get("output1", []) or []

    if not rows:
        return pd.DataFrame(
            columns=["종목코드", "종목명", "보유수량", "평균매입가", "현재가"]
        )

    df = pd.DataFrame(rows)

    def first_existing(*names):
        for name in names:
            if name in df.columns:
                return name
        return None

    code = first_existing("pdno", "mksc_shrn_iscd")
    name = first_existing("prdt_name", "hts_kor_isnm")
    qty = first_existing("hldg_qty", "hold_qty")
    avg = first_existing("pchs_avg_pric", "avg_pric")
    cur = first_existing("prpr", "stck_prpr")

    out = pd.DataFrame(index=df.index)
    out["종목코드"] = (
        df[code].astype(str).str.zfill(6)
        if code
        else ""
    )
    out["종목명"] = (
        df[name].astype(str)
        if name
        else ""
    )
    out["보유수량"] = (
        pd.to_numeric(df[qty], errors="coerce").fillna(0).astype(int)
        if qty
        else 0
    )
    out["평균매입가"] = (
        pd.to_numeric(df[avg], errors="coerce").fillna(0.0)
        if avg
        else 0.0
    )
    out["현재가"] = (
        pd.to_numeric(df[cur], errors="coerce").fillna(0.0)
        if cur
        else 0.0
    )

    return out[out["보유수량"] > 0].reset_index(drop=True)


def _actual_holdings_map(client):
    try:
        raw = client.domestic_balance()
        df = _parse_domestic_holdings(raw)
    except Exception as e:
        return {}, pd.DataFrame(), (
            f"잔고조회 실패: {type(e).__name__}: {e}"
        )

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
    if current_price <= 0:
        return 0, {}, "현재가가 0 이하"

    try:
        raw = client.domestic_buying_power(
            symbol=symbol,
            reference_price=int(current_price),
        )
    except Exception as e:
        return 0, {}, (
            f"매수가능조회 실패: {type(e).__name__}: {e}"
        )

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
        min(
            0.50,
            float(config.buying_power_buffer_pct) / 100.0,
        ),
    )

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
        return (
            0,
            meta,
            f"주문가능금액 부족: 안전여유 적용 후 "
            f"{usable_amount:,}원 < 최소주문금액 "
            f"{config.min_order_amount:,}원",
        )

    if qty <= 0:
        return (
            0,
            meta,
            f"주문가능수량 0: 사용가능금액 {usable_amount:,}원 / "
            f"현재가 {current_price:,.0f}원",
        )

    if usable_amount < int(target_amount):
        reason = (
            f"주문금액 자동축소: 목표 {target_amount:,}원 → "
            f"사용가능 {usable_amount:,}원 / {qty}주"
        )
    else:
        reason = (
            f"주문가능금액 확인 완료: "
            f"{usable_amount:,}원 / {qty}주"
        )

    return qty, meta, reason


def _place_order(
    client,
    state,
    symbol,
    side,
    qty,
    reason,
    execute_orders,
    source="APP",
    duplicate_guard_seconds=90,
):
    if qty <= 0:
        return {
            "status": "SKIP",
            "qty": 0,
            "reason": "주문수량 0",
        }

    if not execute_orders:
        # DRY는 주문 로그만 결과로 돌려주고 상태/누적금액은 변경하지 않습니다.
        return {
            "status": "DRY",
            "qty": int(qty),
            "reason": reason,
        }

    hit, why = _duplicate_guard_hit(
        symbol,
        side,
        duplicate_guard_seconds,
    )
    if hit:
        return {
            "status": "DUPLICATE_BLOCKED",
            "qty": int(qty),
            "reason": why,
        }

    _write_order_lock(symbol, side, source)

    try:
        res = client.domestic_order(
            symbol,
            int(qty),
            side,
            market_order=True,
        )
    except Exception as e:
        _event(state, "ORDER_ERROR", symbol, repr(e))
        return {
            "status": "ERROR",
            "qty": int(qty),
            "error": repr(e),
        }

    if _order_ok(res):
        state["daily_orders"] += 1
        _event(
            state,
            f"{str(side).upper()}_ORDER",
            symbol,
            f"{qty}주 · {reason} · {res}",
        )
        return {
            "status": "ORDERED",
            "qty": int(qty),
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


def _adopt_demo_top5_holdings(
    client,
    state: Dict[str, Any],
    holdings: Dict[str, Dict[str, Any]],
    leader_df: pd.DataFrame,
    config: AutoConfig,
    result: Dict[str, Any],
) -> None:
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
            "buy_stage": 1,
            "avg_price": float(actual.get("avg_price", 0) or 0),
            "actual_qty": qty,
            "adopted_existing": True,
            "take1_sent": False,
            "exit_pending": False,
        }

        _diag(
            result,
            symbol,
            "ADOPT_EXISTING_TOP5",
            f"모의계좌 보유 {qty}주를 TOP5 추적 대상으로 복구",
        )


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
        result["message"] = (
            "국내 분석/관리 시간 외(08:30~16:00)"
        )
        save_state(state)
        return result

    # 실제 KIS 정규장 주문 전송은 09:00~15:30만.
    order_window_open = (
        dtime(9, 0) <= now.time() < dtime(15, 30)
    )

    if execute_orders and not order_window_open:
        execute_orders = False
        result["execute_orders"] = False
        result["order_gate_message"] = (
            "현재는 정규장 주문시간(09:00~15:30) 밖이라 "
            "분석/진단만 수행합니다."
        )

    holdings, holdings_df, balance_warning = (
        _actual_holdings_map(client)
    )

    result["holdings"] = (
        holdings_df.to_dict("records")
        if not holdings_df.empty
        else []
    )

    if balance_warning:
        result["balance_warning"] = balance_warning

    # 상태 초기화/Render 재배포 후에도 TOP5 실제 보유종목은 모의에서 자동 복구
    _adopt_demo_top5_holdings(
        client,
        state,
        holdings,
        leader_df,
        config,
        result,
    )

    # -----------------------------------------------------
    # 1. 보유종목 관리
    # -----------------------------------------------------
    for symbol, pos in list(state["positions"].items()):
        actual = holdings.get(symbol, {})
        actual_qty = int(actual.get("qty", 0))

        if actual_qty <= 0:
            # 주문 직후 잔고 반영 지연은 5분 유예
            created = str(pos.get("created_at", "") or "")
            try:
                age = now - datetime.fromisoformat(created)
            except Exception:
                age = timedelta(minutes=10)

            if age > timedelta(minutes=5):
                state["positions"].pop(symbol, None)
                _diag(
                    result,
                    symbol,
                    "DROP_TRACKING",
                    "실제 잔고에 보유수량이 없어 추적 제거",
                )
            continue

        pos["actual_qty"] = actual_qty

        if float(actual.get("avg_price", 0) or 0) > 0:
            pos["avg_price"] = float(actual["avg_price"])

        # 장마감 강제청산은 현재가 조회 실패와 무관하게 시장가 매도 시도
        if now.time() >= _clock(config.force_exit_time):
            if now.time() < dtime(15, 30):
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
                    "qty": actual_qty,
                    **act,
                })

                if act.get("status") == "ORDERED":
                    pos["exit_pending"] = True
                continue

            _diag(
                result,
                symbol,
                "MISSED_FORCE_EXIT",
                (
                    f"{config.force_exit_time} 강제청산 시각 이후이며 "
                    "15:30 정규장 종료로 신규 시장가 주문을 보내지 않음. "
                    f"실제 보유 {actual_qty}주"
                ),
            )
            continue

        price, price_err = _current_price(client, symbol)

        # KIS 현재가 5xx가 나도 잔고 응답의 현재가를 fallback으로 사용
        if price <= 0:
            fallback = float(actual.get("current_price", 0) or 0)
            if fallback > 0:
                price = fallback
                _diag(
                    result,
                    symbol,
                    "PRICE_FALLBACK_BALANCE",
                    (
                        "현재가 API 실패 → 잔고조회 현재가 사용: "
                        f"{price:,.0f}원 / {price_err}"
                    ),
                )
            else:
                _diag(
                    result,
                    symbol,
                    "SKIP_MANAGE_PRICE",
                    price_err,
                )
                continue

        avg = float(
            pos.get("avg_price")
            or actual.get("avg_price")
            or price
        )
        pnl = (
            (price / avg - 1) * 100
            if avg > 0
            else 0.0
        )

        if pnl <= -abs(float(config.stop_loss_pct)):
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

        if pnl >= float(config.take2_pct):
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

        if (
            pnl >= float(config.take1_pct)
            and not pos.get("take1_sent")
        ):
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

            # DRY에서는 상태 변경 금지
            if act.get("status") == "ORDERED":
                pos["take1_sent"] = True
            continue

        # 2/3차 추가매수는 장마감 신규진입 차단시간 이후에는 금지
        if now.time() >= _clock(config.last_entry_time):
            continue

        if pos.get("leader_exception"):
            continue

        parts = _split_amounts(config)
        stage = int(pos.get("buy_stage", 1))

        if stage == 1 and pnl >= float(config.add2_trigger_pct):
            qty, bp, bp_reason = _safe_buy_qty_from_buying_power(
                client,
                symbol,
                parts[1],
                price,
                config,
            )

            cost = int(qty * price)

            if (
                qty > 0
                and state["daily_buy_amount"] + cost <= config.daily_budget
            ):
                act = _place_order(
                    client,
                    state,
                    symbol,
                    "buy",
                    qty,
                    f"2차 분할매수 +{pnl:.2f}% · {bp_reason}",
                    execute_orders,
                    source,
                    config.duplicate_guard_seconds,
                )
                result["actions"].append({
                    "symbol": symbol,
                    "action": "BUY2",
                    "pnl": round(pnl, 3),
                    "buying_power": bp,
                    **act,
                })

                if act.get("status") == "ORDERED":
                    pos["buy_stage"] = 2
                    state["daily_buy_amount"] += cost
                continue

        if stage == 2 and pnl >= float(config.add3_trigger_pct):
            qty, bp, bp_reason = _safe_buy_qty_from_buying_power(
                client,
                symbol,
                parts[2],
                price,
                config,
            )

            cost = int(qty * price)

            if (
                qty > 0
                and state["daily_buy_amount"] + cost <= config.daily_budget
            ):
                act = _place_order(
                    client,
                    state,
                    symbol,
                    "buy",
                    qty,
                    f"3차 분할매수 +{pnl:.2f}% · {bp_reason}",
                    execute_orders,
                    source,
                    config.duplicate_guard_seconds,
                )
                result["actions"].append({
                    "symbol": symbol,
                    "action": "BUY3",
                    "pnl": round(pnl, 3),
                    "buying_power": bp,
                    **act,
                })

                if act.get("status") == "ORDERED":
                    pos["buy_stage"] = 3
                    state["daily_buy_amount"] += cost
                continue

    # -----------------------------------------------------
    # 2. 신규매수
    # -----------------------------------------------------
    if now.time() >= _clock(config.last_entry_time):
        result["message"] = (
            f"{config.last_entry_time} 이후 신규/추가매수 금지 · "
            f"{config.force_exit_time}부터 당일청산"
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
    is_demo = (
        str(getattr(client, "env", "demo")).lower()
        == "demo"
    )

    for _, row in leader_df.head(5).iterrows():
        if len(state["positions"]) >= int(config.max_positions):
            break

        symbol = str(row.get("종목코드", "")).zfill(6)

        stock_name = str(row.get("종목명", "")).strip()
        banned_keywords = ("레버리지", "인버스")

        if any(keyword in stock_name for keyword in banned_keywords):
            _diag(
                result,
                symbol,
                "SKIP_LEVERAGE_INVERSE",
                f"레버리지/인버스 상품 제외: {stock_name}",
            )
            continue

        if not (len(symbol) == 6 and symbol.isdigit()):
            _diag(
                result,
                symbol,
                "SKIP_BAD_SYMBOL",
                "종목코드가 6자리 숫자가 아님",
            )
            continue

        if symbol in state["positions"]:
            _diag(
                result,
                symbol,
                "SKIP_ALREADY_TRACKED",
                "이미 자동매매 추적중",
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

        is_rank1 = (
            "1위" in rank_text
            or rank_text in ("1", "1.0")
        )

        normal_entry = (
            (
                not bool(config.require_green_signal)
                or "매수 후보" in signal
            )
            and combined >= float(config.min_combined_score)
        )

        leader_exception = (
            bool(config.leader_exception_enabled)
            and is_rank1
            and lead_score
            >= float(config.leader_exception_min_lead_score)
            and combined
            >= float(config.leader_exception_min_combined_score)
        )

        demo_relaxed = (
            is_demo
            and bool(config.demo_relaxed_entry_enabled)
            and combined >= float(config.demo_min_combined_score)
        )

        if not (normal_entry or leader_exception or demo_relaxed):
            _diag(
                result,
                symbol,
                "SKIP_SCORE",
                (
                    f"점수 미달: 종합 {combined:.1f}, "
                    f"주도 {lead_score:.1f}, 신호 {signal}, "
                    f"demo기준 {config.demo_min_combined_score:.1f}"
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

        qty, bp, bp_reason = _safe_buy_qty_from_buying_power(
            client,
            symbol,
            parts[0],
            price,
            config,
        )

        if qty <= 0:
            _diag(
                result,
                symbol,
                "SKIP_BUYING_POWER",
                bp_reason,
                buying_power=bp,
            )
            continue

        cost = int(qty * price)

        if state["daily_buy_amount"] + cost > int(config.daily_budget):
            _diag(
                result,
                symbol,
                "SKIP_DAILY_BUDGET",
                (
                    f"누적 {state['daily_buy_amount']:,} + "
                    f"예상 {cost:,} > "
                    f"일일한도 {config.daily_budget:,}"
                ),
            )
            continue

        if state["daily_orders"] >= int(config.max_daily_orders):
            _diag(
                result,
                symbol,
                "SKIP_DAILY_ORDERS",
                f"일일 주문 {config.max_daily_orders}회 도달",
            )
            break

        if normal_entry:
            action_name = "BUY1"
            entry_reason = (
                f"정상 1차매수 · 종합 {combined:.1f} · {signal}"
            )
            leader_exception_flag = False

        elif leader_exception:
            action_name = "BUY1_EXCEPTION"
            entry_reason = (
                f"대장주 예외 1차매수 · TOP1 · "
                f"주도 {lead_score:.1f} · "
                f"종합 {combined:.1f} · {signal}"
            )
            leader_exception_flag = True

        else:
            action_name = "BUY1_DEMO_RELAXED"
            entry_reason = (
                f"모의완화 1차매수 · TOP5 · "
                f"종합 {combined:.1f} · {signal}"
            )
            leader_exception_flag = False

        act = _place_order(
            client,
            state,
            symbol,
            "buy",
            qty,
            entry_reason + " · " + bp_reason,
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
            "buying_power_reason": bp_reason,
            "buying_power": bp,
            **act,
        })

        # 실제 주문 성공일 때만 상태/누적금액 변경
        if act.get("status") == "ORDERED":
            state["positions"][symbol] = {
                "name": str(row.get("종목명", "")),
                "created_at": now.isoformat(),
                "buy_stage": 1,
                "avg_price": price,
                "expected_qty": qty,
                "leader_exception": leader_exception_flag,
                "take1_sent": False,
                "exit_pending": False,
            }
            state["daily_buy_amount"] += cost

    if not result["actions"]:
        result["message"] = (
            "실제 주문/DRY 액션 없음. "
            "diagnostics에서 종목별 SKIP 이유를 확인하세요."
        )

    save_state(state)
    result["state"] = state
    return result


# -------------------------------------------------------------------
# 미국 상태
# -------------------------------------------------------------------

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
    US_STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


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
    return {
        "NASD": "NAS",
        "NYSE": "NYS",
        "AMEX": "AMS",
    }.get(str(exchange).upper(), "NAS")


def _overseas_current_price(client, symbol: str, exchange: str = "NASD") -> float:
    try:
        raw = client.get(
            "/uapi/overseas-price/v1/quotations/price",
            "HHDFS00000300",
            {
                "AUTH": "",
                "EXCD": _us_quote_exchange(exchange),
                "SYMB": str(symbol).upper(),
            },
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


def _overseas_balance(client, exchange: str = "NASD") -> Dict[str, Any]:
    tr_id = "VTTS3012R" if getattr(client, "env", "demo") == "demo" else "TTTS3012R"
    return client.get(
        "/uapi/overseas-stock/v1/trading/inquire-balance",
        tr_id,
        {
            "CANO": client.account_no,
            "ACNT_PRDT_CD": client.product_code,
            "OVRS_EXCG_CD": exchange,
            "TR_CRCY_CD": "USD",
            "CTX_AREA_FK200": "",
            "CTX_AREA_NK200": "",
        },
    )


def _parse_overseas_holdings(balance_json: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    rows = (balance_json or {}).get("output1", []) or []
    if isinstance(rows, dict):
        rows = [rows]

    result = {}

    for r in rows:
        symbol = str(
            r.get("ovrs_pdno")
            or r.get("pdno")
            or r.get("ovrs_item_cd")
            or ""
        ).strip().upper()

        if not symbol:
            continue

        def num(*names):
            for name in names:
                try:
                    value = float(r.get(name, 0) or 0)
                    if value != 0:
                        return value
                except Exception:
                    pass
            return 0.0

        qty = int(num("ovrs_cblc_qty", "hldg_qty", "hold_qty"))
        if qty <= 0:
            continue

        result[symbol] = {
            "qty": qty,
            "avg_price": num("pchs_avg_pric", "avg_pric"),
            "current_price": num("now_pric2", "ovrs_nmix_prpr", "prpr"),
            "name": str(r.get("ovrs_item_name") or r.get("prdt_name") or symbol),
        }

    return result


def _us_stage_budget(config: AutoConfig, stage: int) -> float:
    weights = [config.buy1_pct, config.buy2_pct, config.buy3_pct]
    total = max(1, sum(weights))
    return float(config.us_per_stock_budget_usd) * weights[stage - 1] / total


def _us_stage_qty(config: AutoConfig, stage: int, price: float) -> int:
    return _calc_qty(_us_stage_budget(config, stage), price)


def _us_first_entry_qty(config: AutoConfig, price: float) -> int:
    qty = _us_stage_qty(config, 1, price)
    if qty > 0:
        return qty

    if (
        getattr(config, "allow_single_share_over_stage_budget", True)
        and price > 0
        and price <= float(config.us_per_stock_budget_usd)
    ):
        return 1

    return 0


def _to_float_us(v) -> float:
    try:
        return float(str(v).replace(",", "").strip() or "0")
    except Exception:
        return 0.0


def _to_int_us(v) -> int:
    try:
        return int(float(str(v).replace(",", "").strip() or "0"))
    except Exception:
        return 0


def _as_dict_list(value):
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [x for x in value if isinstance(x, dict)]
    return []


def _extract_us_cash_from_balance_response(raw: Dict[str, Any]) -> tuple[float, str, Dict[str, Any]]:
    """
    KIS 해외잔고 응답에서 '현금/외화 예수금/주문가능 외화' 계열만 추출합니다.
    평가금액·보유주식 평가액은 매수가능 현금으로 사용하지 않습니다.
    """
    if not isinstance(raw, dict) or str(raw.get("rt_cd", "0")) != "0":
        return 0.0, "", {}

    preferred = [
        "frcr_dncl_amt_2",
        "frcr_dncl_amt",
        "frcr_use_psbl_amt",
        "ord_psbl_frcr_amt",
        "frcr_ord_psbl_amt",
        "ovrs_ord_psbl_amt",
        "frcr_buy_psbl_amt",
        "buy_psbl_frcr_amt",
    ]

    seen = {}
    best_amt = 0.0
    best_key = ""

    for out_key in ("output", "output1", "output2", "output3"):
        for row in _as_dict_list(raw.get(out_key)):
            for key in preferred:
                if key in row:
                    val = _to_float_us(row.get(key))
                    seen[f"{out_key}.{key}"] = val
                    if val > best_amt:
                        best_amt = val
                        best_key = f"{out_key}.{key}"

    return best_amt, best_key, seen


def _safe_us_buy_qty_from_buying_power(
    client,
    symbol: str,
    target_amount_usd: float,
    current_price: float,
    config: AutoConfig,
    exchange: str = "NASD",
    allow_single_share: bool = False,
) -> tuple[int, Dict[str, Any], str]:
    """
    KIS 해외주식 매수가능금액을 먼저 조회하고 5% 안전여유를 적용한 뒤
    실제로 주문 가능한 수량만 반환합니다.

    주문가능금액이 부족하면 KIS 주문 API 자체를 호출하지 않습니다.
    """
    if current_price <= 0:
        return 0, {}, "미국 현재가가 0 이하"

    try:
        raw = client.overseas_buying_power_us(
            symbol=symbol,
            limit_price=current_price,
            exchange=exchange,
        )
    except Exception as e:
        return 0, {}, f"미국 매수가능금액 조회 실패: {type(e).__name__}: {e}"

    if not isinstance(raw, dict):
        return 0, {}, "미국 매수가능금액 응답 형식 오류"

    if str(raw.get("rt_cd", "0")) != "0":
        return 0, {"raw_msg": raw.get("msg1", "")}, (
            f"미국 매수가능금액 조회 거절: {raw.get('msg1', '')}"
        )

    output = raw.get("output", {}) or {}
    if isinstance(output, list):
        output = output[0] if output else {}
    if not isinstance(output, dict):
        return 0, {}, "미국 매수가능금액 output 형식 오류"

    # KIS 공식 샘플의 inquire_psamount 응답 필드.
    money_fields = {
        "ord_psbl_frcr_amt": _to_float_us(output.get("ord_psbl_frcr_amt", 0)),
        "ovrs_ord_psbl_amt": _to_float_us(output.get("ovrs_ord_psbl_amt", 0)),
        "echm_af_ord_psbl_amt": _to_float_us(output.get("echm_af_ord_psbl_amt", 0)),
        "frcr_ord_psbl_amt1": _to_float_us(output.get("frcr_ord_psbl_amt1", 0)),
    }
    qty_fields = {
        "max_ord_psbl_qty": _to_int_us(output.get("max_ord_psbl_qty", 0)),
        "ord_psbl_qty": _to_int_us(output.get("ord_psbl_qty", 0)),
        "echm_af_ord_psbl_qty": _to_int_us(output.get("echm_af_ord_psbl_qty", 0)),
        "ovrs_max_ord_psbl_qty": _to_int_us(output.get("ovrs_max_ord_psbl_qty", 0)),
    }

    available_usd = max(money_fields.values()) if money_fields else 0.0
    available_qty = max(qty_fields.values()) if qty_fields else 0

    # 모의투자에서 금액 필드가 0/공란이어도 '최대주문가능수량'이 정상으로
    # 내려오는 경우가 있습니다. 이때는 API가 계산한 수량을 우선 신뢰합니다.
    derived_from_qty = False
    if available_usd <= 0 and available_qty > 0:
        available_usd = float(available_qty) * float(current_price)
        derived_from_qty = True

    buffer_pct = float(getattr(config, "buying_power_buffer_pct", 5.0))
    buffer_ratio = max(0.0, min(0.50, buffer_pct / 100.0))
    buffered_usd = max(0.0, available_usd * (1.0 - buffer_ratio))

    target = max(0.0, float(target_amount_usd))
    usable_target = min(target, buffered_usd)
    qty = int(usable_target // current_price)

    # 1차 예산보다 비싼 고가종목 1주 허용 규칙은 유지하되,
    # 실제 주문가능금액(5% 여유 적용 후)이 1주 가격 이상일 때만 허용합니다.
    if (
        qty <= 0
        and allow_single_share
        and current_price <= float(config.us_per_stock_budget_usd)
        and buffered_usd >= current_price
    ):
        qty = 1

    if available_qty > 0:
        qty = min(qty, available_qty)

    # 종목당 한도 초과 방지
    max_by_stock = int(float(config.us_per_stock_budget_usd) // current_price)
    if max_by_stock > 0:
        qty = min(qty, max_by_stock)
    elif current_price > float(config.us_per_stock_budget_usd):
        qty = 0

    meta = {
        "available_usd": round(available_usd, 2),
        "buffer_pct": buffer_pct,
        "buffered_available_usd": round(buffered_usd, 2),
        "target_amount_usd": round(target, 2),
        "usable_target_usd": round(usable_target, 2),
        "api_max_qty": available_qty,
        "calculated_qty": qty,
        "currency": output.get("tr_crcy_cd", "USD"),
        "derived_from_qty": derived_from_qty,
        "api_money_fields": money_fields,
        "api_qty_fields": qty_fields,
    }

    if available_usd <= 0 and available_qty <= 0:
        # 모의서버의 inquire-psamount가 0/0을 반환하는 경우가 있어,
        # KIS 공식 해외주식 잔고 API 2종으로 한 번 더 확인합니다.
        fallback_sources = []
        fallback_amt = 0.0
        fallback_key = ""

        try:
            bal_raw = client.overseas_balance_us(exchange=exchange, currency="USD")
            bal_amt, bal_key, bal_seen = _extract_us_cash_from_balance_response(bal_raw)
            fallback_sources.append({
                "api": "inquire-balance",
                "amount": round(bal_amt, 2),
                "field": bal_key,
                "seen": bal_seen,
                "msg1": bal_raw.get("msg1", "") if isinstance(bal_raw, dict) else "",
            })
            if bal_amt > fallback_amt:
                fallback_amt, fallback_key = bal_amt, f"inquire-balance:{bal_key}"
        except Exception as e:
            fallback_sources.append({
                "api": "inquire-balance",
                "error": f"{type(e).__name__}: {e}",
            })

        if fallback_amt <= 0:
            try:
                present_raw = client.overseas_present_balance_us(foreign_currency=True)
                p_amt, p_key, p_seen = _extract_us_cash_from_balance_response(present_raw)
                fallback_sources.append({
                    "api": "inquire-present-balance",
                    "amount": round(p_amt, 2),
                    "field": p_key,
                    "seen": p_seen,
                    "msg1": present_raw.get("msg1", "") if isinstance(present_raw, dict) else "",
                })
                if p_amt > fallback_amt:
                    fallback_amt, fallback_key = p_amt, f"inquire-present-balance:{p_key}"
            except Exception as e:
                fallback_sources.append({
                    "api": "inquire-present-balance",
                    "error": f"{type(e).__name__}: {e}",
                })

        meta["fallback_balance_checks"] = fallback_sources

        if fallback_amt > 0:
            available_usd = fallback_amt
            buffered_usd = max(0.0, available_usd * (1.0 - buffer_ratio))
            usable_target = min(target, buffered_usd)
            qty = int(usable_target // current_price)

            # API 잔고 금액으로도 고가종목 1주 규칙은 실제 자금이 있을 때만 허용
            if (
                qty <= 0
                and allow_single_share
                and current_price <= float(config.us_per_stock_budget_usd)
                and buffered_usd >= current_price
            ):
                qty = 1

            max_by_stock = int(float(config.us_per_stock_budget_usd) // current_price)
            if max_by_stock > 0:
                qty = min(qty, max_by_stock)
            elif current_price > float(config.us_per_stock_budget_usd):
                qty = 0

            meta.update({
                "available_usd": round(available_usd, 2),
                "buffered_available_usd": round(buffered_usd, 2),
                "usable_target_usd": round(usable_target, 2),
                "calculated_qty": qty,
                "fallback_field": fallback_key,
            })

            if qty > 0:
                cost = qty * current_price
                return qty, meta, (
                    f"미국 잔고 API 보완확인({fallback_key}): "
                    f"${available_usd:.2f} → {buffer_pct:.0f}% 안전여유 후 "
                    f"${buffered_usd:.2f} · {qty}주 (${cost:.2f})"
                )

        return 0, meta, (
            "KIS 모의서버의 매수가능금액조회와 해외잔고조회에서 "
            "주문에 사용할 수 있는 USD 현금을 확인하지 못했습니다. "
            "강제로 $100,000을 가정하지 않고 주문을 중지했습니다."
        )

    if buffered_usd < current_price:
        return 0, meta, (
            f"미국 주문가능금액 부족: ${available_usd:.2f}의 "
            f"{100-buffer_pct:.0f}%=${buffered_usd:.2f} < 1주 ${current_price:.2f}"
        )

    if qty <= 0:
        return 0, meta, (
            f"미국 주문가능수량 0: 사용가능 ${buffered_usd:.2f} / "
            f"현재가 ${current_price:.2f}"
        )

    cost = qty * current_price
    source_text = "API 최대수량 기준" if derived_from_qty else "API 금액 기준"
    reason = (
        f"미국 매수가능 확인({source_text}): ${available_usd:.2f} → "
        f"{buffer_pct:.0f}% 안전여유 후 ${buffered_usd:.2f} · "
        f"{qty}주 (${cost:.2f})"
    )
    return qty, meta, reason


def _place_overseas_order(
    client,
    state,
    symbol: str,
    side: str,
    qty: int,
    price: float,
    reason: str,
    execute_orders: bool,
    exchange: str = "NASD",
):
    if qty <= 0:
        return {"status": "SKIP", "reason": "주문수량 0"}
    if price <= 0:
        return {"status": "SKIP", "reason": "주문가격 0"}

    if not execute_orders:
        _us_event(
            state,
            f"DRY_{side.upper()}",
            symbol,
            f"{qty}주 @ ${price:.2f} · {reason}",
        )
        return {
            "status": "DRY",
            "qty": qty,
            "price": price,
        }

    try:
        res = client.overseas_order_us(
            symbol=symbol,
            qty=qty,
            side=side,
            limit_price=price,
            exchange=exchange,
        )
    except Exception as e:
        _us_event(state, "ORDER_ERROR", symbol, repr(e))
        return {"status": "ERROR", "error": repr(e)}

    if _order_ok(res):
        state["daily_orders"] += 1
        _us_event(
            state,
            f"{side.upper()}_ORDER",
            symbol,
            f"{qty}주 @ ${price:.2f} · {reason} · {res}",
        )
        return {
            "status": "ORDERED",
            "qty": qty,
            "price": price,
            "response": res,
        }

    _us_event(state, "ORDER_REJECT", symbol, str(res))
    return {
        "status": "REJECT",
        "qty": qty,
        "price": price,
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
        "execute_orders": execute_orders,
        "actions": [],
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

    try:
        holdings = _parse_overseas_holdings(_overseas_balance(client))
    except Exception as e:
        holdings = {}
        result["balance_warning"] = repr(e)

    for symbol, pos in list(state["positions"].items()):
        actual = holdings.get(symbol, {})
        actual_qty = int(actual.get("qty", 0))

        if actual_qty > 0:
            pos["actual_qty"] = actual_qty
            if float(actual.get("avg_price", 0) or 0) > 0:
                pos["avg_price"] = float(actual["avg_price"])

        if execute_orders and actual_qty <= 0:
            state["positions"].pop(symbol, None)
            continue

        qty_for_manage = actual_qty if actual_qty > 0 else int(pos.get("expected_qty", 0))
        if qty_for_manage <= 0:
            continue

        price = _overseas_current_price(client, symbol)
        if price <= 0:
            continue

        avg = float(pos.get("avg_price") or actual.get("avg_price") or price)
        pnl = (price / avg - 1) * 100 if avg > 0 else 0.0

        if now.time() >= _clock(config.us_force_exit_time):
            act = _place_overseas_order(
                client, state, symbol, "sell", qty_for_manage, price,
                f"당일 강제청산 {config.us_force_exit_time} ET",
                execute_orders,
            )
            result["actions"].append(
                {"symbol": symbol, "action": "FORCE_SELL", "pnl": pnl, **act}
            )
            continue

        if pnl <= -abs(config.stop_loss_pct):
            act = _place_overseas_order(
                client, state, symbol, "sell", qty_for_manage, price,
                f"손절 {pnl:.2f}%",
                execute_orders,
            )
            result["actions"].append(
                {"symbol": symbol, "action": "STOP_LOSS", "pnl": pnl, **act}
            )
            continue

        if pnl >= config.take2_pct:
            act = _place_overseas_order(
                client, state, symbol, "sell", qty_for_manage, price,
                f"2차 익절 {pnl:.2f}%",
                execute_orders,
            )
            result["actions"].append(
                {"symbol": symbol, "action": "TAKE2", "pnl": pnl, **act}
            )
            continue

        if pnl >= config.take1_pct and not pos.get("take1_sent"):
            sell_qty = max(1, qty_for_manage // 2)
            act = _place_overseas_order(
                client, state, symbol, "sell", sell_qty, price,
                f"1차 익절 {pnl:.2f}%",
                execute_orders,
            )
            result["actions"].append(
                {"symbol": symbol, "action": "TAKE1", "pnl": pnl, **act}
            )
            if act["status"] in ("ORDERED", "DRY"):
                pos["take1_sent"] = True
            continue

        stage = int(pos.get("buy_stage", 1))

        if state["daily_orders"] >= config.max_daily_orders:
            continue

        if stage == 1 and pnl >= config.add2_trigger_pct:
            qty, us_bp, us_bp_reason = _safe_us_buy_qty_from_buying_power(
                client, symbol, _us_stage_budget(config, 2), price, config
            )
            cost = qty * price
            if (
                qty > 0
                and state["daily_buy_amount_usd"] + cost <= config.us_daily_budget_usd
            ):
                act = _place_overseas_order(
                    client, state, symbol, "buy", qty, price,
                    f"2차 분할매수 +{pnl:.2f}% · {us_bp_reason}",
                    execute_orders,
                )
                result["actions"].append(
                    {"symbol": symbol, "action": "BUY2", "pnl": pnl, **act}
                )
                if act["status"] in ("ORDERED", "DRY"):
                    pos["buy_stage"] = 2
                    pos["expected_qty"] = int(pos.get("expected_qty", 0)) + qty
                    state["daily_buy_amount_usd"] += cost
                continue

        if stage == 2 and pnl >= config.add3_trigger_pct:
            qty, us_bp, us_bp_reason = _safe_us_buy_qty_from_buying_power(
                client, symbol, _us_stage_budget(config, 3), price, config
            )
            cost = qty * price
            if (
                qty > 0
                and state["daily_buy_amount_usd"] + cost <= config.us_daily_budget_usd
            ):
                act = _place_overseas_order(
                    client, state, symbol, "buy", qty, price,
                    f"3차 분할매수 +{pnl:.2f}% · {us_bp_reason}",
                    execute_orders,
                )
                result["actions"].append(
                    {"symbol": symbol, "action": "BUY3", "pnl": pnl, **act}
                )
                if act["status"] in ("ORDERED", "DRY"):
                    pos["buy_stage"] = 3
                    pos["expected_qty"] = int(pos.get("expected_qty", 0)) + qty
                    state["daily_buy_amount_usd"] += cost

    if now.time() >= _clock(config.us_last_entry_time):
        result["message"] = f"{config.us_last_entry_time} ET 이후 미국 신규매수 금지"
        save_us_state(state)
        result["state"] = state
        return result

    if len(state["positions"]) >= config.max_positions:
        result["message"] = "미국 최대 보유종목 수 도달"
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

        symbol = str(
            row.get("종목코드", row.get("종목", ""))
        ).strip().upper()

        if not symbol or symbol in state["positions"]:
            continue

        signal = str(row.get("판정", row.get("종합신호", "")))
        try:
            combined = float(row.get("종합점수", 0) or 0)
        except Exception:
            combined = 0.0

        is_demo = (
            str(getattr(client, "env", "demo")).lower()
            == "demo"
        )

        us_demo_relaxed = (
            is_demo
            and bool(getattr(config, "demo_relaxed_entry_enabled", True))
            and combined >= float(
                getattr(config, "demo_min_combined_score", 50.0)
            )
        )

        us_normal_entry = (
            (
                not bool(config.require_green_signal)
                or "매수 후보" in signal
            )
            and combined >= float(config.min_combined_score)
        )

        if not (us_normal_entry or us_demo_relaxed):
            continue

        if symbol in holdings:
            continue

        price = _overseas_current_price(client, symbol)
        if price <= 0:
            result["actions"].append({
                "symbol": symbol,
                "action": "SKIP",
                "reason": "KIS 미국 현재가 조회 실패",
            })
            continue

        stage_budget = _us_stage_budget(config, 1)
        qty, us_bp, us_bp_reason = _safe_us_buy_qty_from_buying_power(
            client=client,
            symbol=symbol,
            target_amount_usd=stage_budget,
            current_price=price,
            config=config,
            allow_single_share=bool(
                getattr(config, "allow_single_share_over_stage_budget", True)
            ),
        )
        if qty <= 0:
            result["actions"].append({
                "symbol": symbol,
                "action": "SKIP_BUYING_POWER",
                "status": "SKIP",
                "current_price": price,
                "combined_score": combined,
                "buying_power": us_bp,
                "reason": us_bp_reason,
            })
            continue

        cost = qty * price

        if cost > config.us_per_stock_budget_usd:
            result["actions"].append({
                "symbol": symbol,
                "action": "SKIP",
                "reason": (
                    f"예상매수금액 ${cost:.2f}가 미국 종목당 한도 "
                    f"${config.us_per_stock_budget_usd:.2f}를 초과"
                ),
            })
            continue

        if state["daily_buy_amount_usd"] + cost > config.us_daily_budget_usd:
            continue

        high_price_single = qty == 1 and price > _us_stage_budget(config, 1)

        reason = (
            (
                f"모의 50점 기준 1차매수 · 종합점수 {combined:.1f} · {signal}"
                if us_demo_relaxed and not us_normal_entry
                else f"1차 분할매수 · 종합점수 {combined:.1f} · {signal}"
            )
        )
        if high_price_single:
            reason += (
                f" · 고가종목 1주 허용"
                f"(1차예산 ${_us_stage_budget(config, 1):.2f} < "
                f"주가 ${price:.2f} ≤ 종목한도 ${config.us_per_stock_budget_usd:.2f})"
            )

        reason += f" · {us_bp_reason}"

        act = _place_overseas_order(
            client, state, symbol, "buy", qty, price,
            reason,
            execute_orders,
        )
        result["actions"].append({
            "symbol": symbol,
            "action": "BUY1",
            "reason": reason,
            "buying_power": us_bp,
            **act,
        })

        if act["status"] in ("ORDERED", "DRY"):
            state["positions"][symbol] = {
                "name": str(row.get("종목명", symbol)),
                "created_at": now.isoformat(),
                "buy_stage": 1,
                "avg_price": price,
                "expected_qty": qty,
                "take1_sent": False,
            }
            state["daily_buy_amount_usd"] += cost

    if not result["actions"]:
        result["message"] = (
            "🟢 매수 후보 + 종합점수 기준을 통과한 미국 신규매수 후보 없음"
        )

    save_us_state(state)
    result["state"] = state
    return result

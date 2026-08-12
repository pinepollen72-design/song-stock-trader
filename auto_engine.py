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

    min_combined_score: float = 65.0
    require_green_signal: bool = True

    demo_relaxed_entry_enabled: bool = True
    demo_min_combined_score: float = 40.0

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
    # 원화 예산을 환율로 USD로 자동 환산합니다.
    us_daily_budget_krw: int = 10_000_000
    us_per_stock_budget_krw: int = 10_000_000
    usd_krw_rate: float = 1400.0
    us_daily_budget_usd: float = 0.0
    us_per_stock_budget_usd: float = 0.0

    # 10:00 KST(서머타임 기준 09:00 ET)부터 후보 준비,
    # 실제 신규매수는 미국 정규장 09:30 ET부터.
    us_last_entry_time: str = "15:30"
    us_force_exit_time: str = "15:50"

    # 모의투자에서는 실제 해외잔고를 자동 추적/청산 대상으로 복구.
    us_adopt_existing_demo_holdings: bool = True


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



def sync_us_budget_from_krw(config: AutoConfig) -> Dict[str, float]:
    """
    미국장 예산을 원화 기준에서 USD로 환산합니다.
    USD_KRW 환율은 app/worker 환경변수 또는 UI에서 설정합니다.
    """
    rate = float(getattr(config, "usd_krw_rate", 0) or 0)
    if rate <= 0:
        rate = 1400.0
        config.usd_krw_rate = rate

    daily_krw = int(getattr(config, "us_daily_budget_krw", 0) or 0)
    per_stock_krw = int(getattr(config, "us_per_stock_budget_krw", 0) or 0)

    if daily_krw > 0:
        config.us_daily_budget_usd = round(daily_krw / rate, 2)

    if per_stock_krw > 0:
        config.us_per_stock_budget_usd = round(per_stock_krw / rate, 2)

    return {
        "usd_krw_rate": rate,
        "daily_budget_krw": daily_krw,
        "per_stock_budget_krw": per_stock_krw,
        "daily_budget_usd": float(config.us_daily_budget_usd),
        "per_stock_budget_usd": float(config.us_per_stock_budget_usd),
    }


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


def _overseas_current_price(
    client,
    symbol: str,
    exchange: str = "NASD",
) -> tuple[float, str]:
    try:
        raw = client.overseas_price_us(
            symbol=symbol,
            exchange=exchange,
        )
    except Exception as e:
        return 0.0, (
            f"KIS 미국 현재가 조회 예외: "
            f"{type(e).__name__}: {e}"
        )

    out = (raw or {}).get("output", {}) or {}

    for key in (
        "last",
        "last_price",
        "ovrs_nmix_prpr",
        "stck_prpr",
    ):
        try:
            value = float(out.get(key, 0) or 0)
            if value > 0:
                return value, ""
        except Exception:
            pass

    return 0.0, (
        f"KIS 미국 현재가 0/응답이상: "
        f"{str(raw)[:500]}"
    )


def _overseas_balance(
    client,
    exchange: str = "NASD",
) -> Dict[str, Any]:
    return client.overseas_balance_us(exchange=exchange)


def _parse_overseas_holdings(
    balance_json: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
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
                    raw = str(r.get(name, 0) or 0).replace(",", "")
                    value = float(raw)
                    if value != 0:
                        return value
                except Exception:
                    pass
            return 0.0

        qty = int(
            num(
                "ovrs_cblc_qty",
                "hldg_qty",
                "hold_qty",
                "ord_psbl_qty",
            )
        )

        if qty <= 0:
            continue

        result[symbol] = {
            "qty": qty,
            "avg_price": num(
                "pchs_avg_pric",
                "avg_pric",
                "avg_unit_pchs_pric",
            ),
            "current_price": num(
                "now_pric2",
                "ovrs_nmix_prpr",
                "prpr",
            ),
            "name": str(
                r.get("ovrs_item_name")
                or r.get("prdt_name")
                or symbol
            ),
        }

    return result


def _extract_available_usd(balance_json: Dict[str, Any]) -> float:
    """
    KIS 해외잔고 응답은 환경/버전에 따라 가용 외화 필드명이 다를 수 있어
    알려진 후보 필드를 안전하게 확인합니다.
    값을 찾지 못하면 0을 반환하고, 엔진은 설정 예산만으로 제한합니다.
    """
    containers = []

    for key in ("output2", "output3", "output"):
        value = (balance_json or {}).get(key)
        if isinstance(value, dict):
            containers.append(value)
        elif isinstance(value, list):
            containers.extend(
                x for x in value
                if isinstance(x, dict)
            )

    keys = (
        "ovrs_ord_psbl_amt",
        "frcr_buy_amt_smtl1",
        "frcr_buy_amt_smtl",
        "frcr_use_psbl_amt",
        "ord_psbl_frcr_amt",
        "frcr_ord_psbl_amt",
    )

    for container in containers:
        for key in keys:
            try:
                value = float(
                    str(container.get(key, 0) or 0)
                    .replace(",", "")
                )
                if value > 0:
                    return value
            except Exception:
                pass

    return 0.0


def _us_stage_budget(
    config: AutoConfig,
    stage: int,
) -> float:
    weights = [
        int(config.buy1_pct),
        int(config.buy2_pct),
        int(config.buy3_pct),
    ]
    total = max(1, sum(weights))

    return (
        float(config.us_per_stock_budget_usd)
        * weights[stage - 1]
        / total
    )


def _us_safe_qty(
    config: AutoConfig,
    stage: int,
    price: float,
    available_usd: float = 0.0,
) -> tuple[int, float, str]:
    if price <= 0:
        return 0, 0.0, "미국 현재가 0"

    stage_budget = _us_stage_budget(config, stage)

    # 실제 가용 USD를 확인한 경우에는 5% 안전여유 적용
    usable = stage_budget
    if available_usd > 0:
        buffered = available_usd * (
            1.0
            - max(
                0.0,
                min(
                    0.50,
                    float(config.buying_power_buffer_pct) / 100.0,
                ),
            )
        )
        usable = min(stage_budget, buffered)

    qty = int(usable // price)

    if qty <= 0:
        return (
            0,
            usable,
            f"주문가능 수량 0: 1주 ${price:.2f} / "
            f"이번 차수 사용가능 ${usable:.2f}",
        )

    return (
        qty,
        usable,
        f"이번 차수 USD 예산 ${stage_budget:.2f} / "
        f"사용가능 ${usable:.2f} / {qty}주",
    )


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
        return {
            "status": "SKIP",
            "qty": 0,
            "reason": "주문수량 0",
        }

    if price <= 0:
        return {
            "status": "SKIP",
            "qty": int(qty),
            "reason": "주문가격 0",
        }

    if not execute_orders:
        # 국내와 동일하게 DRY는 상태/누적금액을 절대 변경하지 않습니다.
        return {
            "status": "DRY",
            "qty": int(qty),
            "price": float(price),
            "reason": reason,
        }

    try:
        res = client.overseas_order_us(
            symbol=symbol,
            qty=int(qty),
            side=side,
            limit_price=float(price),
            exchange=exchange,
        )
    except Exception as e:
        _us_event(state, "ORDER_ERROR", symbol, repr(e))
        return {
            "status": "ERROR",
            "qty": int(qty),
            "price": float(price),
            "error": repr(e),
        }

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
            "qty": int(qty),
            "price": float(price),
            "response": res,
        }

    _us_event(
        state,
        "ORDER_REJECT",
        symbol,
        str(res),
    )

    return {
        "status": "REJECT",
        "qty": int(qty),
        "price": float(price),
        "msg_cd": (
            res.get("msg_cd", "")
            if isinstance(res, dict)
            else ""
        ),
        "msg1": (
            res.get("msg1", "")
            if isinstance(res, dict)
            else ""
        ),
        "response": res,
    }


def _us_diag(
    result: Dict[str, Any],
    symbol: str,
    action: str,
    reason: str,
    **extra,
):
    row = {
        "symbol": symbol,
        "action": action,
        "reason": reason,
        **extra,
    }
    result.setdefault("diagnostics", []).append(row)
    return row


def run_overseas_cycle(
    client,
    leader_df: pd.DataFrame,
    config: AutoConfig,
    execute_orders: bool = False,
    source: str = "APP",
) -> Dict[str, Any]:
    sync = sync_us_budget_from_krw(config)

    state = load_us_state()
    now = _now_et()

    result = {
        "time": now.isoformat(timespec="seconds"),
        "execute_orders": bool(execute_orders),
        "source": source,
        "actions": [],
        "diagnostics": [],
        "budget": sync,
        "state": state,
    }

    if now.weekday() >= 5:
        result["message"] = "미국 주말: 주문 없음"
        save_us_state(state)
        return result

    # 09:00~09:29 ET는 후보 준비/진단만.
    if not (dtime(9, 0) <= now.time() < dtime(16, 5)):
        result["message"] = (
            "미국 관리시간 외(09:00~16:05 ET)"
        )
        save_us_state(state)
        return result

    order_window_open = (
        dtime(9, 30) <= now.time() < dtime(16, 0)
    )

    if execute_orders and not order_window_open:
        execute_orders = False
        result["execute_orders"] = False
        result["order_gate_message"] = (
            "미국 정규장 주문시간(09:30~16:00 ET) 밖이라 "
            "후보분석/진단만 수행합니다."
        )

    raw_balance = {}
    try:
        raw_balance = _overseas_balance(client)
        holdings = _parse_overseas_holdings(raw_balance)
    except Exception as e:
        holdings = {}
        result["balance_warning"] = (
            f"{type(e).__name__}: {e}"
        )

    available_usd = _extract_available_usd(raw_balance)
    result["available_usd_detected"] = round(
        available_usd,
        2,
    )

    # 모의투자는 실제 잔고를 자동 추적 복구.
    # 실전에는 절대 적용하지 않음.
    if (
        str(getattr(client, "env", "demo")).lower() == "demo"
        and bool(config.us_adopt_existing_demo_holdings)
    ):
        for symbol, actual in holdings.items():
            if symbol in state["positions"]:
                continue

            state["positions"][symbol] = {
                "name": actual.get("name", symbol),
                "created_at": now.isoformat(),
                "buy_stage": 1,
                "avg_price": float(
                    actual.get("avg_price", 0) or 0
                ),
                "actual_qty": int(actual.get("qty", 0)),
                "adopted_existing": True,
                "take1_sent": False,
            }

            _us_diag(
                result,
                symbol,
                "ADOPT_EXISTING_DEMO",
                f"미국 모의계좌 보유 "
                f"{int(actual.get('qty', 0))}주 추적 복구",
            )

    # ------------------------------------------------------
    # 보유종목 관리
    # ------------------------------------------------------
    for symbol, pos in list(
        state["positions"].items()
    ):
        actual = holdings.get(symbol, {})
        actual_qty = int(actual.get("qty", 0))

        if actual_qty <= 0:
            created = str(pos.get("created_at", "") or "")
            try:
                age = (
                    now - datetime.fromisoformat(created)
                )
            except Exception:
                age = timedelta(minutes=10)

            if age > timedelta(minutes=5):
                state["positions"].pop(symbol, None)
                _us_diag(
                    result,
                    symbol,
                    "DROP_TRACKING",
                    "실제 미국잔고에 보유수량이 없어 추적 제거",
                )
            continue

        pos["actual_qty"] = actual_qty

        if float(actual.get("avg_price", 0) or 0) > 0:
            pos["avg_price"] = float(actual["avg_price"])

        price, price_err = _overseas_current_price(
            client,
            symbol,
        )

        # 현재가 API 실패 시 잔고 현재가 fallback
        if price <= 0:
            fallback = float(
                actual.get("current_price", 0) or 0
            )
            if fallback > 0:
                price = fallback
                _us_diag(
                    result,
                    symbol,
                    "PRICE_FALLBACK_BALANCE",
                    (
                        f"미국 현재가 API 실패 → 잔고 현재가 "
                        f"${price:.2f} 사용 / {price_err}"
                    ),
                )

        # 강제청산은 15:50 ET부터.
        # 지정가 주문 구조이므로 현재가가 있어야 주문 가능.
        if now.time() >= _clock(config.us_force_exit_time):
            if now.time() < dtime(16, 0):
                if price <= 0:
                    _us_diag(
                        result,
                        symbol,
                        "FORCE_EXIT_PRICE_MISSING",
                        price_err,
                    )
                    continue

                act = _place_overseas_order(
                    client,
                    state,
                    symbol,
                    "sell",
                    actual_qty,
                    price,
                    f"미국 당일 강제청산 "
                    f"{config.us_force_exit_time} ET",
                    execute_orders,
                )

                result["actions"].append({
                    "symbol": symbol,
                    "action": "FORCE_SELL",
                    **act,
                })
                continue

            _us_diag(
                result,
                symbol,
                "MISSED_FORCE_EXIT",
                (
                    f"{config.us_force_exit_time} ET 강제청산 이후이며 "
                    f"16:00 ET 정규장 종료. 실제 보유 {actual_qty}주"
                ),
            )
            continue

        if price <= 0:
            _us_diag(
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
            act = _place_overseas_order(
                client,
                state,
                symbol,
                "sell",
                actual_qty,
                price,
                f"미국 손절 {pnl:.2f}%",
                execute_orders,
            )
            result["actions"].append({
                "symbol": symbol,
                "action": "STOP_LOSS",
                "pnl": round(pnl, 3),
                **act,
            })
            continue

        if pnl >= float(config.take2_pct):
            act = _place_overseas_order(
                client,
                state,
                symbol,
                "sell",
                actual_qty,
                price,
                f"미국 2차 익절 {pnl:.2f}%",
                execute_orders,
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
            act = _place_overseas_order(
                client,
                state,
                symbol,
                "sell",
                sell_qty,
                price,
                f"미국 1차 익절 {pnl:.2f}%",
                execute_orders,
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

        # 15:30 ET 이후에는 추가매수 금지
        if now.time() >= _clock(config.us_last_entry_time):
            continue

        stage = int(pos.get("buy_stage", 1))

        if stage == 1 and pnl >= float(config.add2_trigger_pct):
            qty, usable, why = _us_safe_qty(
                config,
                2,
                price,
                available_usd,
            )
            cost = qty * price

            if (
                qty > 0
                and state["daily_buy_amount_usd"] + cost
                <= float(config.us_daily_budget_usd)
            ):
                act = _place_overseas_order(
                    client,
                    state,
                    symbol,
                    "buy",
                    qty,
                    price,
                    f"미국 2차 분할매수 +{pnl:.2f}% · {why}",
                    execute_orders,
                )
                result["actions"].append({
                    "symbol": symbol,
                    "action": "BUY2",
                    "pnl": round(pnl, 3),
                    "usable_usd": round(usable, 2),
                    **act,
                })

                if act.get("status") == "ORDERED":
                    pos["buy_stage"] = 2
                    state["daily_buy_amount_usd"] += cost
                continue

        if stage == 2 and pnl >= float(config.add3_trigger_pct):
            qty, usable, why = _us_safe_qty(
                config,
                3,
                price,
                available_usd,
            )
            cost = qty * price

            if (
                qty > 0
                and state["daily_buy_amount_usd"] + cost
                <= float(config.us_daily_budget_usd)
            ):
                act = _place_overseas_order(
                    client,
                    state,
                    symbol,
                    "buy",
                    qty,
                    price,
                    f"미국 3차 분할매수 +{pnl:.2f}% · {why}",
                    execute_orders,
                )
                result["actions"].append({
                    "symbol": symbol,
                    "action": "BUY3",
                    "pnl": round(pnl, 3),
                    "usable_usd": round(usable, 2),
                    **act,
                })

                if act.get("status") == "ORDERED":
                    pos["buy_stage"] = 3
                    state["daily_buy_amount_usd"] += cost
                continue

    # ------------------------------------------------------
    # 신규매수
    # ------------------------------------------------------
    if now.time() < dtime(9, 30):
        result["message"] = (
            "미국장 준비시간: 후보/잔고 점검 중 · "
            "09:30 ET부터 실제 신규매수 가능"
        )
        save_us_state(state)
        result["state"] = state
        return result

    if now.time() >= _clock(config.us_last_entry_time):
        result["message"] = (
            f"{config.us_last_entry_time} ET 이후 "
            f"미국 신규/추가매수 금지 · "
            f"{config.us_force_exit_time} ET부터 강제청산"
        )
        save_us_state(state)
        result["state"] = state
        return result

    if leader_df is None or leader_df.empty:
        result["message"] = "미국 TOP5 데이터 없음"
        save_us_state(state)
        result["state"] = state
        return result

    is_demo = (
        str(getattr(client, "env", "demo")).lower()
        == "demo"
    )

    for _, row in leader_df.head(5).iterrows():
        if len(state["positions"]) >= int(config.max_positions):
            break

        symbol = str(
            row.get("종목코드")
            or row.get("종목")
            or ""
        ).strip().upper()

        if not symbol:
            continue

        if symbol in state["positions"]:
            _us_diag(
                result,
                symbol,
                "SKIP_ALREADY_TRACKED",
                "미국 자동매매 추적중",
            )
            continue

        if symbol in holdings:
            _us_diag(
                result,
                symbol,
                "SKIP_ALREADY_HELD",
                f"미국 모의계좌 보유중: "
                f"{holdings[symbol].get('qty', 0)}주",
            )
            continue

        signal = str(
            row.get("판정")
            or row.get("종합신호")
            or ""
        )
        combined = float(
            row.get("종합점수", 0) or 0
        )

        normal_entry = (
            (
                not bool(config.require_green_signal)
                or "매수 후보" in signal
            )
            and combined >= float(config.min_combined_score)
        )

        demo_relaxed = (
            is_demo
            and bool(config.demo_relaxed_entry_enabled)
            and combined >= float(config.demo_min_combined_score)
        )

        if not (normal_entry or demo_relaxed):
            _us_diag(
                result,
                symbol,
                "SKIP_SCORE",
                (
                    f"미국 점수 미달: 종합 {combined:.1f}, "
                    f"신호 {signal}, "
                    f"모의기준 {config.demo_min_combined_score:.1f}"
                ),
            )
            continue

        price, price_err = _overseas_current_price(
            client,
            symbol,
        )

        if price <= 0:
            _us_diag(
                result,
                symbol,
                "SKIP_PRICE_LOOKUP",
                price_err,
            )
            continue

        qty, usable, why = _us_safe_qty(
            config,
            1,
            price,
            available_usd,
        )

        if qty <= 0:
            _us_diag(
                result,
                symbol,
                "SKIP_BUYING_POWER",
                why,
            )
            continue

        cost = qty * price

        if (
            state["daily_buy_amount_usd"] + cost
            > float(config.us_daily_budget_usd)
        ):
            _us_diag(
                result,
                symbol,
                "SKIP_DAILY_BUDGET",
                (
                    f"미국 누적 ${state['daily_buy_amount_usd']:.2f} + "
                    f"예상 ${cost:.2f} > "
                    f"일일한도 ${config.us_daily_budget_usd:.2f}"
                ),
            )
            continue

        if state["daily_orders"] >= int(config.max_daily_orders):
            _us_diag(
                result,
                symbol,
                "SKIP_DAILY_ORDERS",
                f"미국 일일주문 "
                f"{config.max_daily_orders}회 도달",
            )
            break

        if normal_entry:
            action_name = "BUY1"
            reason = (
                f"미국 정상 1차매수 · "
                f"종합 {combined:.1f} · {signal} · {why}"
            )
        else:
            action_name = "BUY1_DEMO_RELAXED"
            reason = (
                f"미국 모의완화 1차매수 · "
                f"종합 {combined:.1f} · {signal} · {why}"
            )

        act = _place_overseas_order(
            client,
            state,
            symbol,
            "buy",
            qty,
            price,
            reason,
            execute_orders,
        )

        result["actions"].append({
            "symbol": symbol,
            "action": action_name,
            "current_price": round(price, 4),
            "estimated_cost_usd": round(cost, 2),
            "combined_score": combined,
            "fx_rate": float(config.usd_krw_rate),
            **act,
        })

        # 실제 주문 성공일 때만 상태/누적USD 변경
        if act.get("status") == "ORDERED":
            state["positions"][symbol] = {
                "name": str(
                    row.get("종목명", symbol)
                ),
                "created_at": now.isoformat(),
                "buy_stage": 1,
                "avg_price": price,
                "expected_qty": qty,
                "take1_sent": False,
            }
            state["daily_buy_amount_usd"] += cost

    if not result["actions"]:
        result["message"] = (
            "미국 실제 주문/DRY 액션 없음. "
            "diagnostics에서 종목별 SKIP 이유를 확인하세요."
        )

    save_us_state(state)
    result["state"] = state
    return result

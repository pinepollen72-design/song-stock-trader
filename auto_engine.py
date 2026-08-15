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


def _resolve_state_dir() -> Path:
    """
    Worker와 같은 영구 상태 경로를 사용합니다.

    우선순위:
    1) SONG_TRADER_STATE_DIR
    2) Railway Volume mount path + /song_trader_v2
    3) 로컬/비영구 fallback /tmp/song_trader_v2
    """
    explicit = os.getenv("SONG_TRADER_STATE_DIR", "").strip()
    if explicit:
        return Path(explicit)

    railway_volume = os.getenv("RAILWAY_VOLUME_MOUNT_PATH", "").strip()
    if railway_volume:
        return Path(railway_volume) / "song_trader_v2"

    return Path("/tmp/song_trader_v2")


STATE_DIR = _resolve_state_dir()
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

    # 국내 단타 안전/속도 설정
    kr_add2_trigger_pct: float = 0.40
    kr_profit_guard_trigger_pct: float = 1.20
    kr_profit_guard_drawdown_pct: float = 0.80
    kr_signal_max_age_seconds: int = 180
    kr_new_entry_top_n: int = 5
    kr_require_momentum_confirm: bool = True

    # 미국 단타 안전/속도 설정
    us_add2_trigger_pct: float = 0.40
    us_profit_guard_trigger_pct: float = 1.20
    us_profit_guard_drawdown_pct: float = 0.80
    us_buy_limit_buffer_pct: float = 0.15
    us_sell_limit_buffer_pct: float = 0.15
    us_signal_max_age_seconds: int = 180
    us_new_entry_top_n: int = 5
    us_require_momentum_confirm: bool = True

    # 미국 C 전략: BUY2 B_STRICT + 60분 약화/정체 조기청산 + 재진입 제어
    us_buy2_strict_trigger_pct: float = 0.80
    us_buy2_min_hold_minutes: float = 5.0
    us_buy2_max_rank: int = 3
    us_buy2_min_score: float = 70.0
    us_buy2_require_recent5_positive: bool = True
    us_buy2_require_recent10_positive: bool = True
    us_buy2_require_relative_strength_positive: bool = True

    us_early_exit_enabled: bool = True
    us_early_exit_min_hold_minutes: float = 60.0
    us_early_exit_loss_threshold_pct: float = -0.50
    us_early_exit_weak_points: int = 5
    us_stagnant_exit_enabled: bool = True
    us_stagnant_exit_pnl_max_pct: float = 0.20
    us_stagnant_exit_score_max: float = 50.0
    us_stagnant_exit_momentum_abs_max: float = 0.05

    us_ban_same_symbol_after_early_exit: bool = True
    us_pause_after_early_exits_count: int = 2
    us_pause_new_entries_minutes: float = 60.0

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
    row = {
        "symbol": symbol,
        "action": action,
        "reason_code": action,
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


def _clear_order_lock(symbol: str, side: str | None = None) -> None:
    """체결/잔고반영이 확인된 주문의 짧은 중복잠금을 해제한다."""
    try:
        lock = _read_order_lock()
        if not lock:
            return
        if str(lock.get("symbol")) != str(symbol):
            return
        if side is not None and str(lock.get("side")) != str(side):
            return
        if ORDER_LOCK_FILE.exists():
            ORDER_LOCK_FILE.unlink()
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
    client,
    state,
    symbol,
    side,
    qty,
    reason,
    execute_orders,
    source="APP",
    duplicate_guard_seconds=90,
    reason_code="ORDER",
):
    base = {
        "qty": int(max(0, qty)),
        "reason": reason,
        "reason_code": reason_code,
    }

    if qty <= 0:
        return {
            **base,
            "status": "SKIP",
            "qty": 0,
            "reason": "주문수량 0",
        }
    if not execute_orders:
        return {**base, "status": "DRY"}

    hit, why = _duplicate_guard_hit(symbol, side, duplicate_guard_seconds)
    if hit:
        return {
            **base,
            "status": "DUPLICATE_BLOCKED",
            "reason": why,
            "reason_code": "SKIP_DUPLICATE_ORDER",
        }

    _write_order_lock(symbol, side, source)
    try:
        res = client.domestic_order(symbol, int(qty), side, market_order=True)
    except Exception as e:
        _event(state, "ORDER_ERROR", symbol, repr(e))
        return {
            **base,
            "status": "ERROR",
            "error": repr(e),
        }

    if _order_ok(res):
        state["daily_orders"] += 1
        _event(state, f"{str(side).upper()}_ORDER", symbol, f"{qty}주 · {reason} · {res}")
        return {
            **base,
            "status": "ORDERED",
            "response": res,
        }

    msg_cd = res.get("msg_cd", "") if isinstance(res, dict) else ""
    msg1 = res.get("msg1", "") if isinstance(res, dict) else ""
    _event(state, "ORDER_REJECT", symbol, f"{msg_cd} {msg1} {res}")
    return {
        **base,
        "status": "REJECT",
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
            "buy_stage": 2,
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
        "reason_code": act.get("reason_code", action),
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
        if action in ("TAKE1", "PROFIT_GUARD1"):
            pos["take1_sent"] = True
        if action in ("TAKE2", "PROFIT_GUARD2", "STOP_LOSS", "FORCE_SELL") and actual_qty <= 0:
            pos["exit_confirmed"] = True

        pos.pop("pending_order", None)
        pos["last_confirmed_at"] = now.isoformat()
        _clear_order_lock(symbol, side)
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


def _kr_row_float(row, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except Exception:
        return float(default)


def _kr_row_bool(row, key: str, default: bool = False) -> bool:
    v = row.get(key, default)
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on", "상승")


def _kr_signal_fresh(row, now: datetime, config: AutoConfig) -> bool:
    raw = str(row.get("스캔시각", "") or "").strip()
    if not raw:
        return True
    try:
        ts = datetime.fromisoformat(raw)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=KST)
        age = abs((now - ts.astimezone(KST)).total_seconds())
        return age <= max(30, int(config.kr_signal_max_age_seconds))
    except Exception:
        return False


def _kr_momentum_ok(row, now: datetime, config: AutoConfig) -> bool:
    if row is None:
        return False
    if not _kr_signal_fresh(row, now, config):
        return False
    signal = str(row.get("판정", row.get("종합신호", "")))
    if "매수 후보" not in signal:
        return False
    if _kr_row_bool(row, "모멘텀약화", False):
        return False
    if _kr_row_float(row, "종합점수", 0.0) < float(config.min_combined_score):
        return False
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

    leader_rows = {}
    if leader_df is not None and not leader_df.empty:
        for _, row in leader_df.iterrows():
            sym = str(row.get("종목코드", "")).zfill(6)
            if len(sym) == 6 and sym.isdigit():
                leader_rows[sym] = row

    for symbol, pos in list(state["positions"].items()):
        actual = holdings.get(symbol, {})
        actual_qty = int(actual.get("qty", 0))
        sellable_qty = int(actual.get("sellable_qty", actual_qty) or 0)

        if _confirm_or_wait_pending(result, symbol, pos, actual_qty, now, config):
            continue

        if actual_qty <= 0:
            if pos.get("exit_confirmed"):
                state["positions"].pop(symbol, None)
                _diag(result, symbol, "EXIT_CONFIRMED", "KIS 실제 보유 0주 → 청산 완료", after_qty=0)
                continue

            created = str(pos.get("created_at", "") or "")
            try:
                age = now - datetime.fromisoformat(created)
            except Exception:
                age = timedelta(minutes=10)

            if age > timedelta(minutes=5):
                state["positions"].pop(symbol, None)
                _diag(result, symbol, "DROP_TRACKING", "KIS 실제 보유수량 0주 → 추적 종료", after_qty=0)
            continue

        pos["actual_qty"] = actual_qty
        pos["sellable_qty"] = sellable_qty
        if float(actual.get("avg_price", 0) or 0) > 0:
            pos["avg_price"] = float(actual["avg_price"])

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
                    reason_code="SELL_FORCE_EXIT",
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
        prev_peak = float(pos.get("peak_pnl", pnl) or pnl)
        peak_pnl = max(prev_peak, pnl)
        pos["peak_pnl"] = round(peak_pnl, 4)
        drawdown_from_peak = max(0.0, peak_pnl - pnl)

        if pnl <= -abs(float(config.stop_loss_pct)):
            order_qty = min(actual_qty, sellable_qty) if sellable_qty > 0 else 0
            if order_qty <= 0:
                _diag(result, symbol, "WAIT_SELLABLE_QTY", f"손절 조건이나 매도가능수량 0주 / 보유 {actual_qty}주")
                continue
            act = _place_order(
                client, state, symbol, "sell", order_qty,
                f"손절 {pnl:.2f}% · KIS 실제 매도가능수량 기준",
                execute_orders, source, config.duplicate_guard_seconds,
                reason_code="SELL_STOP_LOSS",
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
                reason_code="SELL_TAKE_PROFIT_1",
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

        if pnl >= float(config.take2_pct) and pos.get("take1_sent"):
            order_qty = min(actual_qty, sellable_qty) if sellable_qty > 0 else 0
            if order_qty <= 0:
                _diag(result, symbol, "WAIT_SELLABLE_QTY", f"2차 익절 조건이나 매도가능수량 0주 / 보유 {actual_qty}주")
                continue
            act = _place_order(
                client, state, symbol, "sell", order_qty,
                f"2차 익절 {pnl:.2f}% · 남은 수량 전량",
                execute_orders, source, config.duplicate_guard_seconds,
                reason_code="SELL_TAKE_PROFIT_2",
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

        if (
            peak_pnl >= float(config.kr_profit_guard_trigger_pct)
            and drawdown_from_peak >= float(config.kr_profit_guard_drawdown_pct)
        ):
            if not pos.get("take1_sent"):
                order_qty = max(1, actual_qty // 2)
                order_qty = min(order_qty, sellable_qty) if sellable_qty > 0 else 0
                if order_qty > 0:
                    act = _place_order(
                        client, state, symbol, "sell", order_qty,
                        (
                            f"수익보호 1차 · 최고 +{peak_pnl:.2f}% → 현재 {pnl:.2f}% "
                            f"({drawdown_from_peak:.2f}%p 되밀림)"
                        ),
                        execute_orders, source, config.duplicate_guard_seconds,
                        reason_code="SELL_PROFIT_GUARD_1",
                    )
                    result["actions"].append({
                        "symbol": symbol,
                        "name": actual.get("name") or pos.get("name", symbol),
                        "action": "PROFIT_GUARD1",
                        "pnl": round(pnl, 3),
                        "peak_pnl": round(peak_pnl, 3),
                        "drawdown_from_peak": round(drawdown_from_peak, 3),
                        "qty": order_qty,
                        "before_qty": actual_qty,
                        "after_qty": max(0, actual_qty - order_qty),
                        **act,
                    })
                    if act.get("status") == "DRY":
                        pos["take1_sent"] = True
                    else:
                        _set_kr_pending(pos, now, "PROFIT_GUARD1", "sell", order_qty, actual_qty, act)
                    continue
            else:
                order_qty = min(actual_qty, sellable_qty) if sellable_qty > 0 else 0
                if order_qty > 0:
                    act = _place_order(
                        client, state, symbol, "sell", order_qty,
                        (
                            f"수익보호 2차 · 최고 +{peak_pnl:.2f}% → 현재 {pnl:.2f}% "
                            f"({drawdown_from_peak:.2f}%p 되밀림)"
                        ),
                        execute_orders, source, config.duplicate_guard_seconds,
                        reason_code="SELL_PROFIT_GUARD_2",
                    )
                    result["actions"].append({
                        "symbol": symbol,
                        "name": actual.get("name") or pos.get("name", symbol),
                        "action": "PROFIT_GUARD2",
                        "pnl": round(pnl, 3),
                        "peak_pnl": round(peak_pnl, 3),
                        "drawdown_from_peak": round(drawdown_from_peak, 3),
                        "qty": order_qty,
                        "before_qty": actual_qty,
                        "after_qty": max(0, actual_qty - order_qty),
                        **act,
                    })
                    _set_kr_pending(pos, now, "PROFIT_GUARD2", "sell", order_qty, actual_qty, act)
                    continue

        if now.time() >= _clock(config.last_entry_time):
            continue

        if pos.get("leader_exception"):
            continue

        parts = _split_amounts(config)
        stage = int(pos.get("buy_stage", 1))

        if stage == 1 and pnl >= float(config.kr_add2_trigger_pct):
            leader_row = leader_rows.get(symbol)
            if bool(config.kr_require_momentum_confirm) and not _kr_momentum_ok(leader_row, now, config):
                _diag(
                    result, symbol, "SKIP_BUY2_MOMENTUM",
                    f"2차매수 보류: +{pnl:.2f}%지만 최신 단기 모멘텀 확인 실패",
                )
                continue

            qty, bp, bp_reason = _safe_buy_qty_from_buying_power(
                client, symbol, parts[1], price, config
            )
            cost = int(qty * price)
            if qty > 0 and state["daily_buy_amount"] + cost <= config.daily_budget:
                act = _place_order(
                    client, state, symbol, "buy", qty,
                    f"2차 분할매수 +{pnl:.2f}% · 모멘텀 유지 확인 · {bp_reason}",
                    execute_orders, source, config.duplicate_guard_seconds,
                    reason_code="BUY_STAGE2",
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

    for _, row in leader_df.head(max(1, int(config.kr_new_entry_top_n))).iterrows():
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

        if not _kr_signal_fresh(row, now, config):
            _diag(result, symbol, "SKIP_STALE_SIGNAL", "국내 TOP5 신호가 오래되어 신규매수 차단")
            continue
        if _kr_row_bool(row, "모멘텀약화", False):
            _diag(result, symbol, "SKIP_WEAK_MOMENTUM", "최근 단기 모멘텀 약화 종목이라 신규매수 차단")
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
            and "매수 후보" in signal
            and is_rank1
            and lead_score >= float(config.leader_exception_min_lead_score)
            and combined >= float(config.leader_exception_min_combined_score)
        )
        demo_relaxed = (
            is_demo
            and bool(config.demo_relaxed_entry_enabled)
            and "매수 후보" in signal
            and combined >= float(config.demo_min_combined_score)
        )

        if not (normal_entry or leader_exception or demo_relaxed):
            _diag(
                result, symbol, "SKIP_SCORE_LOW",
                f"점수 미달: 종합 {combined:.1f}, 주도 {lead_score:.1f}, 신호 {signal}",
                combined_score=combined,
                lead_score=lead_score,
            )
            continue

        price, price_err = _current_price(client, symbol)
        if price <= 0:
            _diag(result, symbol, "SKIP_PRICE_LOOKUP", price_err, combined_score=combined, lead_score=lead_score)
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

        r3 = _kr_row_float(row, "최근3분수익률", 0.0)
        r5 = _kr_row_float(row, "최근5분수익률", 0.0)
        vr = _kr_row_float(row, "거래량배수", 0.0)
        hd = _kr_row_float(row, "고점대비", 0.0)

        if normal_entry:
            action_name = "BUY1"
            entry_reason_code = "BUY_STAGE1"
            entry_reason = (
                f"빠른모멘텀 1차매수 · 종합 {combined:.1f} · "
                f"3분 {r3:+.2f}% · 5분 {r5:+.2f}% · 거래량 {vr:.2f}배 · "
                f"최근고점 {hd:.2f}% · {signal}"
            )
            leader_exception_flag = False
        elif leader_exception:
            action_name = "BUY1_EXCEPTION"
            entry_reason_code = "BUY_STAGE1_LEADER_EXCEPTION"
            entry_reason = (
                f"대장주 예외 1차매수 · TOP1 · 주도 {lead_score:.1f} · "
                f"종합 {combined:.1f} · {signal}"
            )
            leader_exception_flag = True
        else:
            action_name = "BUY1_DEMO_RELAXED"
            entry_reason_code = "BUY_STAGE1_DEMO_RELAXED"
            entry_reason = f"모의완화 1차매수 · TOP5 · 종합 {combined:.1f} · {signal}"
            leader_exception_flag = False

        act = _place_order(
            client, state, symbol, "buy", qty,
            entry_reason + " · " + bp_reason,
            execute_orders, source, config.duplicate_guard_seconds,
            reason_code=entry_reason_code,
        )

        result["actions"].append({
            "symbol": symbol,
            "name": str(row.get("종목명", symbol)),
            "action": action_name,
            "current_price": price,
            "estimated_cost": cost,
            "combined_score": combined,
            "lead_score": lead_score,
            "recent3": r3,
            "recent5": r5,
            "volume_ratio": vr,
            "high_distance": hd,
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

def load_us_state() -> Dict[str, Any]:
    fresh = {
        "date": _today_et(),
        "positions": {},
        "daily_buy_amount_usd": 0.0,
        "daily_orders": 0,
        "events": [],
        # C 전략 재진입 제어 상태 (ET 날짜가 바뀌면 fresh로 자동 초기화)
        "early_exit_banned_symbols": [],
        "early_exits_since_pause": 0,
        "new_entry_pause_until": "",
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
    state.setdefault("early_exit_banned_symbols", [])
    state.setdefault("early_exits_since_pause", 0)
    state.setdefault("new_entry_pause_until", "")
    return state


def save_us_state(state: Dict[str, Any]) -> None:
    US_STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, default=str),
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
    state["events"] = state["events"][-300:]
    append_trade_log({
        "time": row["time"],
        "mode": "AUTO",
        "market": "US",
        "symbol": symbol,
        "event": event,
        "detail": row["detail"],
    })


def _us_normalize_exchange(exchange: str) -> str:
    raw = str(exchange or "NASD").upper().strip()
    aliases = {
        "NASDAQ": "NASD", "NAS": "NASD", "NASD": "NASD",
        "NYSE": "NYSE", "NYS": "NYSE",
        "AMEX": "AMEX", "AMS": "AMEX",
    }
    return aliases.get(raw, "NASD")


def _us_quote_exchange(exchange: str) -> str:
    return {"NASD": "NAS", "NYSE": "NYS", "AMEX": "AMS"}.get(
        _us_normalize_exchange(exchange), "NAS"
    )


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
        msg1 = str(raw.get("msg1", "") or "")
        no_balance = "잔고" in msg1 and ("없" in msg1 or "조회" in msg1)
        if rt_cd and rt_cd != "0" and not no_balance:
            errors.append(f"{exchange}: {raw.get('msg_cd', '')} {msg1}".strip())
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
            "exchange": _us_normalize_exchange(r.get("거래소", "") or "NASD"),
        }
    return result, df, " / ".join(errors)


def _adopt_existing_us_holdings(client, state, holdings, config, result):
    if str(getattr(client, "env", "demo")).lower() != "demo":
        return
    for symbol, actual in holdings.items():
        if symbol in state["positions"]:
            state["positions"][symbol].setdefault(
                "exchange", _us_normalize_exchange(actual.get("exchange", "NASD"))
            )
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
            "exchange": _us_normalize_exchange(actual.get("exchange", "NASD")),
            "take1_sent": False,
            "peak_pnl": 0.0,
            "adopted_existing": True,
        }
        _diag(
            result,
            symbol,
            "ADOPT_EXISTING_US",
            f"한국투자 실제 보유 {qty}주를 자동 관리 대상으로 복구",
        )


def _us_stage_budget(config: AutoConfig, stage: int) -> float:
    weights = [int(config.buy1_pct), int(config.buy2_pct)]
    total = max(1, sum(weights))
    idx = max(0, min(1, int(stage) - 1))
    return float(config.us_per_stock_budget_usd) * weights[idx] / total


def _us_limit_price(config: AutoConfig, side: str, reference_price: float) -> float:
    price = float(reference_price or 0)
    if price <= 0:
        return 0.0
    if side == "buy":
        pct = max(0.0, min(2.0, float(config.us_buy_limit_buffer_pct)))
        price *= 1.0 + pct / 100.0
    else:
        pct = max(0.0, min(2.0, float(config.us_sell_limit_buffer_pct)))
        price *= 1.0 - pct / 100.0
    return round(max(0.01, price), 2)


def _us_stage_qty(config: AutoConfig, stage: int, reference_price: float) -> int:
    limit_price = _us_limit_price(config, "buy", reference_price)
    return _calc_qty(_us_stage_budget(config, stage), limit_price)


def _us_first_entry_qty(config: AutoConfig, reference_price: float) -> int:
    limit_price = _us_limit_price(config, "buy", reference_price)
    qty = _calc_qty(_us_stage_budget(config, 1), limit_price)
    if qty > 0:
        return qty
    if (
        config.allow_single_share_over_stage_budget
        and limit_price > 0
        and limit_price <= float(config.us_per_stock_budget_usd)
    ):
        return 1
    return 0


def _place_overseas_order(
    client,
    state,
    symbol: str,
    side: str,
    qty: int,
    reference_price: float,
    reason: str,
    execute_orders: bool,
    config: AutoConfig,
    exchange: str = "NASD",
    reason_code: str = "ORDER",
):
    base = {
        "qty": int(max(0, qty)),
        "reason": reason,
        "reason_code": reason_code,
    }

    if qty <= 0:
        return {
            **base,
            "status": "SKIP",
            "qty": 0,
            "reason": "주문수량 0",
        }
    if reference_price <= 0:
        return {
            **base,
            "status": "SKIP",
            "reason": "주문가격 0",
        }

    exchange = _us_normalize_exchange(exchange)
    limit_price = _us_limit_price(config, side, reference_price)
    if limit_price <= 0:
        return {
            **base,
            "status": "SKIP",
            "reason": "보정 주문가격 0",
        }

    full_reason = (
        f"{reason} · KIS기준가 ${reference_price:.2f} → "
        f"지정가 ${limit_price:.2f}"
    )

    if not execute_orders:
        _us_event(
            state,
            f"DRY_{side.upper()}",
            symbol,
            f"{qty}주 @ ${limit_price:.2f} · {exchange} · {full_reason}",
        )
        return {
            **base,
            "status": "DRY",
            "price": limit_price,
            "reference_price": float(reference_price),
            "exchange": exchange,
            "reason": full_reason,
        }

    try:
        res = client.overseas_order_us(
            symbol=symbol,
            qty=int(qty),
            side=side,
            limit_price=limit_price,
            exchange=exchange,
        )
    except Exception as e:
        _us_event(state, "ORDER_ERROR", symbol, repr(e))
        return {
            **base,
            "status": "ERROR",
            "price": limit_price,
            "error": repr(e),
            "exchange": exchange,
            "reason": full_reason,
        }

    if _order_ok(res):
        state["daily_orders"] += 1
        _us_event(
            state,
            f"{side.upper()}_ORDER",
            symbol,
            f"{qty}주 @ ${limit_price:.2f} · {exchange} · {full_reason} · {res}",
        )
        return {
            **base,
            "status": "ORDERED",
            "price": limit_price,
            "reference_price": float(reference_price),
            "exchange": exchange,
            "reason": full_reason,
            "response": res,
        }

    _us_event(state, "ORDER_REJECT", symbol, str(res))
    return {
        **base,
        "status": "REJECT",
        "price": limit_price,
        "reference_price": float(reference_price),
        "exchange": exchange,
        "reason": full_reason,
        "msg_cd": res.get("msg_cd", "") if isinstance(res, dict) else "",
        "msg1": res.get("msg1", "") if isinstance(res, dict) else "",
        "response": res,
    }


def _us_row_float(row, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except Exception:
        return float(default)


def _us_row_bool(row, key: str, default: bool = False) -> bool:
    v = row.get(key, default)
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on", "상승")


def _us_signal_fresh(row, now: datetime, config: AutoConfig) -> bool:
    raw = str(row.get("스캔시각", "") or "").strip()
    if not raw:
        return True
    try:
        ts = datetime.fromisoformat(raw)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=ET)
        age = abs((now - ts.astimezone(ET)).total_seconds())
        return age <= max(30, int(config.us_signal_max_age_seconds))
    except Exception:
        return False


def _us_momentum_ok(row, now: datetime, config: AutoConfig) -> bool:
    if row is None:
        return False
    if not _us_signal_fresh(row, now, config):
        return False
    signal = str(row.get("판정", row.get("종합신호", "")))
    if "매수 후보" not in signal:
        return False
    if _us_row_bool(row, "모멘텀약화", False):
        return False
    if _us_row_float(row, "종합점수", 0.0) < float(config.min_combined_score):
        return False
    return True


def _us_rank_value(row) -> int:
    if row is None:
        return 999
    raw = str(row.get("순위", "") or "").strip()
    digits = "".join(ch for ch in raw if ch.isdigit())
    try:
        return int(digits) if digits else 999
    except Exception:
        return 999


def _us_buy2_strict_ok(
    row,
    now: datetime,
    config: AutoConfig,
    hold_minutes: float,
) -> tuple[bool, str]:
    if row is None:
        return False, "전체순위 데이터에 보유종목 없음"
    if hold_minutes < float(config.us_buy2_min_hold_minutes):
        return False, f"보유 {hold_minutes:.1f}분 < {float(config.us_buy2_min_hold_minutes):.1f}분"
    if not _us_signal_fresh(row, now, config):
        return False, "최신 스캔 신호가 오래됨"

    signal = str(row.get("판정", row.get("종합신호", "")))
    rank = _us_rank_value(row)
    score = _us_row_float(row, "종합점수", 0.0)
    r5 = _us_row_float(row, "최근5분수익률", 0.0)
    r10 = _us_row_float(row, "최근10분수익률", 0.0)
    rel = _us_row_float(row, "상대강도", 0.0)

    reasons = []
    if "매수 후보" not in signal:
        reasons.append("매수후보아님")
    if _us_row_bool(row, "모멘텀약화", False):
        reasons.append("모멘텀약화")
    if rank > int(config.us_buy2_max_rank):
        reasons.append(f"TOP{rank}>TOP{int(config.us_buy2_max_rank)}")
    if score < float(config.us_buy2_min_score):
        reasons.append(f"점수{score:.1f}<{float(config.us_buy2_min_score):.1f}")
    if bool(config.us_buy2_require_recent5_positive) and r5 <= 0.0:
        reasons.append(f"5분{r5:+.2f}%")
    if bool(config.us_buy2_require_recent10_positive) and r10 <= 0.0:
        reasons.append(f"10분{r10:+.2f}%")
    if bool(config.us_buy2_require_relative_strength_positive) and rel <= 0.0:
        reasons.append(f"상대강도{rel:+.2f}%p")

    if reasons:
        return False, " / ".join(reasons)

    return True, (
        f"B_STRICT 통과 · 보유 {hold_minutes:.1f}분 · TOP{rank} · "
        f"점수 {score:.1f} · 5분 {r5:+.2f}% · 10분 {r10:+.2f}% · "
        f"상대강도 {rel:+.2f}%p"
    )


def _us_early_exit_decision(
    row,
    pnl: float,
    hold_minutes: float,
    config: AutoConfig,
) -> tuple[str, str, int]:
    """리플레이 C 전략과 같은 60분 약화/정체 청산 판정."""
    if not bool(config.us_early_exit_enabled):
        return "", "", 0
    if row is None:
        # 전체 순위 스캔 자체에 없으면 데이터 이상 가능성이 있으므로 안전하게 매도하지 않는다.
        return "", "", 0
    if hold_minutes < float(config.us_early_exit_min_hold_minutes):
        return "", "", 0

    signal = str(row.get("판정", row.get("종합신호", "")))
    rank = _us_rank_value(row)
    score = _us_row_float(row, "종합점수", 0.0)
    vwap_gap = _us_row_float(row, "VWAP괴리율", 0.0)
    r5 = _us_row_float(row, "최근5분수익률", 0.0)
    r10 = _us_row_float(row, "최근10분수익률", 0.0)
    rel = _us_row_float(row, "상대강도", 0.0)

    points = 0
    reasons = []
    if pnl < 0.0:
        points += 1
        reasons.append("손실중")
    if "매수 후보" not in signal:
        points += 1
        reasons.append("매수후보탈락")
    if rank > 5:
        points += 1
        reasons.append("TOP5탈락")
    if score < 62.0:
        points += 1
        reasons.append("점수62미만")
    if vwap_gap < 0.0:
        points += 1
        reasons.append("VWAP아래")
    if r5 < 0.0:
        points += 1
        reasons.append("5분음수")
    if r10 < 0.0:
        points += 1
        reasons.append("10분음수")
    if rel < 0.0:
        points += 1
        reasons.append("상대강도음수")

    if (
        pnl <= float(config.us_early_exit_loss_threshold_pct)
        and points >= int(config.us_early_exit_weak_points)
    ):
        return (
            "EARLY_EXIT_WEAK60",
            (
                f"60분 강한약화 · 보유 {hold_minutes:.1f}분 · 손익 {pnl:.2f}% · "
                f"약화 {points}점 · " + " / ".join(reasons)
            ),
            points,
        )

    if bool(config.us_stagnant_exit_enabled):
        stagnant = bool(
            pnl <= float(config.us_stagnant_exit_pnl_max_pct)
            and "매수 후보" not in signal
            and score < float(config.us_stagnant_exit_score_max)
            and abs(r5) <= float(config.us_stagnant_exit_momentum_abs_max)
            and abs(r10) <= float(config.us_stagnant_exit_momentum_abs_max)
        )
        if stagnant:
            return (
                "EARLY_EXIT_STAGNANT60",
                (
                    f"60분 정체청산 · 보유 {hold_minutes:.1f}분 · 손익 {pnl:.2f}% · "
                    f"점수 {score:.1f} · TOP{rank} · 5분 {r5:+.2f}% · 10분 {r10:+.2f}%"
                ),
                points,
            )

    return "", "", points


def _us_register_early_exit_control(
    state: Dict[str, Any],
    symbol: str,
    now: datetime,
    config: AutoConfig,
    result: Dict[str, Any],
    action: str,
) -> None:
    banned = {
        str(x).strip().upper()
        for x in (state.get("early_exit_banned_symbols", []) or [])
        if str(x).strip()
    }
    if bool(config.us_ban_same_symbol_after_early_exit):
        banned.add(str(symbol).upper())
        state["early_exit_banned_symbols"] = sorted(banned)

    count = int(state.get("early_exits_since_pause", 0) or 0) + 1
    threshold = max(1, int(config.us_pause_after_early_exits_count))

    if count >= threshold:
        pause_until = now + timedelta(minutes=float(config.us_pause_new_entries_minutes))
        state["new_entry_pause_until"] = pause_until.isoformat()
        state["early_exits_since_pause"] = 0
        _diag(
            result,
            symbol,
            "US_NEW_ENTRY_PAUSE_STARTED",
            (
                f"조기청산 {threshold}회 누적 → 신규매수 "
                f"{float(config.us_pause_new_entries_minutes):.0f}분 중지 · "
                f"해제 {pause_until.strftime('%H:%M:%S')} ET"
            ),
            early_exit_action=action,
        )
    else:
        state["early_exits_since_pause"] = count


def _us_new_entry_pause_active(state: Dict[str, Any], now: datetime) -> tuple[bool, str]:
    raw = str(state.get("new_entry_pause_until", "") or "").strip()
    if not raw:
        return False, ""
    try:
        until = datetime.fromisoformat(raw)
        if until.tzinfo is None:
            until = until.replace(tzinfo=ET)
        until = until.astimezone(ET)
    except Exception:
        state["new_entry_pause_until"] = ""
        return False, ""

    if now < until:
        return True, until.strftime("%H:%M:%S")

    state["new_entry_pause_until"] = ""
    return False, ""


def _effective_us_sellable(actual_qty: int, raw_sellable: int) -> int:
    actual_qty = max(0, int(actual_qty))
    raw_sellable = max(0, int(raw_sellable))
    return min(actual_qty, raw_sellable) if raw_sellable > 0 else actual_qty


def _set_us_pending(
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
        "reason_code": act.get("reason_code", action),
        "side": side,
        "qty": int(order_qty),
        "before_qty": int(before_qty),
        "expected_after_qty": int(expected_after),
        "stage": int(stage),
        "sent_at": now.isoformat(),
        "price": float(act.get("price", 0) or 0),
        "exchange": act.get("exchange", pos.get("exchange", "NASD")),
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
        "execute_orders": bool(execute_orders),
        "source": source,
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

    leader_rows = {}
    if leader_df is not None and not leader_df.empty:
        for _, row in leader_df.iterrows():
            sym = str(row.get("종목코드", row.get("종목", ""))).strip().upper()
            if sym:
                leader_rows[sym] = row

    for symbol, pos in list(state["positions"].items()):
        actual = holdings.get(symbol, {})
        actual_qty = int(actual.get("qty", 0) or 0)
        raw_sellable = int(actual.get("sellable_qty", 0) or 0)
        sellable_qty = _effective_us_sellable(actual_qty, raw_sellable)

        if actual_qty > 0:
            pos["actual_qty"] = actual_qty
            pos["sellable_qty"] = raw_sellable
            pos["exchange"] = _us_normalize_exchange(
                actual.get("exchange", pos.get("exchange", "NASD"))
            )
            if float(actual.get("avg_price", 0) or 0) > 0:
                pos["avg_price"] = float(actual["avg_price"])

        pending = pos.get("pending_order")
        if isinstance(pending, dict) and pending:
            side = str(pending.get("side", "")).lower()
            before_qty = int(pending.get("before_qty", 0) or 0)
            expected_after = int(pending.get("expected_after_qty", before_qty) or before_qty)
            action = str(pending.get("action", ""))
            stage = int(pending.get("stage", 0) or 0)

            if side == "buy":
                confirmed = actual_qty >= expected_after and expected_after > before_qty
            elif side == "sell":
                confirmed = actual_qty <= expected_after
            else:
                confirmed = False

            if confirmed:
                if stage in (1, 2):
                    pos["buy_stage"] = max(int(pos.get("buy_stage", 0) or 0), stage)
                if action in ("TAKE1", "PROFIT_GUARD1"):
                    pos["take1_sent"] = True
                pos.pop("pending_order", None)
                pos["last_confirmed_at"] = now.isoformat()
                _diag(
                    result,
                    symbol,
                    "ORDER_CONFIRMED",
                    f"KIS 잔고 반영 확인: {before_qty}주 → {actual_qty}주 · {action}",
                    before_qty=before_qty,
                    after_qty=actual_qty,
                )

                if side == "sell" and actual_qty <= 0:
                    state["positions"].pop(symbol, None)
                    _diag(result, symbol, "EXIT_CONFIRMED", "KIS 실제 보유 0주 → 청산 완료", after_qty=0)
                    continue
            else:
                sent_at = str(pending.get("sent_at", "") or "")
                try:
                    age = (now - datetime.fromisoformat(sent_at)).total_seconds()
                except Exception:
                    age = 0.0
                label = "PENDING_TIMEOUT_LOCKED" if age >= int(config.pending_timeout_seconds) else "WAIT_PENDING_ORDER"
                _diag(
                    result,
                    symbol,
                    label,
                    (
                        f"체결/잔고반영 대기: {action} {pending.get('qty', 0)}주 · "
                        f"KIS 확인 {actual_qty}주 · {age:.0f}초 · 자동 재주문 차단"
                    ),
                    before_qty=before_qty,
                    after_qty=actual_qty,
                )
                continue

        if actual_qty <= 0:
            created = str(pos.get("created_at", "") or "")
            try:
                age = now - datetime.fromisoformat(created)
            except Exception:
                age = timedelta(minutes=10)
            if age > timedelta(minutes=5):
                state["positions"].pop(symbol, None)
                _diag(result, symbol, "DROP_TRACKING", "KIS 실제 보유수량 0주 → 추적 종료", after_qty=0)
            continue

        exchange = _us_normalize_exchange(actual.get("exchange") or pos.get("exchange") or "NASD")
        price = _overseas_current_price(client, symbol, exchange=exchange)
        if price <= 0:
            fallback = float(actual.get("current_price", 0) or 0)
            if fallback > 0:
                price = fallback
                _diag(result, symbol, "PRICE_FALLBACK_BALANCE", f"KIS 현재가 조회 실패 → 잔고 현재가 ${price:.2f} 사용")
            else:
                _diag(result, symbol, "SKIP_US_PRICE", "KIS 미국 현재가 조회 실패")
                continue

        avg = float(pos.get("avg_price") or actual.get("avg_price") or price)
        pnl = (price / avg - 1.0) * 100.0 if avg > 0 else 0.0
        prev_peak = float(pos.get("peak_pnl", pnl) or pnl)
        peak_pnl = max(prev_peak, pnl)
        pos["peak_pnl"] = round(peak_pnl, 4)
        drawdown_from_peak = max(0.0, peak_pnl - pnl)

        def place_sell(action: str, qty: int, reason: str, reason_code: str):
            qty = min(max(0, int(qty)), sellable_qty)
            if qty <= 0:
                _diag(
                    result,
                    symbol,
                    "WAIT_SELLABLE_QTY",
                    f"{action} 조건이나 주문가능수량 0주 · 실제보유 {actual_qty}주",
                    before_qty=actual_qty,
                    after_qty=actual_qty,
                )
                return None
            act = _place_overseas_order(
                client, state, symbol, "sell", qty, price, reason,
                execute_orders, config, exchange=exchange,
                reason_code=reason_code,
            )
            _set_us_pending(pos, now, action, "sell", qty, actual_qty, act)
            result["actions"].append({
                "symbol": symbol,
                "name": actual.get("name") or pos.get("name", symbol),
                "action": action,
                "pnl": round(pnl, 3),
                "peak_pnl": round(peak_pnl, 3),
                "drawdown_from_peak": round(drawdown_from_peak, 3),
                "qty": qty,
                "before_qty": actual_qty,
                "after_qty": max(0, actual_qty - qty),
                **act,
            })
            return act

        if now.time() >= _clock(config.us_force_exit_time):
            place_sell(
                "FORCE_SELL",
                sellable_qty,
                f"당일 강제청산 {config.us_force_exit_time} ET · KIS 실제 잔고 기준",
                "SELL_FORCE_EXIT",
            )
            continue

        if pnl <= -abs(float(config.stop_loss_pct)):
            place_sell(
                "STOP_LOSS",
                sellable_qty,
                f"손절 {pnl:.2f}%",
                "SELL_STOP_LOSS",
            )
            continue

        # C 전략: 60분 이후 강한약화/정체 포지션을 -3%까지 끌지 않고 조기청산.
        try:
            opened_at = datetime.fromisoformat(str(pos.get("created_at", "") or ""))
            if opened_at.tzinfo is None:
                opened_at = opened_at.replace(tzinfo=ET)
            hold_minutes = max(0.0, (now - opened_at.astimezone(ET)).total_seconds() / 60.0)
        except Exception:
            hold_minutes = 0.0

        early_row = leader_rows.get(symbol)
        early_action, early_reason, weak_points = _us_early_exit_decision(
            early_row, pnl, hold_minutes, config
        )
        if early_action:
            reason_code = (
                "SELL_EARLY_WEAK60"
                if early_action == "EARLY_EXIT_WEAK60"
                else "SELL_EARLY_STAGNANT60"
            )
            act = place_sell(
                early_action,
                sellable_qty,
                early_reason,
                reason_code,
            )
            if act and act.get("status") == "ORDERED":
                _us_register_early_exit_control(
                    state, symbol, now, config, result, early_action
                )
            continue

        if pnl >= float(config.take1_pct) and not pos.get("take1_sent"):
            sell_qty = max(1, actual_qty // 2)
            act = place_sell(
                "TAKE1",
                sell_qty,
                f"1차 익절 {pnl:.2f}% · 약 50%",
                "SELL_TAKE_PROFIT_1",
            )
            if act and act.get("status") == "DRY":
                pos["take1_sent"] = True
            continue

        if pnl >= float(config.take2_pct) and pos.get("take1_sent"):
            place_sell(
                "TAKE2",
                sellable_qty,
                f"2차 익절 {pnl:.2f}% · 잔여 전량",
                "SELL_TAKE_PROFIT_2",
            )
            continue

        if (
            peak_pnl >= float(config.us_profit_guard_trigger_pct)
            and drawdown_from_peak >= float(config.us_profit_guard_drawdown_pct)
        ):
            if not pos.get("take1_sent"):
                sell_qty = max(1, actual_qty // 2)
                act = place_sell(
                    "PROFIT_GUARD1",
                    sell_qty,
                    (
                        f"수익보호 1차 · 최고 +{peak_pnl:.2f}% → 현재 {pnl:.2f}% "
                        f"({drawdown_from_peak:.2f}%p 되밀림)"
                    ),
                    "SELL_PROFIT_GUARD_1",
                )
                if act and act.get("status") == "DRY":
                    pos["take1_sent"] = True
                continue
            else:
                place_sell(
                    "PROFIT_GUARD2",
                    sellable_qty,
                    (
                        f"수익보호 2차 · 최고 +{peak_pnl:.2f}% → 현재 {pnl:.2f}% "
                        f"({drawdown_from_peak:.2f}%p 되밀림)"
                    ),
                    "SELL_PROFIT_GUARD_2",
                )
                continue

        if now.time() >= _clock(config.us_last_entry_time):
            continue

        stage = int(pos.get("buy_stage", 1) or 1)
        if state["daily_orders"] >= int(config.max_daily_orders):
            continue

        if stage == 1 and pnl >= float(config.us_buy2_strict_trigger_pct):
            leader_row = leader_rows.get(symbol)
            try:
                opened_at = datetime.fromisoformat(str(pos.get("created_at", "") or ""))
                if opened_at.tzinfo is None:
                    opened_at = opened_at.replace(tzinfo=ET)
                buy2_hold_minutes = max(
                    0.0,
                    (now - opened_at.astimezone(ET)).total_seconds() / 60.0,
                )
            except Exception:
                buy2_hold_minutes = 0.0

            buy2_ok, buy2_reason = _us_buy2_strict_ok(
                leader_row, now, config, buy2_hold_minutes
            )
            if not buy2_ok:
                _diag(
                    result,
                    symbol,
                    "SKIP_BUY2_B_STRICT",
                    f"B_STRICT 2차매수 보류 · 손익 +{pnl:.2f}% · {buy2_reason}",
                )
                continue

            qty = _us_stage_qty(config, 2, price)
            buy_limit = _us_limit_price(config, "buy", price)
            cost = qty * buy_limit
            if qty > 0 and state["daily_buy_amount_usd"] + cost <= float(config.us_daily_budget_usd):
                act = _place_overseas_order(
                    client,
                    state,
                    symbol,
                    "buy",
                    qty,
                    price,
                    f"B_STRICT 2차 분할매수 +{pnl:.2f}% · {buy2_reason}",
                    execute_orders,
                    config,
                    exchange=exchange,
                    reason_code="BUY_STAGE2",
                )
                result["actions"].append({
                    "symbol": symbol,
                    "name": actual.get("name") or pos.get("name", symbol),
                    "action": "BUY2",
                    "pnl": round(pnl, 3),
                    "qty": qty,
                    "before_qty": actual_qty,
                    "after_qty": actual_qty + qty,
                    **act,
                })
                if act.get("status") == "ORDERED":
                    pos["expected_qty"] = actual_qty + qty
                    state["daily_buy_amount_usd"] += cost
                    _set_us_pending(pos, now, "BUY2", "buy", qty, actual_qty, act, stage=2)
                elif act.get("status") == "DRY":
                    pos["buy_stage"] = 2
                continue

    if now.time() >= _clock(config.us_last_entry_time):
        result["message"] = f"{config.us_last_entry_time} ET 이후 미국 신규/추가매수 금지"
        save_us_state(state)
        result["state"] = state
        return result

    pause_active, pause_until_text = _us_new_entry_pause_active(state, now)
    if pause_active:
        result["message"] = (
            f"C 전략 신규매수 휴식 중 · {pause_until_text} ET까지 신규진입 차단"
        )
        save_us_state(state)
        result["state"] = state
        return result

    if len(state["positions"]) >= int(config.max_positions):
        result["message"] = "미국 최대 보유/주문대기 종목 수 도달"
        save_us_state(state)
        result["state"] = state
        return result

    if leader_df is None or leader_df.empty:
        result["message"] = "미국 최신 모멘텀 TOP5 데이터 없음 · 신규매수 금지, 보유종목만 관리"
        save_us_state(state)
        result["state"] = state
        return result

    for _, row in leader_df.head(max(1, int(config.us_new_entry_top_n))).iterrows():
        if len(state["positions"]) >= int(config.max_positions):
            break
        if state["daily_orders"] >= int(config.max_daily_orders):
            break

        symbol = str(row.get("종목코드", row.get("종목", ""))).strip().upper()
        if not symbol:
            _diag(result, symbol, "SKIP_BAD_SYMBOL", "미국 종목코드 없음")
            continue
        if symbol in state["positions"]:
            _diag(result, symbol, "SKIP_ALREADY_TRACKED", "이미 미국 자동매매 추적중")
            continue
        if symbol in holdings:
            _diag(result, symbol, "SKIP_ALREADY_HELD", f"실제 미국계좌 보유중: {holdings[symbol].get('qty', 0)}주")
            continue

        banned_symbols = {
            str(x).strip().upper()
            for x in (state.get("early_exit_banned_symbols", []) or [])
            if str(x).strip()
        }
        if bool(config.us_ban_same_symbol_after_early_exit) and symbol in banned_symbols:
            _diag(
                result,
                symbol,
                "SKIP_EARLY_EXIT_REENTRY_BAN",
                "C 전략: 조기청산된 종목은 당일 재매수 금지",
            )
            continue

        if not _us_signal_fresh(row, now, config):
            _diag(result, symbol, "SKIP_STALE_SIGNAL", "미국 단기모멘텀 스캔이 오래되어 신규매수 차단")
            continue

        signal = str(row.get("판정", row.get("종합신호", "")))
        combined = _us_row_float(row, "종합점수", 0.0)

        if config.require_green_signal and "매수 후보" not in signal:
            _diag(
                result,
                symbol,
                "SKIP_SIGNAL_NOT_GREEN",
                f"매수 후보 신호 아님: {signal}",
                combined_score=combined,
            )
            continue

        if combined < float(config.min_combined_score):
            _diag(
                result,
                symbol,
                "SKIP_SCORE_LOW",
                f"미국 점수 미달: {combined:.1f} < {float(config.min_combined_score):.1f}",
                combined_score=combined,
            )
            continue

        if _us_row_bool(row, "모멘텀약화", False):
            _diag(
                result,
                symbol,
                "SKIP_WEAK_MOMENTUM",
                "미국 최근 단기 모멘텀 약화 종목이라 신규매수 차단",
                combined_score=combined,
            )
            continue

        exchange = _us_normalize_exchange(row.get("거래소", "NASD"))
        price = _overseas_current_price(client, symbol, exchange=exchange)
        if price <= 0:
            _diag(result, symbol, "SKIP_US_PRICE", "KIS 미국 현재가 조회 실패 · 신규매수 차단")
            continue

        qty = _us_first_entry_qty(config, price)
        if qty <= 0:
            _diag(
                result,
                symbol,
                "SKIP_US_BUDGET",
                f"1차예산 ${_us_stage_budget(config, 1):.2f}으로 주문가능수량 0",
            )
            continue

        buy_limit = _us_limit_price(config, "buy", price)
        cost = qty * buy_limit
        if cost > float(config.us_per_stock_budget_usd):
            _diag(
                result,
                symbol,
                "SKIP_PER_STOCK_BUDGET",
                f"예상 주문금액 ${cost:.2f} > 종목한도 ${float(config.us_per_stock_budget_usd):.2f}",
            )
            continue
        if state["daily_buy_amount_usd"] + cost > float(config.us_daily_budget_usd):
            _diag(
                result,
                symbol,
                "SKIP_DAILY_BUDGET",
                (
                    f"누적 ${state['daily_buy_amount_usd']:.2f} + 예상 ${cost:.2f} > "
                    f"일일한도 ${float(config.us_daily_budget_usd):.2f}"
                ),
            )
            continue

        r5 = _us_row_float(row, "최근5분수익률", 0.0)
        r10 = _us_row_float(row, "최근10분수익률", 0.0)
        vr = _us_row_float(row, "거래량배수", 0.0)
        hd = _us_row_float(row, "고점대비", 0.0)
        day_change = _us_row_float(row, "등락률", _us_row_float(row, "당일등락률", 0.0))

        reason = (
            f"초기모멘텀 1차 50% · 점수 {combined:.1f} · "
            f"5분 {r5:+.2f}% / 10분 {r10:+.2f}% · "
            f"거래량 {vr:.2f}배 · 고점대비 {hd:.2f}% · "
            f"당일 {day_change:+.2f}%"
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
            config,
            exchange=exchange,
            reason_code="BUY_STAGE1",
        )
        result["actions"].append({
            "symbol": symbol,
            "name": str(row.get("종목명", symbol)),
            "action": "BUY1",
            "reason": act.get("reason", reason),
            "reason_code": act.get("reason_code", "BUY_STAGE1"),
            "combined_score": combined,
            "pnl": "",
            "qty": qty,
            "before_qty": 0,
            "after_qty": qty,
            "recent5": r5,
            "recent10": r10,
            "volume_ratio": vr,
            "high_distance": hd,
            "day_change_pct": day_change,
            **act,
        })

        if act.get("status") == "ORDERED":
            pos = {
                "name": str(row.get("종목명", symbol)),
                "created_at": now.isoformat(),
                "buy_stage": 0,
                "avg_price": price,
                "expected_qty": qty,
                "exchange": exchange,
                "take1_sent": False,
                "peak_pnl": 0.0,
            }
            _set_us_pending(pos, now, "BUY1", "buy", qty, 0, act, stage=1)
            state["positions"][symbol] = pos
            state["daily_buy_amount_usd"] += cost

    if not result["actions"]:
        if any(
            isinstance(p, dict) and p.get("pending_order")
            for p in state["positions"].values()
        ):
            result["message"] = "미국 주문 체결/잔고반영 대기 중 · 중복주문 차단"
        else:
            result["message"] = "최신 단기모멘텀 기준을 통과한 미국 신규매수 후보 없음"

    save_us_state(state)
    result["state"] = state
    return result


run_kr_cycle = run_domestic_cycle
run_us_cycle = run_overseas_cycle

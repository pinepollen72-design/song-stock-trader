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

KST = ZoneInfo("Asia/Seoul")
ET = ZoneInfo("America/New_York")


@dataclass
class AutoConfig:
    # 국내
    daily_budget: int = 300_000
    per_stock_budget: int = 100_000

    # 공통
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

    # 국내 대장주 예외 진입
    # TOP1 + 주도주점수 75 이상 + 종합점수 60 이상이면
    # 녹색 신호가 아니어도 1차 진입만 허용합니다.
    leader_exception_enabled: bool = True
    leader_exception_min_lead_score: float = 75.0
    leader_exception_min_combined_score: float = 60.0

    # 국내 장중 규칙
    last_entry_time: str = "14:50"
    force_exit_time: str = "15:15"

    # 미국 장중 규칙
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


# -------------------------------------------------------------------
# 국내 상태
# -------------------------------------------------------------------

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
    if STATE_FILE.exists():
        STATE_FILE.unlink()


def _event(state, event, symbol="", detail=""):
    row = {
        "time": _now_kst().isoformat(timespec="seconds"),
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


def _current_price(client, code: str) -> float:
    raw = client.domestic_price(code)
    out = (raw or {}).get("output", {}) or {}
    try:
        return float(out.get("stck_prpr", 0))
    except Exception:
        return 0.0


def _calc_qty(amount: float, price: float) -> int:
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
    out["보유수량"] = (
        pd.to_numeric(df[qty], errors="coerce").fillna(0).astype(int)
        if qty else 0
    )
    out["평균매입가"] = (
        pd.to_numeric(df[avg], errors="coerce").fillna(0.0)
        if avg else 0.0
    )
    out["현재가"] = (
        pd.to_numeric(df[cur], errors="coerce").fillna(0.0)
        if cur else 0.0
    )

    return out[out["보유수량"] > 0].reset_index(drop=True)


def _domestic_balance(client):
    if hasattr(client, "domestic_balance"):
        return client.domestic_balance()

    tr_id = "VTTC8434R" if getattr(client, "env", "demo") == "demo" else "TTTC8434R"

    return client.get(
        "/uapi/domestic-stock/v1/trading/inquire-balance",
        tr_id,
        {
            "CANO": client.account_no,
            "ACNT_PRDT_CD": client.product_code,
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "00",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        },
    )


def _actual_holdings_map(client):
    raw = _domestic_balance(client)
    df = _parse_domestic_holdings(raw)

    if df.empty:
        return {}, df

    result = {}
    for _, r in df.iterrows():
        result[str(r["종목코드"]).zfill(6)] = {
            "name": r.get("종목명", ""),
            "qty": int(r.get("보유수량", 0)),
            "avg_price": float(r.get("평균매입가", 0)),
            "current_price": float(r.get("현재가", 0)),
        }

    return result, df


def _place_order(client, state, symbol, side, qty, reason, execute_orders):
    if qty <= 0:
        return {"status": "SKIP", "reason": "주문수량 0"}

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
        _event(
            state,
            f"{side.upper()}_ORDER",
            symbol,
            f"{qty}주 · {reason} · {res}",
        )
        return {"status": "ORDERED", "qty": qty, "response": res}

    _event(state, "ORDER_REJECT", symbol, str(res))
    return {
        "status": "REJECT",
        "qty": qty,
        "msg_cd": res.get("msg_cd", "") if isinstance(res, dict) else "",
        "msg1": res.get("msg1", "") if isinstance(res, dict) else "",
        "response": res,
    }


def run_domestic_cycle(
    client,
    leader_df: pd.DataFrame,
    config: AutoConfig,
    execute_orders: bool = False,
) -> Dict[str, Any]:
    state = load_state()
    now = _now_kst()

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

    # 국내 분석/판단 운영시간: 08:30 ~ 16:00
    # 실제 모의/실전 주문 전송은 정규장(09:00 ~ 15:30)에만 허용합니다.
    if not (dtime(8, 30) <= now.time() < dtime(16, 0)):
        result["message"] = "국내 운영시간 외(08:30~16:00): 판단/주문 없음"
        save_state(state)
        return result

    if execute_orders and not (dtime(9, 0) <= now.time() < dtime(15, 30)):
        execute_orders = False
        result["execute_orders"] = False
        result["order_gate_message"] = (
            "08:30~09:00 / 15:30~16:00에는 분석·DRY 판단만 수행합니다. "
            "한국 정규장 주문 전송은 09:00~15:30에만 허용합니다."
        )

    holdings, holdings_df = _actual_holdings_map(client)
    result["holdings"] = holdings_df.to_dict("records")

    for symbol, pos in list(state["positions"].items()):
        actual = holdings.get(symbol, {})
        actual_qty = int(actual.get("qty", 0))

        if actual_qty > 0:
            pos["actual_qty"] = actual_qty
            if actual.get("avg_price", 0) > 0:
                pos["avg_price"] = float(actual["avg_price"])

        if actual_qty <= 0:
            created = pos.get("created_at", "")
            try:
                age = now - datetime.fromisoformat(created)
            except Exception:
                age = timedelta(minutes=10)

            if age > timedelta(minutes=5):
                state["positions"].pop(symbol, None)
            continue

        price = _current_price(client, symbol)
        if price <= 0:
            continue

        avg = float(pos.get("avg_price") or actual.get("avg_price") or price)
        pnl = (price / avg - 1) * 100 if avg > 0 else 0.0

        if now.time() >= _clock(config.force_exit_time):
            act = _place_order(
                client, state, symbol, "sell", actual_qty,
                f"당일 강제청산 {config.force_exit_time}", execute_orders
            )
            result["actions"].append(
                {"symbol": symbol, "action": "FORCE_SELL", "pnl": pnl, **act}
            )
            continue

        if pnl <= -abs(config.stop_loss_pct):
            act = _place_order(
                client, state, symbol, "sell", actual_qty,
                f"손절 {pnl:.2f}%", execute_orders
            )
            result["actions"].append(
                {"symbol": symbol, "action": "STOP_LOSS", "pnl": pnl, **act}
            )
            continue

        if pnl >= config.take2_pct:
            act = _place_order(
                client, state, symbol, "sell", actual_qty,
                f"2차 익절 {pnl:.2f}%", execute_orders
            )
            result["actions"].append(
                {"symbol": symbol, "action": "TAKE2", "pnl": pnl, **act}
            )
            continue

        if pnl >= config.take1_pct and not pos.get("take1_sent"):
            sell_qty = max(1, actual_qty // 2)
            act = _place_order(
                client, state, symbol, "sell", sell_qty,
                f"1차 익절 {pnl:.2f}%", execute_orders
            )
            result["actions"].append(
                {"symbol": symbol, "action": "TAKE1", "pnl": pnl, **act}
            )
            if act["status"] in ("ORDERED", "DRY"):
                pos["take1_sent"] = True
            continue

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
                result["actions"].append(
                    {"symbol": symbol, "action": "BUY2", "pnl": pnl, **act}
                )
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
                result["actions"].append(
                    {"symbol": symbol, "action": "BUY3", "pnl": pnl, **act}
                )
                if act["status"] in ("ORDERED", "DRY"):
                    pos["buy_stage"] = 3
                    state["daily_buy_amount"] += int(qty * price)

    if now.time() >= _clock(config.last_entry_time):
        result["message"] = f"{config.last_entry_time} 이후 신규매수 금지"
        save_state(state)
        result["state"] = state
        return result

    if len(state["positions"]) >= config.max_positions:
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
            continue

        price = _current_price(client, symbol)
        if price <= 0:
            continue

        qty = _calc_qty(parts[0], price)
        if qty <= 0:
            result["actions"].append({
                "symbol": symbol,
                "action": "SKIP",
                "reason": f"1차 예산 {parts[0]:,}원보다 주가가 높음",
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
            execute_orders,
        )
        result["actions"].append({
            "symbol": symbol,
            "action": "BUY1_EXCEPTION" if leader_exception and not normal_entry else "BUY1",
            "reason": entry_reason,
            **act,
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
                "force_exit_sent": False,
            }
            state["daily_buy_amount"] += cost

    if not result["actions"]:
        result["message"] = "조건을 통과한 신규 매수 후보 없음"

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
) -> Dict[str, Any]:
    """
    미국 자동매매 1회 사이클.

    모의투자와 실전투자가 동일한 전략 규칙을 사용합니다.
    - TOP5 중 🟢 매수 후보 + 종합점수 통과 종목만 신규진입
    - 달러 예산 기준 1/2/3차 분할매수
    - 손절 / 1차 익절 / 2차 익절
    - 장마감 전 강제청산
    - 실제 KIS 현재가와 계좌 잔고를 기준으로 판단
    """
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

    # 기존 추적 포지션 관리
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
            qty = _us_stage_qty(config, 2, price)
            cost = qty * price
            if (
                qty > 0
                and state["daily_buy_amount_usd"] + cost <= config.us_daily_budget_usd
            ):
                act = _place_overseas_order(
                    client, state, symbol, "buy", qty, price,
                    f"2차 분할매수 +{pnl:.2f}%",
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
            qty = _us_stage_qty(config, 3, price)
            cost = qty * price
            if (
                qty > 0
                and state["daily_buy_amount_usd"] + cost <= config.us_daily_budget_usd
            ):
                act = _place_overseas_order(
                    client, state, symbol, "buy", qty, price,
                    f"3차 분할매수 +{pnl:.2f}%",
                    execute_orders,
                )
                result["actions"].append(
                    {"symbol": symbol, "action": "BUY3", "pnl": pnl, **act}
                )
                if act["status"] in ("ORDERED", "DRY"):
                    pos["buy_stage"] = 3
                    pos["expected_qty"] = int(pos.get("expected_qty", 0)) + qty
                    state["daily_buy_amount_usd"] += cost

    # 신규진입 제한
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

    # 신규진입
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

        # 실전과 동일: 녹색 매수 후보 + 최소점수 둘 다 통과해야 진입
        if config.require_green_signal and "매수 후보" not in signal:
            continue

        if combined < config.min_combined_score:
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

        qty = _us_stage_qty(config, 1, price)
        if qty <= 0:
            result["actions"].append({
                "symbol": symbol,
                "action": "SKIP",
                "reason": (
                    f"1차 예산 ${_us_stage_budget(config, 1):.2f}보다 "
                    f"주가 ${price:.2f}가 높음"
                ),
            })
            continue

        cost = qty * price

        if state["daily_buy_amount_usd"] + cost > config.us_daily_budget_usd:
            continue

        act = _place_overseas_order(
            client, state, symbol, "buy", qty, price,
            f"1차 분할매수 · 종합점수 {combined:.1f} · {signal}",
            execute_orders,
        )
        result["actions"].append({
            "symbol": symbol,
            "action": "BUY1",
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

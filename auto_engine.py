from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Any, Dict
from zoneinfo import ZoneInfo

import pandas as pd

from trader_core import parse_domestic_holdings, score_ticker

KST = ZoneInfo("Asia/Seoul")
ET = ZoneInfo("America/New_York")
STATE_DIR = Path(os.getenv("SONG_TRADER_STATE_DIR", "/tmp/song_trader_v2"))
STATE_DIR.mkdir(parents=True, exist_ok=True)
KR_STATE_FILE = STATE_DIR / "kr_state.json"
US_STATE_FILE = STATE_DIR / "us_state.json"


@dataclass
class AutoConfig:
    kr_daily_budget: int = 10_000_000
    kr_per_stock_budget: int = 3_000_000
    us_daily_budget_usd: float = 5_000.0
    us_per_stock_budget_usd: float = 1_500.0
    max_positions: int = 3
    min_score: float = 50.0
    buy1_pct: int = 50
    buy2_pct: int = 30
    buy3_pct: int = 20
    add2_trigger_pct: float = 0.5
    add3_trigger_pct: float = 1.0
    stop_loss_pct: float = 3.0
    take1_pct: float = 3.0
    take2_pct: float = 5.0
    kr_last_entry_time: str = "14:50"
    kr_force_exit_time: str = "15:15"
    us_last_entry_time: str = "15:30"
    us_force_exit_time: str = "15:50"
    buying_power_buffer_pct: float = 5.0
    confirm_wait_seconds: int = 8
    # 모의투자 전용 계좌라는 전제. 실전에서는 worker가 자동으로 False 처리한다.
    force_exit_all_demo_holdings: bool = True


def _clock(hhmm: str) -> dtime:
    h, m = [int(x) for x in hhmm.split(":")]
    return dtime(h, m)


def _today(tz) -> str:
    return datetime.now(tz).strftime("%Y-%m-%d")


def _fresh_state(tz) -> dict:
    return {
        "date": _today(tz),
        "positions": {},
        "daily_buy_amount": 0.0,
        "daily_orders": 0,
    }


def _load(path: Path, tz) -> dict:
    fresh = _fresh_state(tz)
    try:
        if not path.exists():
            return fresh
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("date") != fresh["date"]:
            return fresh
        data.setdefault("positions", {})
        data.setdefault("daily_buy_amount", 0.0)
        data.setdefault("daily_orders", 0)
        return data
    except Exception:
        return fresh


def _save(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _domestic_holdings(client) -> tuple[dict, list]:
    raw = client.domestic_balance()
    df = parse_domestic_holdings(raw)
    m = {}
    for _, r in df.iterrows():
        code = str(r.get("종목코드", "")).zfill(6)
        m[code] = {
            "symbol": code,
            "name": str(r.get("종목명", code)),
            "qty": int(r.get("보유수량", 0) or 0),
            "avg_price": float(r.get("평균매입가", 0) or 0),
            "current_price": float(r.get("현재가", 0) or 0),
        }
    return m, list(m.values())


def _us_num(r: dict, *names) -> float:
    for name in names:
        try:
            v = float(str(r.get(name, 0) or 0).replace(",", ""))
            if v != 0:
                return v
        except Exception:
            pass
    return 0.0


def _overseas_holdings(client) -> tuple[dict, list]:
    merged = {}
    for exchange in ("NASD", "NYSE", "AMEX"):
        try:
            raw = client.overseas_balance_us(exchange=exchange)
        except Exception:
            continue
        rows = (raw or {}).get("output1", []) or []
        for r in rows:
            symbol = str(r.get("ovrs_pdno") or r.get("pdno") or r.get("ovrs_item_cd") or "").strip().upper()
            if not symbol:
                continue
            qty = int(_us_num(r, "ovrs_cblc_qty", "hldg_qty", "hold_qty"))
            if qty <= 0:
                continue
            merged[symbol] = {
                "symbol": symbol,
                "name": str(r.get("ovrs_item_name") or r.get("prdt_name") or symbol),
                "qty": qty,
                "avg_price": _us_num(r, "pchs_avg_pric", "avg_pric"),
                "current_price": _us_num(r, "now_pric2", "ovrs_nmix_prpr", "prpr"),
                "exchange": exchange,
            }
    return merged, list(merged.values())


def _kr_price(client, symbol: str) -> float:
    try:
        out = (client.domestic_price(symbol) or {}).get("output", {}) or {}
        return float(out.get("stck_prpr", 0) or 0)
    except Exception:
        return 0.0


def _us_price(client, symbol: str) -> float:
    # 주문가격은 전략 TOP5의 yfinance 가격을 우선 사용하고, 필요 시 보유잔고 현재가를 사용한다.
    try:
        row = score_ticker(symbol, market="미국") or {}
        return float(row.get("현재가", 0) or 0)
    except Exception:
        return 0.0


def _kr_buy_qty(client, symbol: str, target: int, price: float, cfg: AutoConfig) -> tuple[int, str]:
    if price <= 0:
        return 0, "현재가 조회 실패"
    try:
        raw = client.domestic_buying_power(symbol=symbol, reference_price=int(price))
        out = (raw or {}).get("output", {}) or {}
    except Exception as e:
        return 0, f"매수가능조회 실패: {type(e).__name__}: {e}"

    def iv(name):
        try:
            return int(float(str(out.get(name, 0) or 0).replace(",", "")))
        except Exception:
            return 0

    cash = max(iv("nrcvb_buy_amt"), iv("ord_psbl_cash"), iv("max_buy_amt"))
    max_qty = max(iv("nrcvb_buy_qty"), iv("max_buy_qty"))
    usable = min(int(target), int(cash * (1 - cfg.buying_power_buffer_pct / 100.0))) if cash > 0 else int(target)
    qty = int(usable // price)
    if max_qty > 0:
        qty = min(qty, max_qty)
    if qty <= 0:
        return 0, f"주문가능수량 0 (목표 {target:,}원 / 사용가능 {usable:,}원)"
    return qty, f"사용가능 {usable:,}원 / {qty}주"


def _us_buy_qty(client, symbol: str, target: float, price: float, cfg: AutoConfig, exchange="NASD") -> tuple[int, str]:
    if price <= 0:
        return 0, "현재가 조회 실패"
    try:
        raw = client.overseas_buying_power_us(symbol=symbol, limit_price=price, exchange=exchange)
        out = (raw or {}).get("output", {}) or {}
        if isinstance(out, list):
            out = out[0] if out else {}
    except Exception as e:
        return 0, f"매수가능조회 실패: {type(e).__name__}: {e}"

    def fv(name):
        try:
            return float(str(out.get(name, 0) or 0).replace(",", ""))
        except Exception:
            return 0.0

    cash = max(
        fv("ord_psbl_frcr_amt"), fv("ovrs_ord_psbl_amt"),
        fv("echm_af_ord_psbl_amt"), fv("frcr_ord_psbl_amt1")
    )
    max_qty = int(max(fv("max_ord_psbl_qty"), fv("ord_psbl_qty"), fv("ovrs_max_ord_psbl_qty")))
    usable = min(float(target), cash * (1 - cfg.buying_power_buffer_pct / 100.0)) if cash > 0 else float(target)
    qty = int(usable // price)
    if max_qty > 0:
        qty = min(qty, max_qty)
    if qty <= 0:
        return 0, f"주문가능수량 0 (목표 ${target:.2f} / 사용가능 ${usable:.2f})"
    return qty, f"사용가능 ${usable:.2f} / {qty}주"


def _confirm_kr_qty(client, symbol: str, before_qty: int, side: str, wait_seconds: int) -> tuple[bool, int]:
    deadline = time.time() + max(1, wait_seconds)
    last = before_qty
    while time.time() < deadline:
        time.sleep(2)
        try:
            h, _ = _domestic_holdings(client)
            last = int(h.get(symbol, {}).get("qty", 0))
            if side == "buy" and last > before_qty:
                return True, last
            if side == "sell" and last < before_qty:
                return True, last
        except Exception:
            pass
    return False, last


def _confirm_us_qty(client, symbol: str, before_qty: int, side: str, wait_seconds: int) -> tuple[bool, int]:
    deadline = time.time() + max(1, wait_seconds)
    last = before_qty
    while time.time() < deadline:
        time.sleep(2)
        try:
            h, _ = _overseas_holdings(client)
            last = int(h.get(symbol, {}).get("qty", 0))
            if side == "buy" and last > before_qty:
                return True, last
            if side == "sell" and last < before_qty:
                return True, last
        except Exception:
            pass
    return False, last


def _kr_order(client, symbol: str, side: str, qty: int, before_qty: int, reason: str, execute: bool, cfg: AutoConfig) -> dict:
    if qty <= 0:
        return {"status": "SKIP", "qty": 0, "reason": reason}
    if not execute:
        return {"status": "DRY", "qty": qty, "reason": reason}
    try:
        res = client.domestic_order(symbol, qty, side, market_order=True)
    except Exception as e:
        return {"status": "ERROR", "qty": qty, "reason": reason, "msg1": repr(e)}
    if str((res or {}).get("rt_cd", "")) != "0":
        return {"status": "REJECT", "qty": qty, "reason": reason, "msg1": str((res or {}).get("msg1", ""))}
    confirmed, after_qty = _confirm_kr_qty(client, symbol, before_qty, side, cfg.confirm_wait_seconds)
    return {
        "status": "FILLED" if confirmed else "ORDERED",
        "qty": qty,
        "reason": reason,
        "before_qty": before_qty,
        "after_qty": after_qty,
        "msg1": str((res or {}).get("msg1", "")),
    }


def _us_order(client, symbol: str, side: str, qty: int, price: float, before_qty: int, reason: str, execute: bool, cfg: AutoConfig, exchange="NASD") -> dict:
    if qty <= 0 or price <= 0:
        return {"status": "SKIP", "qty": qty, "reason": reason}
    if not execute:
        return {"status": "DRY", "qty": qty, "price": price, "reason": reason}
    try:
        res = client.overseas_order_us(symbol=symbol, qty=qty, side=side, limit_price=price, exchange=exchange)
    except Exception as e:
        return {"status": "ERROR", "qty": qty, "price": price, "reason": reason, "msg1": repr(e)}
    if str((res or {}).get("rt_cd", "")) != "0":
        return {"status": "REJECT", "qty": qty, "price": price, "reason": reason, "msg1": str((res or {}).get("msg1", ""))}
    confirmed, after_qty = _confirm_us_qty(client, symbol, before_qty, side, cfg.confirm_wait_seconds)
    return {
        "status": "FILLED" if confirmed else "ORDERED",
        "qty": qty,
        "price": price,
        "reason": reason,
        "before_qty": before_qty,
        "after_qty": after_qty,
        "msg1": str((res or {}).get("msg1", "")),
    }


def _top_map(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {}
    out = {}
    for _, r in df.iterrows():
        symbol = str(r.get("종목코드", "")).strip().upper()
        if symbol:
            out[symbol] = r.to_dict()
    return out


def run_kr_cycle(client, top5: pd.DataFrame, cfg: AutoConfig, execute_orders: bool) -> Dict[str, Any]:
    now = datetime.now(KST)
    state = _load(KR_STATE_FILE, KST)
    result = {"time": now.isoformat(timespec="seconds"), "market": "KR", "actions": [], "diagnostics": []}

    holdings, holdings_list = _domestic_holdings(client)
    result["holdings"] = holdings_list
    leaders = _top_map(top5)
    result["top5_symbols"] = list(leaders.keys())

    if now.weekday() >= 5 or not (dtime(9, 0) <= now.time() < dtime(15, 30)):
        result["message"] = "국내 정규장 주문시간 외"
        _save(KR_STATE_FILE, state)
        return result

    # 1) 15:15 이후: TOP5/state와 무관하게 모의계좌 실제 보유잔고 전량 청산.
    if now.time() >= _clock(cfg.kr_force_exit_time):
        targets = holdings
        if str(getattr(client, "env", "demo")).lower() == "real" or not cfg.force_exit_all_demo_holdings:
            tracked = set(state["positions"].keys())
            targets = {s: h for s, h in holdings.items() if s in tracked}
        for symbol, h in list(targets.items()):
            qty = int(h.get("qty", 0))
            if qty <= 0:
                continue
            act = _kr_order(client, symbol, "sell", qty, qty, f"당일 강제청산 {cfg.kr_force_exit_time}", execute_orders, cfg)
            result["actions"].append({"symbol": symbol, "name": h.get("name", symbol), "action": "FORCE_SELL", **act})
            if act.get("status") in ("FILLED", "ORDERED"):
                state["daily_orders"] += 1
                if act.get("status") == "FILLED" and int(act.get("after_qty", qty)) == 0:
                    state["positions"].pop(symbol, None)
        result["message"] = f"{cfg.kr_force_exit_time} 이후 실제 보유잔고 강제청산"
        _save(KR_STATE_FILE, state)
        return result

    # 2) 추적 보유종목 손절/익절/기술매도/추가매수.
    for symbol, pos in list(state["positions"].items()):
        h = holdings.get(symbol)
        if not h:
            state["positions"].pop(symbol, None)
            continue
        qty = int(h["qty"])
        avg = float(h.get("avg_price", 0) or 0)
        price = _kr_price(client, symbol) or float(h.get("current_price", 0) or 0)
        if qty <= 0 or avg <= 0 or price <= 0:
            continue
        pnl = (price / avg - 1.0) * 100.0
        sell_reason = None
        sell_qty = qty
        action = ""
        if pnl <= -abs(cfg.stop_loss_pct):
            sell_reason, action = f"손절 {pnl:.2f}%", "STOP_LOSS"
        elif pnl >= cfg.take2_pct:
            sell_reason, action = f"2차 익절 {pnl:.2f}%", "TAKE2"
        elif pnl >= cfg.take1_pct and not bool(pos.get("take1_done")):
            sell_qty = max(1, qty // 2)
            sell_reason, action = f"1차 익절 {pnl:.2f}%", "TAKE1"
        else:
            try:
                tech = score_ticker(symbol, market="국내") or {}
                if float(tech.get("순점수", 0) or 0) <= -4:
                    sell_reason, action = f"기술 매도신호 순점수 {tech.get('순점수')}", "TECH_SELL"
            except Exception:
                pass

        if sell_reason:
            act = _kr_order(client, symbol, "sell", sell_qty, qty, sell_reason, execute_orders, cfg)
            result["actions"].append({"symbol": symbol, "name": h.get("name", symbol), "action": action, "pnl": round(pnl, 2), **act})
            if act.get("status") in ("FILLED", "ORDERED"):
                state["daily_orders"] += 1
                if action == "TAKE1":
                    pos["take1_done"] = True
                elif act.get("status") == "FILLED" and int(act.get("after_qty", qty)) == 0:
                    state["positions"].pop(symbol, None)
            continue

        # 추가매수는 현재 TOP5에 남아 있을 때만 허용.
        if symbol not in leaders or now.time() >= _clock(cfg.kr_last_entry_time):
            continue
        stage = int(pos.get("buy_stage", 1))
        weights = [cfg.buy1_pct, cfg.buy2_pct, cfg.buy3_pct]
        total_w = max(1, sum(weights))
        next_stage = 2 if stage == 1 and pnl >= cfg.add2_trigger_pct else 3 if stage == 2 and pnl >= cfg.add3_trigger_pct else None
        if next_stage:
            target = int(cfg.kr_per_stock_budget * weights[next_stage - 1] / total_w)
            add_qty, bp_reason = _kr_buy_qty(client, symbol, target, price, cfg)
            cost = int(add_qty * price)
            if add_qty > 0 and float(state["daily_buy_amount"]) + cost <= cfg.kr_daily_budget:
                act = _kr_order(client, symbol, "buy", add_qty, qty, f"{next_stage}차 분할매수 +{pnl:.2f}% · {bp_reason}", execute_orders, cfg)
                result["actions"].append({"symbol": symbol, "name": h.get("name", symbol), "action": f"BUY{next_stage}", "pnl": round(pnl, 2), **act})
                if act.get("status") in ("FILLED", "ORDERED"):
                    pos["buy_stage"] = next_stage
                    state["daily_buy_amount"] = float(state["daily_buy_amount"]) + cost
                    state["daily_orders"] += 1

    # 3) 신규매수: 화면/worker TOP5에 실제로 보이는 종목만 가능.
    if now.time() < _clock(cfg.kr_last_entry_time):
        weights = [cfg.buy1_pct, cfg.buy2_pct, cfg.buy3_pct]
        first_budget = int(cfg.kr_per_stock_budget * weights[0] / max(1, sum(weights)))
        for symbol, row in leaders.items():
            if len(state["positions"]) >= cfg.max_positions:
                break
            if symbol in holdings or symbol in state["positions"]:
                continue
            score = float(row.get("종합점수", 0) or 0)
            if score < cfg.min_score:
                result["diagnostics"].append({"symbol": symbol, "reason": f"점수 미달 {score:.1f} < {cfg.min_score:.1f}"})
                continue
            price = _kr_price(client, symbol) or float(row.get("현재가", 0) or 0)
            qty, bp_reason = _kr_buy_qty(client, symbol, first_budget, price, cfg)
            cost = int(qty * price)
            if qty <= 0:
                result["diagnostics"].append({"symbol": symbol, "reason": bp_reason})
                continue
            if float(state["daily_buy_amount"]) + cost > cfg.kr_daily_budget:
                result["diagnostics"].append({"symbol": symbol, "reason": "일일 신규매수 한도 초과"})
                continue
            act = _kr_order(client, symbol, "buy", qty, 0, f"TOP5 신규매수 · 종합점수 {score:.1f} · {bp_reason}", execute_orders, cfg)
            result["actions"].append({"symbol": symbol, "name": row.get("종목명", symbol), "action": "BUY1", "combined_score": score, **act})
            if act.get("status") in ("FILLED", "ORDERED"):
                state["positions"][symbol] = {"buy_stage": 1, "take1_done": False, "opened_at": now.isoformat(timespec="seconds")}
                state["daily_buy_amount"] = float(state["daily_buy_amount"]) + cost
                state["daily_orders"] += 1

    # 결과 저장 전 한국투자 잔고를 다시 읽어 화면과 맞춘다.
    try:
        _, latest = _domestic_holdings(client)
        result["holdings"] = latest
    except Exception:
        pass
    result["state"] = state
    _save(KR_STATE_FILE, state)
    return result


def run_us_cycle(client, top5: pd.DataFrame, cfg: AutoConfig, execute_orders: bool) -> Dict[str, Any]:
    now = datetime.now(ET)
    state = _load(US_STATE_FILE, ET)
    result = {"time": now.isoformat(timespec="seconds"), "market": "US", "actions": [], "diagnostics": []}
    holdings, holdings_list = _overseas_holdings(client)
    result["holdings"] = holdings_list
    leaders = _top_map(top5)
    result["top5_symbols"] = list(leaders.keys())

    if now.weekday() >= 5 or not (dtime(9, 30) <= now.time() < dtime(16, 0)):
        result["message"] = "미국 정규장 주문시간 외"
        _save(US_STATE_FILE, state)
        return result

    if now.time() >= _clock(cfg.us_force_exit_time):
        targets = holdings
        if str(getattr(client, "env", "demo")).lower() == "real" or not cfg.force_exit_all_demo_holdings:
            tracked = set(state["positions"].keys())
            targets = {s: h for s, h in holdings.items() if s in tracked}
        for symbol, h in list(targets.items()):
            qty = int(h.get("qty", 0))
            price = _us_price(client, symbol) or float(h.get("current_price", 0) or 0)
            if qty <= 0 or price <= 0:
                continue
            act = _us_order(client, symbol, "sell", qty, price, qty, f"당일 강제청산 {cfg.us_force_exit_time} ET", execute_orders, cfg, h.get("exchange", "NASD"))
            result["actions"].append({"symbol": symbol, "name": h.get("name", symbol), "action": "FORCE_SELL", **act})
            if act.get("status") in ("FILLED", "ORDERED"):
                state["daily_orders"] += 1
                if act.get("status") == "FILLED" and int(act.get("after_qty", qty)) == 0:
                    state["positions"].pop(symbol, None)
        result["message"] = f"{cfg.us_force_exit_time} ET 이후 실제 보유잔고 강제청산"
        _save(US_STATE_FILE, state)
        return result

    for symbol, pos in list(state["positions"].items()):
        h = holdings.get(symbol)
        if not h:
            state["positions"].pop(symbol, None)
            continue
        qty = int(h["qty"])
        avg = float(h.get("avg_price", 0) or 0)
        price = _us_price(client, symbol) or float(h.get("current_price", 0) or 0)
        if qty <= 0 or avg <= 0 or price <= 0:
            continue
        pnl = (price / avg - 1.0) * 100.0
        sell_reason = None
        sell_qty = qty
        action = ""
        if pnl <= -abs(cfg.stop_loss_pct):
            sell_reason, action = f"손절 {pnl:.2f}%", "STOP_LOSS"
        elif pnl >= cfg.take2_pct:
            sell_reason, action = f"2차 익절 {pnl:.2f}%", "TAKE2"
        elif pnl >= cfg.take1_pct and not bool(pos.get("take1_done")):
            sell_qty = max(1, qty // 2)
            sell_reason, action = f"1차 익절 {pnl:.2f}%", "TAKE1"
        else:
            try:
                tech = score_ticker(symbol, market="미국") or {}
                if float(tech.get("순점수", 0) or 0) <= -4:
                    sell_reason, action = f"기술 매도신호 순점수 {tech.get('순점수')}", "TECH_SELL"
            except Exception:
                pass
        if sell_reason:
            act = _us_order(client, symbol, "sell", sell_qty, price, qty, sell_reason, execute_orders, cfg, h.get("exchange", "NASD"))
            result["actions"].append({"symbol": symbol, "name": h.get("name", symbol), "action": action, "pnl": round(pnl, 2), **act})
            if act.get("status") in ("FILLED", "ORDERED"):
                state["daily_orders"] += 1
                if action == "TAKE1":
                    pos["take1_done"] = True
                elif act.get("status") == "FILLED" and int(act.get("after_qty", qty)) == 0:
                    state["positions"].pop(symbol, None)
            continue

        if symbol not in leaders or now.time() >= _clock(cfg.us_last_entry_time):
            continue
        stage = int(pos.get("buy_stage", 1))
        weights = [cfg.buy1_pct, cfg.buy2_pct, cfg.buy3_pct]
        total_w = max(1, sum(weights))
        next_stage = 2 if stage == 1 and pnl >= cfg.add2_trigger_pct else 3 if stage == 2 and pnl >= cfg.add3_trigger_pct else None
        if next_stage:
            target = cfg.us_per_stock_budget_usd * weights[next_stage - 1] / total_w
            add_qty, bp_reason = _us_buy_qty(client, symbol, target, price, cfg, h.get("exchange", "NASD"))
            cost = add_qty * price
            if add_qty > 0 and float(state["daily_buy_amount"]) + cost <= cfg.us_daily_budget_usd:
                act = _us_order(client, symbol, "buy", add_qty, price, qty, f"{next_stage}차 분할매수 +{pnl:.2f}% · {bp_reason}", execute_orders, cfg, h.get("exchange", "NASD"))
                result["actions"].append({"symbol": symbol, "name": h.get("name", symbol), "action": f"BUY{next_stage}", "pnl": round(pnl, 2), **act})
                if act.get("status") in ("FILLED", "ORDERED"):
                    pos["buy_stage"] = next_stage
                    state["daily_buy_amount"] = float(state["daily_buy_amount"]) + cost
                    state["daily_orders"] += 1

    if now.time() < _clock(cfg.us_last_entry_time):
        weights = [cfg.buy1_pct, cfg.buy2_pct, cfg.buy3_pct]
        first_budget = cfg.us_per_stock_budget_usd * weights[0] / max(1, sum(weights))
        for symbol, row in leaders.items():
            if len(state["positions"]) >= cfg.max_positions:
                break
            if symbol in holdings or symbol in state["positions"]:
                continue
            score = float(row.get("종합점수", 0) or 0)
            if score < cfg.min_score:
                result["diagnostics"].append({"symbol": symbol, "reason": f"점수 미달 {score:.1f} < {cfg.min_score:.1f}"})
                continue
            price = float(row.get("현재가", 0) or 0) or _us_price(client, symbol)
            qty, bp_reason = _us_buy_qty(client, symbol, first_budget, price, cfg)
            cost = qty * price
            if qty <= 0:
                result["diagnostics"].append({"symbol": symbol, "reason": bp_reason})
                continue
            if float(state["daily_buy_amount"]) + cost > cfg.us_daily_budget_usd:
                result["diagnostics"].append({"symbol": symbol, "reason": "일일 신규매수 한도 초과"})
                continue
            act = _us_order(client, symbol, "buy", qty, price, 0, f"TOP5 신규매수 · 종합점수 {score:.1f} · {bp_reason}", execute_orders, cfg)
            result["actions"].append({"symbol": symbol, "name": symbol, "action": "BUY1", "combined_score": score, **act})
            if act.get("status") in ("FILLED", "ORDERED"):
                state["positions"][symbol] = {"buy_stage": 1, "take1_done": False, "opened_at": now.isoformat(timespec="seconds")}
                state["daily_buy_amount"] = float(state["daily_buy_amount"]) + cost
                state["daily_orders"] += 1

    try:
        _, latest = _overseas_holdings(client)
        result["holdings"] = latest
    except Exception:
        pass
    result["state"] = state
    _save(US_STATE_FILE, state)
    return result

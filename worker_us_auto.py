from __future__ import annotations

import json
import os
import time
from datetime import datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from trader_core import Settings, KISClient, score_ticker
from ai_judge import analyze_market_with_ai, merge_ai_filter


NY = ZoneInfo("America/New_York")
STATE_DIR = Path(os.getenv("SONG_TRADER_STATE_DIR", "/tmp/song_trader"))
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "us_paper_auto_state.json"

DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META",
    "TSLA", "AMD", "GOOGL", "AVGO", "NFLX"
]

NASDAQ_DEFAULTS = {
    "AAPL", "MSFT", "NVDA", "AMZN", "META",
    "TSLA", "AMD", "GOOGL", "AVGO", "NFLX"
}


def now_et():
    return datetime.now(NY)


def market_open_now():
    now = now_et()
    if now.weekday() >= 5:
        return False
    return dtime(9, 30) <= now.time() < dtime(16, 0)


def load_state():
    today = now_et().strftime("%Y-%m-%d")
    fresh = {
        "date": today,
        "positions": {},
        "orders_today": 0,
        "last_scan": "",
    }

    if not STATE_FILE.exists():
        return fresh

    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return fresh

    if state.get("date") != today:
        return fresh

    state.setdefault("positions", {})
    state.setdefault("orders_today", 0)
    state.setdefault("last_scan", "")
    return state


def save_state(state):
    STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def kis_us_price(client, symbol: str, excd: str = "NAS") -> float:
    raw = client.get(
        "/uapi/overseas-price/v1/quotations/price",
        "HHDFS00000300",
        {"AUTH": "", "EXCD": excd, "SYMB": symbol},
    )
    out = (raw or {}).get("output", {}) or {}

    # KIS overseas-price responses may expose current price under several fields.
    for key in ("last", "stck_prpr", "ovrs_nmix_prpr", "clos", "base"):
        value = out.get(key)
        if value not in (None, ""):
            try:
                price = float(value)
                if price > 0:
                    return price
            except Exception:
                pass

    raise RuntimeError(f"{symbol} 현재가를 읽지 못했습니다: {out}")


def exchange_codes(symbol: str):
    # Current default universe is NASDAQ-heavy.
    # Add an explicit map here when NYSE/AMEX names are added.
    if symbol in NASDAQ_DEFAULTS:
        return "NAS", "NASD"
    return "NAS", "NASD"


def build_us_top5():
    rows = []
    for symbol in DEFAULT_UNIVERSE:
        try:
            row = score_ticker(symbol, market="미국")
            if row:
                rows.append(row)
        except Exception as e:
            print("score error", symbol, repr(e))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values(
        ["순점수", "거래량배수"],
        ascending=[False, False],
    ).head(5).reset_index(drop=True)

    tech100 = ((df["순점수"].clip(-6, 6) + 6) / 12 * 100).astype(float)
    vol_bonus = (df["거래량배수"].clip(0, 2.0) / 2.0 * 10).astype(float)
    df["종합점수"] = (tech100 * 0.9 + vol_bonus).round(1)

    df["종목코드"] = df["종목"].astype(str)
    df["종목명"] = df["종목"].astype(str)
    df["판정"] = df["종합신호"]
    df["진입근거"] = (
        "RSI " + df["RSI"].astype(str)
        + " / 거래량배수 " + df["거래량배수"].astype(str)
        + " / 매수점수 " + df["매수점수"].astype(str)
        + " / 매도점수 " + df["매도점수"].astype(str)
    )

    labels = ["1위", "2위", "3위", "4위", "5위"]
    df.insert(0, "순위", labels[:len(df)])
    return df


def paper_order(client, symbol: str, side: str, qty: int, price: float):
    _, order_exchange = exchange_codes(symbol)

    # trader_core.overseas_order_us() already sends:
    # demo buy -> VTTT1002U
    # demo sell -> VTTT1001U
    # ORD_DVSN 00 (limit order)
    return client.overseas_order_us(
        symbol=symbol,
        qty=int(qty),
        side=side,
        limit_price=float(price),
        exchange=order_exchange,
    )


def order_ok(resp):
    return isinstance(resp, dict) and str(resp.get("rt_cd", "")) == "0"


def run_cycle():
    settings = Settings.from_env()
    client = KISClient(settings, env="demo")
    client.get_token()

    state = load_state()
    now = now_et()

    if not market_open_now():
        print(now.isoformat(), "US market closed")
        save_state(state)
        return

    max_positions = int(os.getenv("US_MAX_POSITIONS", "2"))
    max_orders = int(os.getenv("US_MAX_DAILY_ORDERS", "6"))
    min_combined = float(os.getenv("US_MIN_COMBINED_SCORE", "55"))
    stop_loss = float(os.getenv("US_STOP_LOSS_PCT", "3.0"))
    take_profit = float(os.getenv("US_TAKE_PROFIT_PCT", "3.0"))
    force_exit_hhmm = os.getenv("US_FORCE_EXIT_ET", "15:50")
    force_h, force_m = [int(x) for x in force_exit_hhmm.split(":")]

    # ---- Existing bot positions: sell automatically by fixed risk rules ----
    for symbol, pos in list(state["positions"].items()):
        try:
            quote_excd, _ = exchange_codes(symbol)
            price = kis_us_price(client, symbol, quote_excd)
        except Exception as e:
            print("price error", symbol, repr(e))
            continue

        avg = float(pos.get("entry_price", price))
        qty = int(pos.get("qty", 1))
        pnl = (price / avg - 1) * 100 if avg > 0 else 0.0

        sell_reason = None
        if now.time() >= dtime(force_h, force_m):
            sell_reason = "force_exit"
        elif pnl <= -abs(stop_loss):
            sell_reason = "stop_loss"
        elif pnl >= take_profit:
            sell_reason = "take_profit"

        if sell_reason and state["orders_today"] < max_orders:
            try:
                # For a sell limit, place close to the observed price.
                resp = paper_order(client, symbol, "sell", qty, price)
                print("SELL", symbol, qty, price, sell_reason, resp)
                if order_ok(resp):
                    state["orders_today"] += 1
                    state["positions"].pop(symbol, None)
            except Exception as e:
                print("sell order error", symbol, repr(e))

    # No new entries late in session
    if now.time() >= dtime(15, 30):
        save_state(state)
        return

    if len(state["positions"]) >= max_positions or state["orders_today"] >= max_orders:
        save_state(state)
        return

    # ---- Scan automatically ----
    leaders = build_us_top5()
    if leaders.empty:
        print("no candidates")
        save_state(state)
        return

    # Fast AI only searches if there is a green buy candidate.
    # Use env-backed settings object compatible with ai_judge.
    try:
        ai_result = analyze_market_with_ai(
            leaders,
            os.environ,
            strategy_name="미국 기술·모멘텀 자동매매",
            market="미국",
        )
        filtered = merge_ai_filter(
            leaders,
            ai_result,
            min_ai_score=int(os.getenv("AI_MIN_SCORE", "60")),
            min_confidence=int(os.getenv("AI_MIN_CONFIDENCE", "55")),
        )
    except Exception as e:
        print("AI filter error:", repr(e))
        save_state(state)
        return

    # If AI skipped because no buy candidate, there is nothing to do.
    if filtered.empty or "AI통과" not in filtered.columns:
        save_state(state)
        return

    passed = filtered[
        (filtered["AI통과"] == True)
        & (pd.to_numeric(filtered["종합점수"], errors="coerce").fillna(0) >= min_combined)
        & (filtered["판정"].astype(str).str.contains("매수", na=False))
    ].copy()

    if passed.empty:
        print("no final US buy candidate")
        save_state(state)
        return

    # ---- Automatic paper buy; no UI button ----
    for _, row in passed.iterrows():
        if len(state["positions"]) >= max_positions or state["orders_today"] >= max_orders:
            break

        symbol = str(row["종목코드"]).upper()
        if symbol in state["positions"]:
            continue

        try:
            quote_excd, _ = exchange_codes(symbol)
            current = kis_us_price(client, symbol, quote_excd)

            # Paper API supports limit order. Slightly above current for buy test
            # to improve fill probability without using a market order.
            limit_price = round(current * 1.002, 2)

            resp = paper_order(client, symbol, "buy", 1, limit_price)
            print("BUY", symbol, 1, limit_price, resp)

            if order_ok(resp):
                state["orders_today"] += 1
                state["positions"][symbol] = {
                    "qty": 1,
                    "entry_price": limit_price,
                    "created_at": now.isoformat(),
                    "ai_score": float(row.get("AI점수", 0)),
                    "combined_score": float(row.get("종합점수", 0)),
                }
        except Exception as e:
            print("buy order error", symbol, repr(e))

    state["last_scan"] = now.isoformat()
    save_state(state)


def main():
    # This file is intentionally PAPER-ONLY.
    # It refuses to run unless explicitly enabled.
    if os.getenv("ENABLE_US_PAPER_AUTO", "false").lower() != "true":
        raise SystemExit(
            "ENABLE_US_PAPER_AUTO=true 로 설정해야 미국 모의자동매매 워커가 실행됩니다."
        )

    poll_seconds = max(60, int(os.getenv("US_POLL_SECONDS", "180")))
    print("US paper auto worker started. poll=", poll_seconds)

    while True:
        try:
            run_cycle()
        except Exception as e:
            print("cycle error:", repr(e))
        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()

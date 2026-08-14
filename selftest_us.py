from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# 반드시 auto_engine import 전에 상태경로를 격리한다.
_tmp = tempfile.TemporaryDirectory()
os.environ["SONG_TRADER_STATE_DIR"] = _tmp.name

import auto_engine as ae
from strategy_us import _score_frame

ET = ZoneInfo("America/New_York")
FIXED = datetime(2026, 8, 14, 10, 30, tzinfo=ET)
ae._now_et = lambda: FIXED


class FakeClient:
    env = "demo"

    def __init__(self):
        self.qty = 0
        self.avg = 100.0
        self.price = 100.0
        self.orders = []

    def overseas_balance_us(self, exchange="NASD", currency="USD"):
        if exchange != "NASD" or self.qty <= 0:
            return {"rt_cd": "0", "output1": [], "msg1": ""}
        return {
            "rt_cd": "0",
            "output1": [{
                "ovrs_pdno": "AAPL",
                "ovrs_item_name": "AAPL",
                "ovrs_cblc_qty": str(self.qty),
                "ord_psbl_qty": str(self.qty),
                "pchs_avg_pric": str(self.avg),
                "now_pric2": str(self.price),
                "ovrs_excg_cd": "NASD",
            }],
        }

    def get(self, path, tr_id, params):
        return {"rt_cd": "0", "output": {"last": str(self.price)}}

    def overseas_order_us(self, symbol, qty, side, limit_price, exchange="NASD"):
        self.orders.append((symbol, int(qty), side, float(limit_price), exchange))
        return {"rt_cd": "0", "msg_cd": "0", "msg1": "정상접수"}


def leader(score=85.0):
    return pd.DataFrame([{
        "종목코드": "AAPL",
        "종목명": "AAPL",
        "거래소": "NASD",
        "판정": "🟢 매수 후보",
        "종합점수": score,
        "주도주점수": score,
        "최근5분수익률": 0.8,
        "최근10분수익률": 1.2,
        "거래량배수": 1.8,
        "고점대비": -0.3,
        "모멘텀약화": False,
        "스캔시각": FIXED.isoformat(timespec="seconds"),
    }])


def assert_action(result, expected):
    actions = result.get("actions", [])
    assert actions, f"액션 없음: {result}"
    got = actions[0].get("action")
    assert got == expected, f"expected {expected}, got {got}: {actions}"
    return actions[0]


def test_strategy_score():
    n = 40
    close = np.linspace(100, 103.2, n)
    vol = np.concatenate([np.full(n - 5, 1000.0), np.full(5, 2400.0)])
    frame = pd.DataFrame({
        "Open": close - 0.05,
        "High": close + 0.10,
        "Low": close - 0.12,
        "Close": close,
        "Volume": vol,
    })
    row = _score_frame("AAPL", frame)
    assert row is not None
    assert row["종합점수"] >= 58.0, row
    assert "매수 후보" in row["판정"], row


def test_order_sequence():
    ae.reset_us_state()
    client = FakeClient()
    cfg = ae.AutoConfig(
        us_daily_budget_usd=5000,
        us_per_stock_budget_usd=1500,
        max_positions=3,
        min_score=50,
        take1_pct=3.0,
        take2_pct=5.0,
        us_add2_trigger_pct=0.4,
        us_profit_guard_trigger_pct=1.2,
        us_profit_guard_drawdown_pct=0.8,
    )

    # 1) 신규 1차매수
    r1 = ae.run_us_cycle(client, leader(), cfg, True, source="SELFTEST")
    a1 = assert_action(r1, "BUY1")
    q1 = int(a1["qty"])
    assert q1 > 0
    assert len(client.orders) == 1

    # 2) 잔고 반영 전에는 같은 BUY1 재주문 금지
    r2 = ae.run_us_cycle(client, leader(), cfg, True, source="SELFTEST")
    assert len(client.orders) == 1, client.orders
    assert any(d.get("action") in ("WAIT_PENDING_ORDER", "PENDING_TIMEOUT_LOCKED") for d in r2.get("diagnostics", []))

    # 3) BUY1 체결 확인 후 +0.5%, 모멘텀 유지 -> BUY2
    client.qty = q1
    client.avg = 100.0
    client.price = 100.5
    r3 = ae.run_us_cycle(client, leader(), cfg, True, source="SELFTEST")
    a3 = assert_action(r3, "BUY2")
    q2 = int(a3["qty"])
    assert q2 > 0
    assert len(client.orders) == 2

    # 4) BUY2 체결 확인 후 +5.5%여도 TAKE1부터
    client.qty = q1 + q2
    client.price = 105.5
    r4 = ae.run_us_cycle(client, leader(), cfg, True, source="SELFTEST")
    a4 = assert_action(r4, "TAKE1")
    sell1 = int(a4["qty"])
    assert sell1 > 0
    assert len(client.orders) == 3

    # 5) TAKE1 체결 확인 후에야 TAKE2
    client.qty = q1 + q2 - sell1
    r5 = ae.run_us_cycle(client, leader(), cfg, True, source="SELFTEST")
    a5 = assert_action(r5, "TAKE2")
    assert len(client.orders) == 4

    # 6) TAKE2 체결 완료 후 실제잔고 0이면 추적 종료
    client.qty = 0
    r6 = ae.run_us_cycle(client, leader(score=10), cfg, True, source="SELFTEST")
    state = ae.load_us_state()
    assert "AAPL" not in state.get("positions", {}), state


if __name__ == "__main__":
    test_strategy_score()
    test_order_sequence()
    print("SELFTEST PASS: 전략 점수 / 2회 매수 / pending 중복차단 / TAKE1→TAKE2 / 청산확인")

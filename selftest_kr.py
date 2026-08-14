from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

_tmp = tempfile.TemporaryDirectory()
os.environ["SONG_TRADER_STATE_DIR"] = _tmp.name

import auto_engine as ae
from strategy_kr import build_kr_top5, _score_candidate

KST = ZoneInfo("Asia/Seoul")
FIXED = datetime(2026, 8, 14, 10, 30, tzinfo=KST)
ae._now_kst = lambda: FIXED


def make_intraday(symbol: str, rising: bool = True) -> dict:
    n = 30
    if rising:
        close = np.linspace(10000, 10220, n)
        close[-6:] += np.array([0, 15, 30, 50, 75, 105])
        vol = np.concatenate([np.full(n - 3, 1000.0), np.full(3, 2600.0)])
    else:
        close = np.linspace(10400, 10200, n)
        close[-10:] = np.linspace(10240, 10180, 10)
        vol = np.concatenate([np.full(n - 3, 1800.0), np.full(3, 700.0)])

    start = FIXED - timedelta(minutes=n)
    rows = []
    for i in range(n):
        t = (start + timedelta(minutes=i)).strftime("%H%M%S")
        c = float(close[i])
        rows.append({
            "stck_cntg_hour": t,
            "stck_oprc": str(int(c - 8)),
            "stck_hgpr": str(int(c + 12)),
            "stck_lwpr": str(int(c - 14)),
            "stck_prpr": str(int(c)),
            "cntg_vol": str(int(vol[i])),
        })
    return {"rt_cd": "0", "output2": rows}


class StrategyClient:
    env = "demo"

    def domestic_volume_rank(self):
        return {
            "rt_cd": "0",
            "output": [
                {
                    "mksc_shrn_iscd": "111111",
                    "hts_kor_isnm": "빠른종목",
                    "stck_prpr": "10300",
                    "prdy_ctrt": "4.5",
                    "acml_vol": "2000000",
                    "acml_tr_pbmn": "22000000000",
                },
                {
                    "mksc_shrn_iscd": "222222",
                    "hts_kor_isnm": "오전급등후약화",
                    "stck_prpr": "10180",
                    "prdy_ctrt": "11.0",
                    "acml_vol": "3500000",
                    "acml_tr_pbmn": "30000000000",
                },
            ],
        }

    def domestic_intraday_minutes(self, symbol, *args, **kwargs):
        return make_intraday(symbol, rising=(symbol == "111111"))


class FakeClient:
    env = "demo"

    def __init__(self):
        self.qty = 0
        self.avg = 10000.0
        self.price = 10000.0
        self.orders = []

    def domestic_balance(self):
        if self.qty <= 0:
            return {"rt_cd": "0", "output1": [], "msg1": ""}
        return {
            "rt_cd": "0",
            "output1": [{
                "pdno": "111111",
                "prdt_name": "빠른종목",
                "hldg_qty": str(self.qty),
                "ord_psbl_qty": str(self.qty),
                "pchs_avg_pric": str(self.avg),
                "prpr": str(self.price),
            }],
        }

    def domestic_price(self, code):
        return {"rt_cd": "0", "output": {"stck_prpr": str(int(self.price))}}

    def domestic_buying_power(self, symbol, reference_price=0):
        return {
            "rt_cd": "0",
            "output": {
                "nrcvb_buy_amt": "10000000",
                "nrcvb_buy_qty": "10000",
                "ord_psbl_cash": "10000000",
                "max_buy_amt": "10000000",
                "max_buy_qty": "10000",
            },
        }

    def domestic_order(self, code, qty, side, price=0, market_order=True):
        self.orders.append((code, int(qty), side))
        return {"rt_cd": "0", "msg_cd": "0", "msg1": "정상접수"}


def leader(score=85.0):
    return pd.DataFrame([{
        "종목코드": "111111",
        "종목명": "빠른종목",
        "판정": "🟢 매수 후보",
        "종합점수": score,
        "주도주점수": 80.0,
        "최근3분수익률": 0.45,
        "최근5분수익률": 0.80,
        "최근10분수익률": 1.20,
        "거래량배수": 1.8,
        "고점대비": -0.2,
        "모멘텀약화": False,
        "스캔시각": FIXED.isoformat(timespec="seconds"),
        "순위": "1위",
    }])


def assert_action(result, expected):
    actions = result.get("actions", [])
    assert actions, f"액션 없음: {result}"
    got = actions[0].get("action")
    assert got == expected, f"expected {expected}, got {got}: {actions}"
    return actions[0]


def test_strategy_fast_vs_stale():
    df = build_kr_top5(StrategyClient())
    assert not df.empty, df
    first = df.iloc[0]
    assert first["종목코드"] == "111111", df
    assert "매수 후보" in str(first["판정"]), df
    stale = df[df["종목코드"] == "222222"]
    if not stale.empty:
        assert "매수 후보" not in str(stale.iloc[0]["판정"]), stale


def test_order_sequence():
    ae.reset_today_state()
    client = FakeClient()
    cfg = ae.AutoConfig(
        kr_daily_budget=10_000_000,
        kr_per_stock_budget=3_000_000,
        max_positions=3,
        min_score=50,
        take1_pct=3.0,
        take2_pct=5.0,
        kr_add2_trigger_pct=0.4,
        kr_profit_guard_trigger_pct=1.2,
        kr_profit_guard_drawdown_pct=0.8,
        duplicate_guard_seconds=90,
    )

    # 1차 매수
    r1 = ae.run_kr_cycle(client, leader(), cfg, True, source="SELFTEST")
    a1 = assert_action(r1, "BUY1")
    q1 = int(a1["qty"])
    assert q1 > 0 and len(client.orders) == 1

    # 잔고 반영 전 중복 주문 차단
    r2 = ae.run_kr_cycle(client, leader(), cfg, True, source="SELFTEST")
    assert len(client.orders) == 1, client.orders
    assert any(d.get("action") in ("WAIT_PENDING_ORDER", "PENDING_TIMEOUT_LOCKED") for d in r2.get("diagnostics", []))

    # BUY1 체결 확인 후 +0.5%, 모멘텀 유지 -> BUY2
    client.qty = q1
    client.avg = 10000.0
    client.price = 10050.0
    r3 = ae.run_kr_cycle(client, leader(), cfg, True, source="SELFTEST")
    a3 = assert_action(r3, "BUY2")
    q2 = int(a3["qty"])
    assert q2 > 0 and len(client.orders) == 2, client.orders

    # BUY2 체결 후 +5.5%여도 TAKE1부터
    client.qty = q1 + q2
    client.price = 10550.0
    r4 = ae.run_kr_cycle(client, leader(), cfg, True, source="SELFTEST")
    a4 = assert_action(r4, "TAKE1")
    sell1 = int(a4["qty"])
    assert sell1 > 0 and len(client.orders) == 3, client.orders

    # TAKE1 체결 확인 후에만 TAKE2
    client.qty = q1 + q2 - sell1
    r5 = ae.run_kr_cycle(client, leader(), cfg, True, source="SELFTEST")
    a5 = assert_action(r5, "TAKE2")
    assert len(client.orders) == 4, client.orders

    # 최종 잔고 0이면 추적 종료
    client.qty = 0
    ae.run_kr_cycle(client, pd.DataFrame(), cfg, True, source="SELFTEST")
    state = ae.load_state()
    assert "111111" not in state.get("positions", {}), state


def test_stale_signal_blocks_entry():
    ae.reset_today_state()
    client = FakeClient()
    cfg = ae.AutoConfig(kr_signal_max_age_seconds=120)
    stale = leader().copy()
    stale.loc[0, "스캔시각"] = (FIXED - timedelta(minutes=10)).isoformat(timespec="seconds")
    result = ae.run_kr_cycle(client, stale, cfg, True, source="SELFTEST")
    assert not result.get("actions"), result
    assert any(d.get("action") == "SKIP_STALE_SIGNAL" for d in result.get("diagnostics", [])), result


if __name__ == "__main__":
    test_strategy_fast_vs_stale()
    test_order_sequence()
    test_stale_signal_blocks_entry()
    print("SELFTEST KR PASS: 빠른모멘텀 선별 / 후행강세 차단 / 2회매수 / pending / TAKE1→TAKE2 / stale 차단")

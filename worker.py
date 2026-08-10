"""
쏭 자동매매 24시간 워커 v3
기본값은 DRY-RUN + 모의투자입니다.

실제 모의주문:
  TRADING_MODE=demo
  ENABLE_AUTO_ORDERS=true

실전은 추가 잠금까지 필요:
  TRADING_MODE=real
  ALLOW_LIVE_TRADING=true
  LIVE_UNLOCK_PHRASE=<secrets와 같은 문구>
  ENABLE_AUTO_ORDERS=true

주의: 워커는 대장주 후보를 매 사이클 새로 계산합니다.
"""
import os
import time
import pandas as pd

from trader_core import Settings, KISClient, discover_domestic_candidates, score_ticker
from auto_engine import AutoConfig, run_domestic_cycle

POLL_SECONDS = max(60, int(os.getenv("POLL_SECONDS", "60")))
MODE = os.getenv("TRADING_MODE", "demo")
ENABLE_AUTO_ORDERS = os.getenv("ENABLE_AUTO_ORDERS", "false").lower() == "true"

def build_leaders(client):
    candidates = discover_domestic_candidates(client, top_n=20)
    if candidates.empty:
        return pd.DataFrame()

    rows = []
    for _, r in candidates.head(12).iterrows():
        code = str(r["종목코드"]).zfill(6)
        try:
            tech = score_ticker(code, "국내")
        except Exception:
            tech = None
        if not tech:
            continue

        lead = float(r.get("주도주점수", 0))
        net = int(tech.get("순점수", 0))
        tech100 = max(0.0, min(100.0, ((net + 6) / 12) * 100))
        combined = lead * 0.65 + tech100 * 0.35

        rows.append({
            "종목코드": code,
            "종목명": r.get("종목명", ""),
            "현재가": r.get("현재가", ""),
            "등락률": r.get("등락률", ""),
            "주도주점수": round(lead, 1),
            "RSI": tech.get("RSI"),
            "거래량배수": tech.get("거래량배수"),
            "매수점수": tech.get("매수점수"),
            "매도점수": tech.get("매도점수"),
            "기술순점수": net,
            "종합점수": round(combined, 1),
            "판정": tech.get("종합신호"),
        })

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(["종합점수","주도주점수","기술순점수"], ascending=False)
        .head(5)
        .reset_index(drop=True)
    )

def main():
    settings = Settings.from_env()

    if MODE == "real":
        phrase_ok = os.getenv("LIVE_UNLOCK_PHRASE", "") == settings.live_unlock_phrase
        if not (settings.allow_live and phrase_ok):
            raise SystemExit("실전 잠금이 해제되지 않았습니다.")

    client = KISClient(settings, MODE)
    client.get_token()

    cfg = AutoConfig(
        daily_budget=int(os.getenv("MAX_DAILY_BUY_KRW", "300000")),
        per_stock_budget=int(os.getenv("PER_STOCK_KRW", "100000")),
        max_positions=int(os.getenv("MAX_POSITIONS", "3")),
        stop_loss_pct=float(os.getenv("STOP_LOSS_PCT", "3.0")),
        take1_pct=float(os.getenv("TAKE1_PCT", "3.0")),
        take2_pct=float(os.getenv("TAKE2_PCT", "5.0")),
    )

    print("worker v3 started", MODE, "orders=", ENABLE_AUTO_ORDERS)

    while True:
        try:
            leaders = build_leaders(client)
            cycle = run_domestic_cycle(
                client,
                leaders,
                cfg,
                execute_orders=ENABLE_AUTO_ORDERS
            )
            print(cycle.get("time"), cycle.get("message", ""), cycle.get("actions", []))
        except Exception as e:
            print("worker error:", repr(e))
        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()

"""
24시간 자동매매 워커 v1.

중요:
- 기본은 모의투자(demo).
- 실전(real)은 환경변수 ALLOW_LIVE_TRADING=true 와 LIVE_UNLOCK_PHRASE 일치가 모두 필요.
- v1은 국내 자동 후보탐색을 중심으로 구성.
- 미국 자동주문은 후보/점수 모듈은 포함하지만, 실제 자동 진입은 운영 전 추가 체결조회/잔고동기화 검증 권장.
"""

import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from trader_core import (
    Settings, KISClient, discover_domestic_candidates,
    score_ticker, is_market_open, append_trade_log
)

POLL_SECONDS = int(os.getenv("POLL_SECONDS","60"))
MODE = os.getenv("TRADING_MODE","demo")  # demo | real
MARKET = os.getenv("TRADING_MARKET","KR") # KR | US

MAX_DAILY_BUY_KRW = int(os.getenv("MAX_DAILY_BUY_KRW","300000"))
PER_STOCK_KRW = int(os.getenv("PER_STOCK_KRW","100000"))
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS","3"))
BUY_SCORE_MIN = int(os.getenv("BUY_SCORE_MIN","4"))

def live_allowed(settings):
    return (
        MODE != "real"
        or (
            settings.allow_live
            and os.getenv("LIVE_UNLOCK_PHRASE","") == settings.live_unlock_phrase
        )
    )

def main():
    settings = Settings.from_env()

    if MODE == "real" and not live_allowed(settings):
        raise SystemExit("실전 잠금이 해제되지 않았습니다.")

    client = KISClient(settings, MODE)
    client.get_token()
    print("worker started:", MODE, MARKET)

    # v1에서는 자동 '주문 실행'보다 후보 탐지 + 로그를 기본으로 둡니다.
    # 실제 자동 주문을 켜려면 ENABLE_AUTO_ORDERS=true 필요.
    auto_orders = os.getenv("ENABLE_AUTO_ORDERS","false").lower() == "true"

    while True:
        try:
            if MARKET == "KR":
                if is_market_open("KR"):
                    candidates = discover_domestic_candidates(client, top_n=15)
                    for code in candidates.get("종목코드", [])[:10]:
                        row = score_ticker(str(code), "국내")
                        if not row:
                            continue
                        if row["매수점수"] >= BUY_SCORE_MIN and row["순점수"] >= BUY_SCORE_MIN:
                            print("BUY CANDIDATE", row)
                            append_trade_log({
                                "time": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
                                "mode": MODE,
                                "market":"KR",
                                "symbol":code,
                                "event":"BUY_CANDIDATE",
                                "score":row["순점수"],
                                "price":row["현재가"],
                            })

                            # 안전상 v1은 자동 주문 플래그가 명시적으로 켜진 경우만 1차 주문 실행.
                            if auto_orders:
                                qty = max(1, int((PER_STOCK_KRW * 0.40) / max(row["현재가"],1)))
                                res = client.domestic_order(str(code), qty, "buy", market_order=True)
                                append_trade_log({
                                    "time": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
                                    "mode": MODE, "market":"KR", "symbol":code,
                                    "event":"BUY_ORDER", "qty":qty,
                                    "response":str(res)[:1000],
                                })
                else:
                    print("KR market closed")
            else:
                # 미국 v1: 기본 후보를 기술점수로 감시.
                # 지정가 주문은 실제 bid/ask/호가 기반 가격 산정이 필요하므로,
                # 자동주문은 추후 실시간 시세/체결조회 모듈 검증 후 활성화 권장.
                if is_market_open("US"):
                    for symbol in ["AAPL","MSFT","NVDA","AMZN","META","TSLA","AMD","GOOGL","AVGO","NFLX"]:
                        row = score_ticker(symbol, "미국")
                        if row and row["순점수"] >= BUY_SCORE_MIN:
                            print("US BUY CANDIDATE", row)
                            append_trade_log({
                                "time": datetime.now(ZoneInfo("America/New_York")).isoformat(),
                                "mode": MODE, "market":"US", "symbol":symbol,
                                "event":"BUY_CANDIDATE", "score":row["순점수"],
                                "price":row["현재가"],
                            })
                else:
                    print("US market closed")
        except Exception as e:
            print("worker error:", repr(e))

        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()

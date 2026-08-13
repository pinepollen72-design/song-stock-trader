# auto_engine.py 미국장 부분 교체용 패치
# 아래 함수/도우미들을 기존 auto_engine.py의 같은 이름 함수들과 교체하세요.
# trader_core.py에는 parse_overseas_holdings / merge_overseas_holdings가 있어야 합니다.

from trader_core import (
    append_trade_log,
    parse_overseas_holdings,
    merge_overseas_holdings,
)


def _overseas_all_balances(client):
    responses = []
    errors = []

    for exchange in ("NASD", "NYSE", "AMEX"):
        try:
            raw = client.overseas_balance_us(
                exchange=exchange,
                currency="USD",
            )
        except Exception as e:
            errors.append(
                f"{exchange}: {type(e).__name__}: {e}"
            )
            continue

        if not isinstance(raw, dict):
            errors.append(f"{exchange}: 응답 형식 이상")
            continue

        rt_cd = str(raw.get("rt_cd", ""))
        if rt_cd and rt_cd != "0":
            errors.append(
                (
                    f"{exchange}: "
                    f"{raw.get('msg_cd', '')} "
                    f"{raw.get('msg1', '')}"
                ).strip()
            )
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
        return {}, pd.DataFrame(), (
            f"미국 잔고 파싱 실패: {type(e).__name__}: {e}"
        )

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
            "sellable_qty": int(
                float(r.get("매도가능수량", 0) or 0)
            ),
            "avg_price": float(r.get("평균매입가", 0) or 0),
            "current_price": float(r.get("현재가", 0) or 0),
            "name": str(r.get("종목명", "") or symbol),
            "exchange": str(r.get("거래소", "") or "NASD").upper(),
        }

    return result, df, " / ".join(errors)


def _adopt_existing_us_holdings(
    client,
    state,
    holdings,
    config,
    result,
):
    """
    Railway 재배포/상태파일 초기화 후에도
    한국투자 모의계좌에 실제로 남아 있는 미국주식을
    자동매도 관리 대상으로 복구합니다.
    """
    if str(getattr(client, "env", "demo")).lower() != "demo":
        return

    for symbol, actual in holdings.items():
        if symbol in state["positions"]:
            # 거래소 정보가 없던 구버전 state 보정
            state["positions"][symbol].setdefault(
                "exchange",
                actual.get("exchange", "NASD"),
            )
            continue

        qty = int(actual.get("qty", 0) or 0)
        if qty <= 0:
            continue

        state["positions"][symbol] = {
            "name": actual.get("name", symbol),
            "created_at": _now_et().isoformat(),
            "buy_stage": 1,
            "avg_price": float(actual.get("avg_price", 0) or 0),
            "actual_qty": qty,
            "expected_qty": qty,
            "exchange": actual.get("exchange", "NASD"),
            "take1_sent": False,
            "exit_pending": False,
            "adopted_existing": True,
        }

        _diag(
            result,
            symbol,
            "ADOPT_EXISTING_US",
            (
                f"한국투자 실제 보유 {qty}주를 "
                f"자동 손절/익절/강제청산 관리 대상으로 복구"
            ),
            exchange=actual.get("exchange", "NASD"),
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
        return {"status": "SKIP", "reason": "주문수량 0"}
    if price <= 0:
        return {"status": "SKIP", "reason": "주문가격 0"}

    exchange = str(exchange or "NASD").upper()

    if not execute_orders:
        _us_event(
            state,
            f"DRY_{side.upper()}",
            symbol,
            f"{qty}주 @ ${price:.2f} · {exchange} · {reason}",
        )
        return {
            "status": "DRY",
            "qty": qty,
            "price": price,
            "exchange": exchange,
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
        return {
            "status": "ERROR",
            "error": repr(e),
            "exchange": exchange,
        }

    if _order_ok(res):
        state["daily_orders"] += 1
        _us_event(
            state,
            f"{side.upper()}_ORDER",
            symbol,
            f"{qty}주 @ ${price:.2f} · {exchange} · {reason} · {res}",
        )
        return {
            "status": "ORDERED",
            "qty": qty,
            "price": price,
            "exchange": exchange,
            "response": res,
        }

    _us_event(state, "ORDER_REJECT", symbol, str(res))
    return {
        "status": "REJECT",
        "qty": qty,
        "price": price,
        "exchange": exchange,
        "msg_cd": res.get("msg_cd", "") if isinstance(res, dict) else "",
        "msg1": res.get("msg1", "") if isinstance(res, dict) else "",
        "response": res,
    }

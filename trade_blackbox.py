from __future__ import annotations

import json
import os
import sqlite3
import sys
import uuid
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")

# Railway에서는 /data 를 Volume mount path로 쓰는 것을 권장합니다.
# 필요하면 Railway Variables에서 SONG_TRADER_DATA_DIR 값을 바꿀 수 있습니다.
DATA_DIR = Path(os.getenv("SONG_TRADER_DATA_DIR", "/data/song_trader"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "trade_blackbox.sqlite3"


def _now_iso() -> str:
    return datetime.now(KST).isoformat(timespec="seconds")


def _json_dumps(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
        )
    except Exception:
        return json.dumps({"raw": str(value)}, ensure_ascii=False)


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(
        DB_PATH,
        timeout=10,
        isolation_level=None,
    )
    conn.row_factory = sqlite3.Row

    # 단일 Worker + Railway Volume에서 안정적으로 쓰기 위한 설정
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


def init_blackbox() -> None:
    """
    거래 판단/주문/체결 기록 DB를 준비합니다.
    여러 번 호출해도 안전합니다.
    """
    with closing(_connect()) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS decision_log (
                id TEXT PRIMARY KEY,
                ts TEXT NOT NULL,
                market TEXT NOT NULL,
                symbol TEXT NOT NULL,

                event TEXT NOT NULL,
                reason_code TEXT NOT NULL,
                reason_text TEXT,

                price REAL,

                total_score REAL,
                leader_score REAL,
                entry_score REAL,
                risk_score REAL,

                day_change_pct REAL,
                relative_strength REAL,
                high_distance_pct REAL,
                vwap REAL,
                momentum_5m REAL,
                momentum_10m REAL,
                volume_accel REAL,

                position_qty INTEGER,
                order_qty INTEGER,
                stage TEXT,

                worker_run_id TEXT,
                extra_json TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_decision_ts
                ON decision_log(ts);

            CREATE INDEX IF NOT EXISTS idx_decision_symbol_ts
                ON decision_log(symbol, ts);

            CREATE TABLE IF NOT EXISTS order_log (
                id TEXT PRIMARY KEY,
                ts TEXT NOT NULL,
                market TEXT NOT NULL,
                symbol TEXT NOT NULL,

                side TEXT NOT NULL,
                event TEXT NOT NULL,
                reason_code TEXT NOT NULL,

                request_qty INTEGER,
                request_price REAL,

                kis_order_no TEXT,
                status TEXT,

                filled_qty INTEGER,
                filled_price REAL,

                error_code TEXT,
                error_message TEXT,

                worker_run_id TEXT,
                response_json TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_order_ts
                ON order_log(ts);

            CREATE INDEX IF NOT EXISTS idx_order_symbol_ts
                ON order_log(symbol, ts);
            """
        )


def _stdout(event_type: str, payload: Dict[str, Any]) -> None:
    """
    DB와 별도로 Railway 로그에도 한 줄 JSON을 남깁니다.
    """
    try:
        line = {
            "blackbox": event_type,
            **payload,
        }
        print(
            _json_dumps(line),
            file=sys.stdout,
            flush=True,
        )
    except Exception:
        pass


def log_decision(
    *,
    market: str,
    symbol: str,
    event: str,
    reason_code: str,
    reason_text: str = "",
    price: Optional[float] = None,

    total_score: Optional[float] = None,
    leader_score: Optional[float] = None,
    entry_score: Optional[float] = None,
    risk_score: Optional[float] = None,

    day_change_pct: Optional[float] = None,
    relative_strength: Optional[float] = None,
    high_distance_pct: Optional[float] = None,
    vwap: Optional[float] = None,
    momentum_5m: Optional[float] = None,
    momentum_10m: Optional[float] = None,
    volume_accel: Optional[float] = None,

    position_qty: Optional[int] = None,
    order_qty: Optional[int] = None,
    stage: str = "",
    worker_run_id: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """
    BUY / SELL / SKIP / WATCH 등 '판단'을 기록합니다.

    예)
    log_decision(
        market="US",
        symbol="NVDA",
        event="BUY",
        reason_code="BUY_LEADER_BREAKOUT",
        price=224.13,
        total_score=78.2,
        leader_score=82.0,
        order_qty=1,
    )
    """
    init_blackbox()

    row_id = uuid.uuid4().hex
    ts = _now_iso()

    values = (
        row_id,
        ts,
        str(market).upper(),
        str(symbol).upper(),
        str(event).upper(),
        str(reason_code).upper(),
        str(reason_text or ""),
        price,
        total_score,
        leader_score,
        entry_score,
        risk_score,
        day_change_pct,
        relative_strength,
        high_distance_pct,
        vwap,
        momentum_5m,
        momentum_10m,
        volume_accel,
        position_qty,
        order_qty,
        str(stage or ""),
        str(worker_run_id or ""),
        _json_dumps(extra or {}),
    )

    try:
        with closing(_connect()) as conn:
            conn.execute(
                """
                INSERT INTO decision_log (
                    id, ts, market, symbol,
                    event, reason_code, reason_text,
                    price,
                    total_score, leader_score, entry_score, risk_score,
                    day_change_pct, relative_strength, high_distance_pct,
                    vwap, momentum_5m, momentum_10m, volume_accel,
                    position_qty, order_qty, stage,
                    worker_run_id, extra_json
                )
                VALUES (
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?
                )
                """,
                values,
            )
    except Exception as exc:
        # 로그 실패가 실제 매매 Worker를 죽이지 않도록 합니다.
        _stdout(
            "DECISION_WRITE_ERROR",
            {
                "ts": ts,
                "market": market,
                "symbol": symbol,
                "event": event,
                "reason_code": reason_code,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        return row_id

    _stdout(
        "DECISION",
        {
            "id": row_id,
            "ts": ts,
            "market": str(market).upper(),
            "symbol": str(symbol).upper(),
            "event": str(event).upper(),
            "reason_code": str(reason_code).upper(),
            "price": price,
            "total_score": total_score,
            "position_qty": position_qty,
            "order_qty": order_qty,
        },
    )
    return row_id


def log_order(
    *,
    market: str,
    symbol: str,
    side: str,
    event: str,
    reason_code: str,

    request_qty: Optional[int] = None,
    request_price: Optional[float] = None,

    kis_order_no: str = "",
    status: str = "",

    filled_qty: Optional[int] = None,
    filled_price: Optional[float] = None,

    error_code: str = "",
    error_message: str = "",

    worker_run_id: str = "",
    response: Optional[Dict[str, Any]] = None,
) -> str:
    """
    KIS에 실제 주문을 보내기 직전/직후/체결확인 결과를 기록합니다.

    event 예:
      ORDER_SENT
      ORDER_ACCEPTED
      ORDER_REJECTED
      FILLED
      PARTIAL_FILLED
      ORDER_FAILED
    """
    init_blackbox()

    row_id = uuid.uuid4().hex
    ts = _now_iso()

    values = (
        row_id,
        ts,
        str(market).upper(),
        str(symbol).upper(),
        str(side).upper(),
        str(event).upper(),
        str(reason_code).upper(),
        request_qty,
        request_price,
        str(kis_order_no or ""),
        str(status or ""),
        filled_qty,
        filled_price,
        str(error_code or ""),
        str(error_message or ""),
        str(worker_run_id or ""),
        _json_dumps(response or {}),
    )

    try:
        with closing(_connect()) as conn:
            conn.execute(
                """
                INSERT INTO order_log (
                    id, ts, market, symbol,
                    side, event, reason_code,
                    request_qty, request_price,
                    kis_order_no, status,
                    filled_qty, filled_price,
                    error_code, error_message,
                    worker_run_id, response_json
                )
                VALUES (
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?
                )
                """,
                values,
            )
    except Exception as exc:
        _stdout(
            "ORDER_WRITE_ERROR",
            {
                "ts": ts,
                "market": market,
                "symbol": symbol,
                "side": side,
                "event": event,
                "reason_code": reason_code,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        return row_id

    _stdout(
        "ORDER",
        {
            "id": row_id,
            "ts": ts,
            "market": str(market).upper(),
            "symbol": str(symbol).upper(),
            "side": str(side).upper(),
            "event": str(event).upper(),
            "reason_code": str(reason_code).upper(),
            "request_qty": request_qty,
            "request_price": request_price,
            "kis_order_no": kis_order_no,
            "status": status,
            "filled_qty": filled_qty,
            "filled_price": filled_price,
        },
    )
    return row_id


def recent_decisions(
    limit: int = 100,
    market: str = "",
    symbol: str = "",
) -> List[Dict[str, Any]]:
    init_blackbox()

    where: List[str] = []
    params: List[Any] = []

    if market:
        where.append("market = ?")
        params.append(str(market).upper())

    if symbol:
        where.append("symbol = ?")
        params.append(str(symbol).upper())

    sql = "SELECT * FROM decision_log"
    if where:
        sql += " WHERE " + " AND ".join(where)

    sql += " ORDER BY ts DESC LIMIT ?"
    params.append(max(1, int(limit)))

    with closing(_connect()) as conn:
        rows = conn.execute(sql, params).fetchall()

    return [dict(row) for row in rows]


def recent_orders(
    limit: int = 100,
    market: str = "",
    symbol: str = "",
) -> List[Dict[str, Any]]:
    init_blackbox()

    where: List[str] = []
    params: List[Any] = []

    if market:
        where.append("market = ?")
        params.append(str(market).upper())

    if symbol:
        where.append("symbol = ?")
        params.append(str(symbol).upper())

    sql = "SELECT * FROM order_log"
    if where:
        sql += " WHERE " + " AND ".join(where)

    sql += " ORDER BY ts DESC LIMIT ?"
    params.append(max(1, int(limit)))

    with closing(_connect()) as conn:
        rows = conn.execute(sql, params).fetchall()

    return [dict(row) for row in rows]


def blackbox_status() -> Dict[str, Any]:
    """
    Worker /status 화면 등에 붙일 수 있는 간단한 상태 정보입니다.
    """
    init_blackbox()

    with closing(_connect()) as conn:
        decision_count = conn.execute(
            "SELECT COUNT(*) AS n FROM decision_log"
        ).fetchone()["n"]

        order_count = conn.execute(
            "SELECT COUNT(*) AS n FROM order_log"
        ).fetchone()["n"]

        last_decision = conn.execute(
            """
            SELECT ts, market, symbol, event, reason_code
            FROM decision_log
            ORDER BY ts DESC
            LIMIT 1
            """
        ).fetchone()

        last_order = conn.execute(
            """
            SELECT ts, market, symbol, side, event, reason_code, status
            FROM order_log
            ORDER BY ts DESC
            LIMIT 1
            """
        ).fetchone()

    return {
        "ok": True,
        "db_path": str(DB_PATH),
        "decision_count": int(decision_count),
        "order_count": int(order_count),
        "last_decision": dict(last_decision) if last_decision else None,
        "last_order": dict(last_order) if last_order else None,
    }


# import 되는 순간 DB 준비
try:
    init_blackbox()
except Exception as exc:
    _stdout(
        "INIT_ERROR",
        {
            "ts": _now_iso(),
            "db_path": str(DB_PATH),
            "error": f"{type(exc).__name__}: {exc}",
        },
    )

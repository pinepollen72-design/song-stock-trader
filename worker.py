from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
import traceback
from datetime import datetime, time as dtime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from trader_core import (
    Settings,
    KISClient,
    parse_domestic_holdings,
    merge_overseas_holdings,
)
from strategy_kr import build_kr_top5
from strategy_us import build_us_top5
from auto_engine import AutoConfig, run_kr_cycle, run_us_cycle

KST = ZoneInfo("Asia/Seoul")
ET = ZoneInfo("America/New_York")


def _resolve_state_dir() -> Path:
    explicit = os.getenv("SONG_TRADER_STATE_DIR", "").strip()
    if explicit:
        return Path(explicit)
    railway_volume = os.getenv("RAILWAY_VOLUME_MOUNT_PATH", "").strip()
    if railway_volume:
        return Path(railway_volume) / "song_trader_v2"
    return Path("/tmp/song_trader_v2")


STATE_DIR = _resolve_state_dir()
STATE_DIR.mkdir(parents=True, exist_ok=True)

STATUS_FILE = STATE_DIR / "worker_status.json"
JOURNAL_FILE = STATE_DIR / "trade_journal.json"
LOG_FILE = STATE_DIR / "worker.log"

STATE_IS_PERSISTENT = not str(STATE_DIR).startswith("/tmp/")
PORT = int(os.getenv("PORT", "8080"))

ENV = os.getenv("SONG_WORKER_ENV", "demo").strip().lower()
if ENV not in ("demo", "real"):
    ENV = "demo"

EXECUTE = os.getenv("WORKER_EXECUTE_ORDERS", "false").lower() in ("1", "true", "yes", "on")
ALLOW_REAL = os.getenv("ALLOW_REAL_WORKER", "false").lower() in ("1", "true", "yes", "on")
REAL_CONFIRM = os.getenv("REAL_WORKER_CONFIRM", "") == "I-UNDERSTAND-LIVE-ORDERS"
PRIMARY = os.getenv("WORKER_PRIMARY", "false").lower() in ("1", "true", "yes", "on")
EFFECTIVE_EXECUTE = bool(EXECUTE and PRIMARY)

PROVIDER = os.getenv("WORKER_PROVIDER", "Railway" if os.getenv("RAILWAY_ENVIRONMENT") else "Worker")

LOOP_SECONDS = max(30, int(os.getenv("WORKER_LOOP_SECONDS", "45")))
KR_RESCAN_SECONDS = max(60, int(os.getenv("KR_RESCAN_SECONDS", "90")))
US_RESCAN_SECONDS = max(60, int(os.getenv("US_RESCAN_SECONDS", "90")))
BALANCE_SYNC_SECONDS = max(60, int(os.getenv("BALANCE_SYNC_SECONDS", "120")))

US_UNIVERSE = [
    x.strip().upper()
    for x in os.getenv(
        "US_UNIVERSE",
        "AAPL,MSFT,NVDA,AMZN,META,TSLA,AMD,GOOGL,AVGO,NFLX,PLTR,MU,INTC,SMCI,ARM,COIN,HOOD,SOFI,MSTR,RBLX,UBER,CRWD,PANW,QCOM,AMAT,TSM,MRVL,LLY,JPM,BAC",
    ).split(",")
    if x.strip()
]

CFG = AutoConfig(
    kr_daily_budget=int(os.getenv("KR_DAILY_BUDGET", "10000000")),
    kr_per_stock_budget=int(os.getenv("KR_PER_STOCK_BUDGET", "3000000")),
    us_daily_budget_usd=float(os.getenv("US_DAILY_BUDGET", "5000")),
    us_per_stock_budget_usd=float(os.getenv("US_PER_STOCK_BUDGET", "1500")),
    max_positions=int(os.getenv("MAX_POSITIONS", "3")),
    min_score=float(os.getenv("MIN_COMBINED_SCORE", "50")),
    stop_loss_pct=float(os.getenv("STOP_LOSS_PCT", "3.0")),
    take1_pct=float(os.getenv("TAKE1_PCT", "3.0")),
    take2_pct=float(os.getenv("TAKE2_PCT", "5.0")),
    kr_last_entry_time=os.getenv("KR_LAST_ENTRY_TIME", "14:50"),
    kr_force_exit_time=os.getenv("KR_FORCE_EXIT_TIME", "15:15"),
    kr_add2_trigger_pct=float(os.getenv("KR_ADD2_TRIGGER_PCT", "0.40")),
    kr_profit_guard_trigger_pct=float(os.getenv("KR_PROFIT_GUARD_TRIGGER_PCT", "1.20")),
    kr_profit_guard_drawdown_pct=float(os.getenv("KR_PROFIT_GUARD_DRAWDOWN_PCT", "0.80")),
    kr_signal_max_age_seconds=int(os.getenv("KR_SIGNAL_MAX_AGE_SECONDS", "180")),
    kr_new_entry_top_n=int(os.getenv("KR_NEW_ENTRY_TOP_N", "5")),
    kr_require_momentum_confirm=os.getenv("KR_REQUIRE_MOMENTUM_CONFIRM", "true").lower()
    in ("1", "true", "yes", "on"),
    us_last_entry_time=os.getenv("US_LAST_ENTRY_TIME", "15:30"),
    us_force_exit_time=os.getenv("US_FORCE_EXIT_TIME", "15:50"),
    us_add2_trigger_pct=float(os.getenv("US_ADD2_TRIGGER_PCT", "0.40")),
    us_profit_guard_trigger_pct=float(os.getenv("US_PROFIT_GUARD_TRIGGER_PCT", "1.20")),
    us_profit_guard_drawdown_pct=float(os.getenv("US_PROFIT_GUARD_DRAWDOWN_PCT", "0.80")),
    us_buy_limit_buffer_pct=float(os.getenv("US_BUY_LIMIT_BUFFER_PCT", "0.15")),
    us_sell_limit_buffer_pct=float(os.getenv("US_SELL_LIMIT_BUFFER_PCT", "0.15")),
    us_signal_max_age_seconds=int(os.getenv("US_SIGNAL_MAX_AGE_SECONDS", "180")),
    us_new_entry_top_n=int(os.getenv("US_NEW_ENTRY_TOP_N", "5")),
    us_require_momentum_confirm=os.getenv("US_REQUIRE_MOMENTUM_CONFIRM", "true").lower()
    in ("1", "true", "yes", "on"),
    buying_power_buffer_pct=float(os.getenv("BUYING_POWER_BUFFER_PCT", "5")),
    confirm_wait_seconds=int(os.getenv("ORDER_CONFIRM_WAIT_SECONDS", "8")),
    pending_timeout_seconds=int(os.getenv("PENDING_TIMEOUT_SECONDS", "300")),
    buy1_pct=int(os.getenv("BUY1_PCT", "50")),
    buy2_pct=int(os.getenv("BUY2_PCT", "50")),
    force_exit_all_demo_holdings=os.getenv("FORCE_EXIT_ALL_DEMO_HOLDINGS", "true").lower()
    in ("1", "true", "yes", "on"),
)

RUNNING = True


def log(text: str) -> None:
    line = f"[{datetime.now(KST).isoformat(timespec='seconds')}] {text}"
    print(line, flush=True)
    try:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _atomic_write_json(path: Path, payload) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def load_status() -> dict:
    try:
        if STATUS_FILE.exists():
            return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _load_journal_store() -> dict:
    try:
        if JOURNAL_FILE.exists():
            data = json.loads(JOURNAL_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {
                    "kr_journal": list(data.get("kr_journal", []) or []),
                    "us_journal": list(data.get("us_journal", []) or []),
                }
    except Exception:
        pass
    return {"kr_journal": [], "us_journal": []}


def _save_journal_store(kr_journal: list, us_journal: list) -> None:
    _atomic_write_json(
        JOURNAL_FILE,
        {
            "updated_at": datetime.now(KST).isoformat(timespec="seconds"),
            "kr_journal": list(kr_journal or [])[-1000:],
            "us_journal": list(us_journal or [])[-1000:],
        },
    )


def _merge_journals(*sources: list) -> list:
    merged, seen = [], set()
    for source in sources:
        for row in list(source or []):
            if not isinstance(row, dict):
                continue
            key = row.get("_key")
            if not key:
                key = "|".join(
                    str(row.get(k, ""))
                    for k in ("시간", "시장", "종목코드", "구분", "수량", "상태", "손익률")
                )
            if key in seen:
                continue
            seen.add(key)
            merged.append(row)
    return merged[-1000:]


def save_status(**updates) -> None:
    status = load_status()
    status.update(updates)
    now = datetime.now(KST).isoformat(timespec="seconds")
    status.update({
        "updated_at": now,
        "heartbeat_at": now,
        "env": ENV,
        "execute_orders": EFFECTIVE_EXECUTE,
        "requested_execute_orders": EXECUTE,
        "primary_worker": PRIMARY,
        "provider": PROVIDER,
        "version": "2.7.1-readonly-status",
        "status_api": "/status",
        "state_dir": str(STATE_DIR),
        "persistent_state": STATE_IS_PERSISTENT,
    })
    _atomic_write_json(STATUS_FILE, status)


def _safe_result(result: dict | None) -> dict | None:
    if not isinstance(result, dict):
        return None
    safe = {
        "time": result.get("time"),
        "market": result.get("market"),
        "message": result.get("message"),
        "top5_symbols": result.get("top5_symbols", []),
        "holdings": result.get("holdings", []),
        "actions": [],
        "diagnostics": (result.get("diagnostics", []) or [])[-40:],
    }
    for a in result.get("actions", []) or []:
        if not isinstance(a, dict):
            continue
        safe["actions"].append({
            k: a.get(k)
            for k in (
                "symbol", "name", "action", "status", "qty", "price",
                "combined_score", "pnl", "reason", "before_qty", "after_qty", "msg1",
            )
            if k in a
        })
    return safe


def _journal_append(journal: list, result: dict | None, market: str) -> list:
    if not isinstance(result, dict):
        return journal[-300:]

    for a in result.get("actions", []) or []:
        order_status = str(a.get("status", ""))
        if order_status not in ("FILLED", "ORDERED", "REJECT", "ERROR"):
            continue

        action = str(a.get("action", ""))
        side = "매수" if action.startswith("BUY") else "매도"
        ts = str(result.get("time") or datetime.now(KST).isoformat(timespec="seconds"))

        # 같은 시각/종목/액션/수량/상태 중복 방지
        key = f"{market}|{ts}|{a.get('symbol')}|{action}|{a.get('qty')}|{order_status}"
        if any(isinstance(x, dict) and x.get("_key") == key for x in journal):
            continue

        journal.append({
            "_key": key,
            "시간": ts,
            "시장": "국내" if market == "KR" else "미국",
            "종목": a.get("name") or a.get("symbol"),
            "종목코드": a.get("symbol"),
            "구분": side,
            "수량": a.get("qty", 0),
            "종합점수": a.get("combined_score", ""),
            "손익률": a.get("pnl", ""),
            "이유": a.get("reason", ""),
            "상태": (
                "체결확인" if order_status == "FILLED"
                else "주문접수" if order_status == "ORDERED"
                else "주문실패"
            ),
            # 주문 직후 예상수량. 실제 체결확인은 다음 KIS 잔고 동기화 진단으로 확인.
            "한국투자확인수량": a.get("after_qty", ""),
            "오류": (
                a.get("msg1", "") or a.get("error", "")
                if order_status in ("REJECT", "ERROR")
                else ""
            ),
        })
    return journal[-300:]


def _stage() -> tuple[str, str]:
    kr = datetime.now(KST)
    us = datetime.now(ET)
    if kr.weekday() < 5 and dtime(9, 0) <= kr.time() < dtime(15, 30):
        if kr.time() >= _clock_env("KR_FORCE_EXIT_TIME", "15:15"):
            return "KR_EXIT", "🇰🇷 국내장 실제 보유잔고 강제청산 구간"
        return "KR", "🇰🇷 국내장 자동매매 실행 중"
    if us.weekday() < 5 and dtime(9, 30) <= us.time() < dtime(16, 0):
        if us.time() >= _clock_env("US_FORCE_EXIT_TIME", "15:50"):
            return "US_EXIT", "🇺🇸 미국장 실제 보유잔고 강제청산 구간"
        return "US", "🇺🇸 미국장 자동매매 실행 중"
    return "WAIT", "⏳ 정규장 대기 중 · 한국투자 잔고는 계속 동기화"


def _clock_env(name: str, default: str) -> dtime:
    raw = os.getenv(name, default)
    h, m = [int(x) for x in raw.split(":")]
    return dtime(h, m)


def _kr_balance_snapshot(client: KISClient) -> dict:
    at = datetime.now(KST).isoformat(timespec="seconds")
    try:
        raw = client.domestic_balance()
    except Exception as e:
        return {"ok": False, "at": at, "error": f"{type(e).__name__}: {e}"}

    if not isinstance(raw, dict):
        return {"ok": False, "at": at, "error": "한국투자 국내 잔고 응답이 JSON 객체가 아님"}

    rt_cd = str(raw.get("rt_cd", ""))
    msg1 = str(raw.get("msg1", "") or "")
    no_balance = "잔고내역이 없습니다" in msg1

    if rt_cd and rt_cd != "0" and not no_balance:
        return {
            "ok": False,
            "at": at,
            "error": f"{raw.get('msg_cd', '')} {msg1}".strip(),
        }

    try:
        df = parse_domestic_holdings(raw)
    except Exception as e:
        return {"ok": False, "at": at, "error": f"잔고 파싱 오류: {type(e).__name__}: {e}"}

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "종목코드": str(r.get("종목코드", "")).zfill(6),
            "종목명": str(r.get("종목명", "")),
            "보유수량": int(r.get("보유수량", 0) or 0),
            "매도가능수량": int(r.get("매도가능수량", r.get("보유수량", 0)) or 0),
            "평균매입가": float(r.get("평균매입가", 0) or 0),
            "현재가": float(r.get("현재가", 0) or 0),
        })

    return {
        "ok": True,
        "at": at,
        "count": len(rows),
        "holdings": rows,
        "msg1": msg1,
        "empty_balance": no_balance or len(rows) == 0,
    }


def _us_balance_snapshot(client: KISClient) -> dict:
    at = datetime.now(KST).isoformat(timespec="seconds")
    responses, errors, success_calls = [], [], 0
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
        if rt_cd and rt_cd != "0":
            errors.append(f"{exchange}: {raw.get('msg_cd', '')} {raw.get('msg1', '')}".strip())
            continue
        success_calls += 1
        raw = dict(raw)
        raw["_exchange"] = exchange
        responses.append(raw)

    if success_calls == 0:
        return {"ok": False, "at": at, "error": " / ".join(errors) or "미국 잔고조회 실패"}

    try:
        df = merge_overseas_holdings(responses)
    except Exception as e:
        return {"ok": False, "at": at, "error": f"미국 잔고 파싱 오류: {type(e).__name__}: {e}"}

    rows = []
    for _, r in df.iterrows():
        qty = float(r.get("보유수량", 0) or 0)
        sellable = float(r.get("매도가능수량", 0) or 0)
        rows.append({
            "종목코드": str(r.get("종목코드", "")).upper(),
            "종목명": str(r.get("종목명", "") or r.get("종목코드", "")),
            "거래소": str(r.get("거래소", "")).upper(),
            "보유수량": int(qty) if qty.is_integer() else qty,
            "매도가능수량": int(sellable) if sellable.is_integer() else sellable,
            "평균매입가": float(r.get("평균매입가", 0) or 0),
            "현재가": float(r.get("현재가", 0) or 0),
            "평가금액": float(r.get("평가금액", 0) or 0),
            "평가손익": float(r.get("평가손익", 0) or 0),
            "수익률": float(r.get("수익률", 0) or 0),
        })

    return {
        "ok": True,
        "at": at,
        "count": len(rows),
        "holdings": rows,
        "warning": " / ".join(errors),
        "source": "KIS 해외주식 잔고 직접조회",
    }


def _apply_balance_sync(status_updates: dict, prefix: str, snap: dict) -> None:
    status_updates[f"{prefix}_holdings_sync"] = {k: v for k, v in snap.items() if k != "holdings"}
    if snap.get("ok"):
        status_updates[f"{prefix}_holdings"] = snap.get("holdings", [])



def _public_status_payload(status: dict) -> dict:
    """공개 읽기전용 페이지에 필요한 안전한 상태만 반환합니다."""
    allowed = (
        "running",
        "status",
        "stage",
        "stage_message",
        "heartbeat_at",
        "updated_at",
        "env",
        "execute_orders",
        "provider",
        "version",
        "last_error",
        "kr_scan_error",
        "us_scan_error",
        "kr_top5",
        "us_top5",
        "kr_holdings",
        "us_holdings",
        "kr_holdings_sync",
        "us_holdings_sync",
        "kr_last_result",
        "us_last_result",
        "kr_journal",
        "us_journal",
        "config",
    )
    return {key: status.get(key) for key in allowed if key in status}

def _result_has_order_activity(result: dict | None) -> bool:
    if not isinstance(result, dict):
        return False
    return any(str(a.get("status", "")) in ("FILLED", "ORDERED") for a in result.get("actions", []) or [])


class Handler(BaseHTTPRequestHandler):
    def _send_bytes(self, status_code: int, body: bytes, content_type: str) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/") or "/"

        if path == "/ping":
            self._send_bytes(
                200,
                b"song-trader-worker-ok",
                "text/plain; charset=utf-8",
            )
            return

        if path not in ("/", "/status", "/public-status", "/health"):
            self._send_bytes(
                404,
                b'{"ok":false,"error":"not found"}',
                "application/json; charset=utf-8",
            )
            return

        status = load_status()

        if path == "/health":
            payload = {
                "ok": True,
                "running": bool(status.get("running")),
                "heartbeat_at": status.get("heartbeat_at"),
                "version": status.get("version"),
                "listen_port_env": PORT,
                "public_status": "/public-status",
            }
        elif path == "/public-status":
            payload = _public_status_payload(status)
        else:
            payload = status

        body = json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")

        self._send_bytes(
            200,
            body,
            "application/json; charset=utf-8",
        )

    def log_message(self, fmt, *args):
        return


_HTTP_SERVERS = []


def _start_http_server(port: int):
    try:
        srv = ThreadingHTTPServer(("0.0.0.0", int(port)), Handler)
        threading.Thread(
            target=srv.serve_forever,
            daemon=True,
            name=f"status-http-{port}",
        ).start()
        _HTTP_SERVERS.append(srv)
        log(f"상태 HTTP 서버 시작: 0.0.0.0:{port}")
        return srv
    except OSError as e:
        log(f"상태 HTTP 서버 포트 {port} 시작 실패: {type(e).__name__}: {e}")
        return None


def start_server():
    # Railway의 환경 PORT와 현재 Public Networking Target Port 8080을 모두 시도합니다.
    ports = []
    for candidate in (PORT, 8080):
        candidate = int(candidate)
        if candidate not in ports:
            ports.append(candidate)

    started = []
    for port in ports:
        srv = _start_http_server(port)
        if srv is not None:
            started.append(srv)

    if not started:
        raise RuntimeError(f"상태 HTTP 서버 시작 실패: 시도 포트 {ports}")
    return started


def stop_handler(signum, frame):
    global RUNNING
    RUNNING = False


def main() -> int:
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    start_server()

    if ENV == "real" and not (ALLOW_REAL and REAL_CONFIRM):
        save_status(
            running=False,
            status="locked",
            stage="LOCKED",
            stage_message="🔒 실전 Worker 잠금 상태",
        )
        log("실전 Worker 잠금: ALLOW_REAL_WORKER + REAL_WORKER_CONFIRM 필요")
        return 2

    settings = Settings.from_env()
    client = KISClient(settings=settings, env=ENV)
    client.get_token()

    old = load_status()
    journal_store = _load_journal_store()
    kr_journal = _merge_journals(old.get("kr_journal", []) or [], journal_store.get("kr_journal", []) or [])
    us_journal = _merge_journals(old.get("us_journal", []) or [], journal_store.get("us_journal", []) or [])
    kr_top5 = pd.DataFrame(old.get("kr_top5", []) or [])
    us_top5 = pd.DataFrame(old.get("us_top5", []) or [])

    kr_last_scan = 0.0
    us_last_scan = 0.0
    last_balance_sync = 0.0

    log(
        f"V2.7 Worker 시작 env={ENV}, "
        f"주문요청={'ON' if EXECUTE else 'OFF'}, 실제주문={'ON' if EFFECTIVE_EXECUTE else 'DRY'}, "
        f"PRIMARY={'YES' if PRIMARY else 'NO'}, 계좌끝4자리={client.account_no[-4:]}"
    )

    initial_updates = {
        "running": True,
        "status": "running",
        "kr_journal": kr_journal,
        "us_journal": us_journal,
        "account": {"last4": client.account_no[-4:], "product_code": client.product_code},
    }
    _apply_balance_sync(initial_updates, "kr", _kr_balance_snapshot(client))
    _apply_balance_sync(initial_updates, "us", _us_balance_snapshot(client))
    _save_journal_store(kr_journal, us_journal)
    save_status(**initial_updates)

    if not STATE_IS_PERSISTENT:
        log("⚠️ 상태 저장소가 /tmp 입니다. Railway Volume 또는 SONG_TRADER_STATE_DIR 영구 경로를 권장합니다.")
    else:
        log(f"✅ 영구 상태 저장소 사용: {STATE_DIR}")

    last_balance_sync = time.time()

    while RUNNING:
        started = time.time()
        kr_result = None
        us_result = None
        try:
            kr_now = datetime.now(KST)
            us_now = datetime.now(ET)
            stage, msg = _stage()

            updates = {
                "running": True,
                "status": "running",
                "stage": stage,
                "stage_message": msg,
                "last_error": "",
            }

            if time.time() - last_balance_sync >= BALANCE_SYNC_SECONDS:
                _apply_balance_sync(updates, "kr", _kr_balance_snapshot(client))
                _apply_balance_sync(updates, "us", _us_balance_snapshot(client))
                last_balance_sync = time.time()

            if kr_now.weekday() < 5 and dtime(8, 30) <= kr_now.time() < dtime(15, 30):
                if (
                    kr_now.time() < _clock_env("KR_FORCE_EXIT_TIME", "15:15")
                    and ((time.time() - kr_last_scan) >= KR_RESCAN_SECONDS or kr_top5.empty)
                ):
                    try:
                        kr_top5 = build_kr_top5(client)
                        kr_last_scan = time.time()
                        updates["kr_scan_error"] = ""
                    except Exception as e:
                        # 오래된 후보로 뒤늦게 진입하지 않도록 즉시 비운다.
                        kr_top5 = pd.DataFrame()
                        kr_last_scan = time.time()
                        updates["kr_scan_error"] = f"{type(e).__name__}: {e}"
                        log(f"KR TOP5 오류(신규매수 차단): {type(e).__name__}: {e}")

                kr_result = run_kr_cycle(client, kr_top5, CFG, EFFECTIVE_EXECUTE, source="WORKER")
                kr_journal = _journal_append(kr_journal, kr_result, "KR")
                updates["kr_last_result"] = _safe_result(kr_result)

                if isinstance(kr_result, dict):
                    updates["kr_holdings"] = kr_result.get("holdings", []) or []
                    updates["kr_holdings_sync"] = {
                        "ok": not bool(kr_result.get("balance_warning")),
                        "at": kr_result.get("time"),
                        "count": len(kr_result.get("holdings", []) or []),
                        "source": "자동매매 사이클 KIS 잔고조회",
                        "warning": kr_result.get("balance_warning", ""),
                    }

                if _result_has_order_activity(kr_result):
                    # 주문 직후 한 번 더 조회. 실제 체결확정은 다음 사이클 pending 로직이 담당.
                    time.sleep(max(0, min(3, int(CFG.confirm_wait_seconds))))
                    _apply_balance_sync(updates, "kr", _kr_balance_snapshot(client))

            if us_now.weekday() < 5 and dtime(9, 30) <= us_now.time() < dtime(16, 0):
                if (
                    us_now.time() < _clock_env("US_FORCE_EXIT_TIME", "15:50")
                    and ((time.time() - us_last_scan) >= US_RESCAN_SECONDS or us_top5.empty)
                ):
                    try:
                        us_top5 = build_us_top5(US_UNIVERSE)
                        us_last_scan = time.time()
                        updates["us_scan_error"] = ""
                    except Exception as e:
                        # 오래된 후보로 잘못 진입하지 않도록 신규후보를 즉시 비운다.
                        us_top5 = pd.DataFrame()
                        us_last_scan = time.time()
                        updates["us_scan_error"] = f"{type(e).__name__}: {e}"
                        log(f"US TOP5 오류(신규매수 차단): {type(e).__name__}: {e}")

                us_result = run_us_cycle(client, us_top5, CFG, EFFECTIVE_EXECUTE, source="WORKER")
                us_journal = _journal_append(us_journal, us_result, "US")
                updates["us_last_result"] = _safe_result(us_result)

                if _result_has_order_activity(us_result):
                    _apply_balance_sync(updates, "us", _us_balance_snapshot(client))

            updates.update({
                "kr_top5": kr_top5.to_dict("records") if not kr_top5.empty else [],
                "us_top5": us_top5.to_dict("records") if not us_top5.empty else [],
                "kr_journal": kr_journal,
                "us_journal": us_journal,
                "config": {
                    "min_score": CFG.min_score,
                    "stop_loss_pct": CFG.stop_loss_pct,
                    "take1_pct": CFG.take1_pct,
                    "take2_pct": CFG.take2_pct,
                    "buy_split": f"{CFG.buy1_pct}:{CFG.buy2_pct}",
                    "kr_force_exit_time": CFG.kr_force_exit_time,
                    "us_force_exit_time": CFG.us_force_exit_time,
                    "kr_rescan_seconds": KR_RESCAN_SECONDS,
                    "kr_profit_guard_trigger_pct": CFG.kr_profit_guard_trigger_pct,
                    "kr_profit_guard_drawdown_pct": CFG.kr_profit_guard_drawdown_pct,
                    "kr_signal_max_age_seconds": CFG.kr_signal_max_age_seconds,
                    "us_rescan_seconds": US_RESCAN_SECONDS,
                    "us_profit_guard_trigger_pct": CFG.us_profit_guard_trigger_pct,
                    "us_profit_guard_drawdown_pct": CFG.us_profit_guard_drawdown_pct,
                    "us_buy_limit_buffer_pct": CFG.us_buy_limit_buffer_pct,
                    "us_sell_limit_buffer_pct": CFG.us_sell_limit_buffer_pct,
                    "pending_timeout_seconds": CFG.pending_timeout_seconds,
                },
            })

            _save_journal_store(kr_journal, us_journal)
            save_status(**updates)

        except Exception as e:
            log(f"메인 루프 오류: {type(e).__name__}: {e}")
            log(traceback.format_exc())
            save_status(running=True, status="error", last_error=f"{type(e).__name__}: {e}")

        elapsed = time.time() - started
        time.sleep(max(1.0, LOOP_SECONDS - elapsed))

    save_status(running=False, status="stopped", stage="STOPPED", stage_message="⏹️ Worker 종료")
    return 0


if __name__ == "__main__":
    sys.exit(main())

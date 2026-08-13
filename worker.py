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

from trader_core import Settings, KISClient
from strategy_kr import build_kr_top5
from strategy_us import build_us_top5
from auto_engine import AutoConfig, run_kr_cycle, run_us_cycle

KST = ZoneInfo("Asia/Seoul")
ET = ZoneInfo("America/New_York")
STATE_DIR = Path(os.getenv("SONG_TRADER_STATE_DIR", "/tmp/song_trader_v2"))
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATUS_FILE = STATE_DIR / "worker_status.json"
LOG_FILE = STATE_DIR / "worker.log"
PORT = int(os.getenv("PORT", "8080"))

ENV = os.getenv("SONG_WORKER_ENV", "demo").strip().lower()
if ENV not in ("demo", "real"):
    ENV = "demo"
EXECUTE = os.getenv("WORKER_EXECUTE_ORDERS", "false").lower() in ("1", "true", "yes", "on")
ALLOW_REAL = os.getenv("ALLOW_REAL_WORKER", "false").lower() in ("1", "true", "yes", "on")
REAL_CONFIRM = os.getenv("REAL_WORKER_CONFIRM", "") == "I-UNDERSTAND-LIVE-ORDERS"
LOOP_SECONDS = max(30, int(os.getenv("WORKER_LOOP_SECONDS", "60")))
KR_RESCAN_SECONDS = max(120, int(os.getenv("KR_RESCAN_SECONDS", "300")))
US_RESCAN_SECONDS = max(120, int(os.getenv("US_RESCAN_SECONDS", "300")))

US_UNIVERSE = [x.strip().upper() for x in os.getenv(
    "US_UNIVERSE", "AAPL,MSFT,NVDA,AMZN,META,TSLA,AMD,GOOGL,AVGO,NFLX"
).split(",") if x.strip()]

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
    us_last_entry_time=os.getenv("US_LAST_ENTRY_TIME", "15:30"),
    us_force_exit_time=os.getenv("US_FORCE_EXIT_TIME", "15:50"),
    buying_power_buffer_pct=float(os.getenv("BUYING_POWER_BUFFER_PCT", "5")),
    confirm_wait_seconds=int(os.getenv("ORDER_CONFIRM_WAIT_SECONDS", "8")),
    force_exit_all_demo_holdings=os.getenv("FORCE_EXIT_ALL_DEMO_HOLDINGS", "true").lower() in ("1", "true", "yes", "on"),
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


def load_status() -> dict:
    try:
        return json.loads(STATUS_FILE.read_text(encoding="utf-8")) if STATUS_FILE.exists() else {}
    except Exception:
        return {}


def save_status(**updates) -> None:
    status = load_status()
    status.update(updates)
    status.update({
        "updated_at": datetime.now(KST).isoformat(timespec="seconds"),
        "heartbeat_at": datetime.now(KST).isoformat(timespec="seconds"),
        "env": ENV,
        "execute_orders": EXECUTE,
    })
    STATUS_FILE.write_text(json.dumps(status, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


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
        "diagnostics": result.get("diagnostics", [])[-20:],
    }
    for a in result.get("actions", []) or []:
        if not isinstance(a, dict):
            continue
        safe["actions"].append({k: a.get(k) for k in (
            "symbol", "name", "action", "status", "qty", "price", "combined_score",
            "pnl", "reason", "before_qty", "after_qty", "msg1"
        ) if k in a})
    return safe


def _journal_append(journal: list, result: dict | None, market: str) -> list:
    if not isinstance(result, dict):
        return journal[-300:]
    for a in result.get("actions", []) or []:
        status = str(a.get("status", ""))
        if status not in ("FILLED", "ORDERED", "REJECT", "ERROR"):
            continue
        action = str(a.get("action", ""))
        side = "매수" if action.startswith("BUY") else "매도"
        ts = str(result.get("time") or datetime.now(KST).isoformat(timespec="seconds"))
        key = f"{market}|{ts}|{a.get('symbol')}|{action}|{a.get('qty')}|{status}"
        if any(x.get("_key") == key for x in journal if isinstance(x, dict)):
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
            "상태": "체결확인" if status == "FILLED" else "주문접수" if status == "ORDERED" else "주문실패",
            "한국투자확인수량": a.get("after_qty", ""),
            "오류": a.get("msg1", "") if status in ("REJECT", "ERROR") else "",
        })
    return journal[-300:]


def _stage() -> tuple[str, str]:
    kr = datetime.now(KST)
    us = datetime.now(ET)
    if kr.weekday() < 5 and dtime(9, 0) <= kr.time() < dtime(15, 30):
        if kr.time() >= dtime(15, 15):
            return "KR_EXIT", "🇰🇷 국내장 보유잔고 강제청산 구간"
        return "KR", "🇰🇷 국내장 자동매매 실행 중"
    if us.weekday() < 5 and dtime(9, 30) <= us.time() < dtime(16, 0):
        if us.time() >= dtime(15, 50):
            return "US_EXIT", "🇺🇸 미국장 보유잔고 강제청산 구간"
        return "US", "🇺🇸 미국장 자동매매 실행 중"
    return "WAIT", "⏳ 다음 정규장 대기 중"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/", "/status", "/health"):
            self.send_response(404); self.end_headers(); return
        status = load_status()
        if self.path == "/health":
            payload = {"ok": True, "running": bool(status.get("running")), "heartbeat_at": status.get("heartbeat_at")}
        else:
            # 민감정보는 상태 파일에 저장하지 않으므로 그대로 공개 가능
            payload = status
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        return


def start_server():
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def stop_handler(signum, frame):
    global RUNNING
    RUNNING = False


def main() -> int:
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    start_server()

    if ENV == "real" and not (ALLOW_REAL and REAL_CONFIRM):
        save_status(running=False, status="locked", stage="LOCKED", stage_message="🔒 실전 Worker 잠금 상태")
        log("실전 Worker 잠금: ALLOW_REAL_WORKER + REAL_WORKER_CONFIRM 필요")
        return 2

    settings = Settings.from_env()
    client = KISClient(settings=settings, env=ENV)
    client.get_token()

    old = load_status()
    kr_journal = list(old.get("kr_journal", []) or [])
    us_journal = list(old.get("us_journal", []) or [])
    kr_top5 = pd.DataFrame(old.get("kr_top5", []) or [])
    us_top5 = pd.DataFrame(old.get("us_top5", []) or [])
    kr_last_scan = 0.0
    us_last_scan = 0.0

    log(f"V2 Worker 시작 env={ENV}, 주문={'ON' if EXECUTE else 'DRY'}")
    save_status(running=True, status="running", version="2.0-clean", kr_journal=kr_journal, us_journal=us_journal)

    while RUNNING:
        started = time.time()
        kr_result = None
        us_result = None
        try:
            kr_now = datetime.now(KST)
            us_now = datetime.now(ET)
            stage, msg = _stage()
            save_status(running=True, status="running", stage=stage, stage_message=msg)

            # 국내: 08:30~15:30. 15:15 이후에는 TOP5가 없어도 잔고청산 사이클이 반드시 돈다.
            if kr_now.weekday() < 5 and dtime(8, 30) <= kr_now.time() < dtime(15, 30):
                if kr_now.time() < dtime(15, 15) and ((time.time() - kr_last_scan) >= KR_RESCAN_SECONDS or kr_top5.empty):
                    try:
                        kr_top5 = build_kr_top5(client)
                        kr_last_scan = time.time()
                    except Exception as e:
                        log(f"KR TOP5 오류: {type(e).__name__}: {e}")
                kr_result = run_kr_cycle(client, kr_top5, CFG, EXECUTE)
                kr_journal = _journal_append(kr_journal, kr_result, "KR")

            # 미국: 정규장 동안만. 15:50 이후에는 TOP5 없이도 잔고청산 사이클이 돈다.
            if us_now.weekday() < 5 and dtime(9, 30) <= us_now.time() < dtime(16, 0):
                if us_now.time() < dtime(15, 50) and ((time.time() - us_last_scan) >= US_RESCAN_SECONDS or us_top5.empty):
                    try:
                        us_top5 = build_us_top5(US_UNIVERSE)
                        us_last_scan = time.time()
                    except Exception as e:
                        log(f"US TOP5 오류: {type(e).__name__}: {e}")
                us_result = run_us_cycle(client, us_top5, CFG, EXECUTE)
                us_journal = _journal_append(us_journal, us_result, "US")

            stage, msg = _stage()
            updates = {
                "running": True, "status": "running", "stage": stage, "stage_message": msg,
                "kr_top5": kr_top5.to_dict("records") if not kr_top5.empty else [],
                "us_top5": us_top5.to_dict("records") if not us_top5.empty else [],
                "kr_journal": kr_journal, "us_journal": us_journal,
                "config": {
                    "min_score": CFG.min_score,
                    "stop_loss_pct": CFG.stop_loss_pct,
                    "take1_pct": CFG.take1_pct,
                    "take2_pct": CFG.take2_pct,
                    "kr_force_exit_time": CFG.kr_force_exit_time,
                    "us_force_exit_time": CFG.us_force_exit_time,
                },
            }
            if kr_result is not None:
                updates["kr_last_result"] = _safe_result(kr_result)
                updates["kr_holdings"] = kr_result.get("holdings", [])
            if us_result is not None:
                updates["us_last_result"] = _safe_result(us_result)
                updates["us_holdings"] = us_result.get("holdings", [])
            save_status(**updates)

        except Exception as e:
            log(f"메인 루프 오류: {type(e).__name__}: {e}")
            log(traceback.format_exc())
            save_status(running=True, status="error", last_error=f"{type(e).__name__}: {e}")

        time.sleep(max(1.0, LOOP_SECONDS - (time.time() - started)))

    save_status(running=False, status="stopped", stage="STOPPED", stage_message="⏹️ Worker 종료")
    return 0


if __name__ == "__main__":
    sys.exit(main())

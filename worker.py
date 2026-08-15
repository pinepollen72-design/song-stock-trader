from __future__ import annotations

import json
import os
import re
import secrets
import signal
import sys
import threading
import time
import traceback
from datetime import datetime, time as dtime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from zoneinfo import ZoneInfo
from urllib.parse import urlparse, parse_qs

import pandas as pd

from trader_core import (
    Settings,
    KISClient,
    parse_domestic_holdings,
    merge_overseas_holdings,
)
from strategy_kr import build_kr_top5
from strategy_us import build_us_ranked
from auto_engine import AutoConfig, run_kr_cycle, run_us_cycle
from trade_blackbox import (
    log_decision as blackbox_log_decision,
    log_order as blackbox_log_order,
    blackbox_status,
)
from replay_us import run_replay
from replay_kr import run_kr_trade_replay, append_kr_top5_snapshot, compare_kr_buy2_strategies
from trade_diagnose_us import diagnose_trade_day
from trade_exit_diagnose_us import diagnose_exit_state_day

# AI 투자위원회 V1 — 그림자 모드.
# 어떤 오류가 나도 주문 Worker는 계속 동작하도록 fail-open 한다.
AI_COMMITTEE_IMPORT_ERROR = ""
try:
    from ai_committee import (
        submit_shadow_scan as ai_submit_shadow_scan,
        record_worker_result as ai_record_worker_result,
        committee_status as ai_committee_status,
        committee_shutdown as ai_committee_shutdown,
    )
except Exception as _ai_import_e:
    AI_COMMITTEE_IMPORT_ERROR = (
        f"{type(_ai_import_e).__name__}: {_ai_import_e}"
    )

    def ai_submit_shadow_scan(*args, **kwargs):
        return {
            "accepted": False,
            "reason": "AI committee import error",
            "error": AI_COMMITTEE_IMPORT_ERROR,
        }

    def ai_record_worker_result(*args, **kwargs):
        return None

    def ai_committee_status():
        return {
            "ok": False,
            "enabled": False,
            "configured": False,
            "shadow_mode": True,
            "import_error": AI_COMMITTEE_IMPORT_ERROR,
        }

    def ai_committee_shutdown():
        return None

AI_COMMITTEE_REPLAY_IMPORT_ERROR = ""
try:
    from ai_committee_replay import (
        run_kr_ai_committee_replay,
        summarize_kr_ai_committee_replays,
        run_kr_ai_committee_replay_v2,
        summarize_kr_ai_committee_replays_v2,
    )
except Exception as _ai_replay_import_e:
    AI_COMMITTEE_REPLAY_IMPORT_ERROR = (
        f"{type(_ai_replay_import_e).__name__}: {_ai_replay_import_e}"
    )

    def run_kr_ai_committee_replay(*args, **kwargs):
        raise RuntimeError(
            "AI committee replay import error: "
            + AI_COMMITTEE_REPLAY_IMPORT_ERROR
        )

    def summarize_kr_ai_committee_replays(*args, **kwargs):
        return {
            "ok": False,
            "error": "AI committee replay import error",
            "detail": AI_COMMITTEE_REPLAY_IMPORT_ERROR,
        }

    def run_kr_ai_committee_replay_v2(*args, **kwargs):
        raise RuntimeError(
            "AI committee V2 replay import error: "
            + AI_COMMITTEE_REPLAY_IMPORT_ERROR
        )

    def summarize_kr_ai_committee_replays_v2(*args, **kwargs):
        return {
            "ok": False,
            "error": "AI committee V2 replay import error",
            "detail": AI_COMMITTEE_REPLAY_IMPORT_ERROR,
        }

from trade_replay_us import (
    run_trade_replay,
    compare_buy2_strategies,
    compare_buy1_market_strategies,
    compare_buy1_confirmation_strategies,
    compare_entry_sizing_strategies,
    compare_exit_strategies,
    compare_reentry_control_strategies,
)

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
STATUS_TOKEN = os.getenv("SONG_STATUS_TOKEN", "").strip()
CHATGPT_VIEW_TOKEN = os.getenv("SONG_CHATGPT_VIEW_TOKEN", "").strip()

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
    kr_buy2_enabled=os.getenv("KR_BUY2_ENABLED", "false").lower()
    in ("1", "true", "yes", "on"),
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

    # 미국 C 전략 (모의운용 검증 기본값)
    us_buy2_strict_trigger_pct=float(os.getenv("US_BUY2_STRICT_TRIGGER_PCT", "0.80")),
    us_buy2_min_hold_minutes=float(os.getenv("US_BUY2_MIN_HOLD_MINUTES", "5")),
    us_buy2_max_rank=int(os.getenv("US_BUY2_MAX_RANK", "3")),
    us_buy2_min_score=float(os.getenv("US_BUY2_MIN_SCORE", "70")),
    us_buy2_require_recent5_positive=os.getenv("US_BUY2_REQUIRE_5M_POSITIVE", "true").lower()
    in ("1", "true", "yes", "on"),
    us_buy2_require_recent10_positive=os.getenv("US_BUY2_REQUIRE_10M_POSITIVE", "true").lower()
    in ("1", "true", "yes", "on"),
    us_buy2_require_relative_strength_positive=os.getenv("US_BUY2_REQUIRE_RS_POSITIVE", "true").lower()
    in ("1", "true", "yes", "on"),

    us_early_exit_enabled=os.getenv("US_EARLY_EXIT_ENABLED", "true").lower()
    in ("1", "true", "yes", "on"),
    us_early_exit_min_hold_minutes=float(os.getenv("US_EARLY_EXIT_MIN_HOLD_MINUTES", "60")),
    us_early_exit_loss_threshold_pct=float(os.getenv("US_EARLY_EXIT_LOSS_THRESHOLD_PCT", "-0.50")),
    us_early_exit_weak_points=int(os.getenv("US_EARLY_EXIT_WEAK_POINTS", "5")),
    us_stagnant_exit_enabled=os.getenv("US_STAGNANT_EXIT_ENABLED", "true").lower()
    in ("1", "true", "yes", "on"),
    us_stagnant_exit_pnl_max_pct=float(os.getenv("US_STAGNANT_EXIT_PNL_MAX_PCT", "0.20")),
    us_stagnant_exit_score_max=float(os.getenv("US_STAGNANT_EXIT_SCORE_MAX", "50")),
    us_stagnant_exit_momentum_abs_max=float(os.getenv("US_STAGNANT_EXIT_MOMENTUM_ABS_MAX", "0.05")),
    us_ban_same_symbol_after_early_exit=os.getenv("US_BAN_SAME_AFTER_EARLY_EXIT", "true").lower()
    in ("1", "true", "yes", "on"),
    us_pause_after_early_exits_count=int(os.getenv("US_PAUSE_AFTER_EARLY_EXITS_COUNT", "2")),
    us_pause_new_entries_minutes=float(os.getenv("US_PAUSE_NEW_ENTRIES_MINUTES", "60")),

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


def _blackbox_status_safe() -> dict:
    try:
        return blackbox_status()
    except Exception as e:
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
        }


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
        "version": "2.11.2a-kis-tokenfix-ai-committee-v2-replay",
        "status_api": "/status",
        "state_dir": str(STATE_DIR),
        "persistent_state": STATE_IS_PERSISTENT,
        "blackbox": _blackbox_status_safe(),
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
                "combined_score", "pnl", "reason", "reason_code",
                "peak_pnl", "drawdown_from_peak",
                "before_qty", "after_qty", "msg1",
            )
            if k in a
        })
    return safe


def _normalized_reason_code(action: str, reason: str, status: str) -> str:
    """
    사람이 보는 한국어 reason을 블랙박스 분석용 고정 코드로 바꿉니다.
    기존 자동매매 판단 자체는 전혀 바꾸지 않습니다.
    """
    action_u = str(action or "").upper().strip()
    status_u = str(status or "").upper().strip()
    text = f"{action_u} {reason or ''}".lower()

    if action_u.startswith("BUY"):
        if "2차" in text or "add2" in text or "stage2" in text or action_u in ("BUY2", "BUY_2"):
            return "BUY_STAGE2"
        if "돌파" in text or "breakout" in text:
            return "BUY_LEADER_BREAKOUT"
        if "거래량" in text or "volume" in text:
            return "BUY_VOLUME_ACCEL"
        return "BUY_STAGE1"

    if action_u.startswith("SELL"):
        if "손절" in text or "stop" in text:
            return "SELL_STOP_LOSS"
        if "1차" in text and ("익절" in text or "take" in text):
            return "SELL_TAKE_PROFIT_1"
        if "2차" in text and ("익절" in text or "take" in text):
            return "SELL_TAKE_PROFIT_2"
        if "강제" in text or "force" in text or "마감" in text:
            return "SELL_FORCE_EXIT"
        if "수익보호" in text or "profit guard" in text or "profit_guard" in text:
            return "SELL_PROFIT_GUARD"
        if "추세" in text or "momentum" in text or "signal" in text or "vwap" in text:
            return "SELL_TREND_BREAK"
        return "SELL_OTHER"

    if status_u in ("REJECT", "ERROR"):
        return "ORDER_FAILED"

    raw = f"{action_u}_{status_u}".strip("_") or "UNKNOWN"
    return re.sub(r"[^A-Z0-9_]+", "_", raw)[:80]


def _extract_order_no(action_row: dict) -> str:
    for key in (
        "kis_order_no",
        "order_no",
        "odno",
        "ODNO",
        "주문번호",
    ):
        value = action_row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _blackbox_record_action(
    *,
    market: str,
    result_time: str,
    action_row: dict,
) -> None:
    """
    기존 run_kr_cycle / run_us_cycle 결과를 블랙박스에 복사합니다.
    이 함수는 주문 판단을 바꾸지 않고 기록만 합니다.
    """
    try:
        action = str(action_row.get("action", "") or "")
        status = str(action_row.get("status", "") or "")
        symbol = str(action_row.get("symbol", "") or "").upper()
        if not symbol:
            return

        reason = str(action_row.get("reason", "") or "")
        direct_reason_code = str(action_row.get("reason_code", "") or "").strip().upper()
        reason_code = direct_reason_code or _normalized_reason_code(action, reason, status)

        qty = action_row.get("qty")
        price = action_row.get("price")
        before_qty = action_row.get("before_qty")
        after_qty = action_row.get("after_qty")
        score = action_row.get("combined_score")
        pnl = action_row.get("pnl")
        msg1 = str(action_row.get("msg1", "") or action_row.get("error", "") or "")

        side = ""
        if action.upper().startswith("BUY"):
            side = "BUY"
        elif action.upper().startswith("SELL"):
            side = "SELL"

        # 매수/매도 판단 자체를 기록
        if side:
            blackbox_log_decision(
                market=market,
                symbol=symbol,
                event=side,
                reason_code=reason_code,
                reason_text=reason,
                price=price,
                total_score=score,
                position_qty=before_qty,
                order_qty=qty,
                stage=action,
                worker_run_id=result_time,
                extra={
                    "status": status,
                    "after_qty": after_qty,
                    "pnl": pnl,
                    "peak_pnl": action_row.get("peak_pnl"),
                    "drawdown_from_peak": action_row.get("drawdown_from_peak"),
                    "recent3": action_row.get("recent3"),
                    "recent5": action_row.get("recent5"),
                    "recent10": action_row.get("recent10"),
                    "volume_ratio": action_row.get("volume_ratio"),
                    "high_distance": action_row.get("high_distance"),
                    "day_change_pct": action_row.get("day_change_pct"),
                    "lead_score": action_row.get("lead_score"),
                    "name": action_row.get("name"),
                },
            )

        # 실제 주문 관련 상태만 order_log에 기록
        if side and status in ("FILLED", "ORDERED", "REJECT", "ERROR"):
            if status == "FILLED":
                order_event = "FILLED"
            elif status == "ORDERED":
                order_event = "ORDER_ACCEPTED"
            elif status == "REJECT":
                order_event = "ORDER_REJECTED"
            else:
                order_event = "ORDER_FAILED"

            blackbox_log_order(
                market=market,
                symbol=symbol,
                side=side,
                event=order_event,
                reason_code=reason_code,
                request_qty=qty,
                request_price=price,
                kis_order_no=_extract_order_no(action_row),
                status=status,
                filled_qty=qty if status == "FILLED" else None,
                filled_price=price if status == "FILLED" else None,
                error_message=msg1 if status in ("REJECT", "ERROR") else "",
                worker_run_id=result_time,
                response={
                    "action": action,
                    "reason_code": reason_code,
                    "before_qty": before_qty,
                    "after_qty": after_qty,
                    "combined_score": score,
                    "pnl": pnl,
                    "peak_pnl": action_row.get("peak_pnl"),
                    "drawdown_from_peak": action_row.get("drawdown_from_peak"),
                    "msg1": msg1,
                },
            )
    except Exception as e:
        # 블랙박스 기록 문제로 실제 Worker가 멈추면 안 됩니다.
        log(
            f"블랙박스 기록 오류 "
            f"{market}/{action_row.get('symbol', '')}: "
            f"{type(e).__name__}: {e}"
        )


def _journal_append(journal: list, result: dict | None, market: str) -> list:
    if not isinstance(result, dict):
        return journal[-300:]

    result_time = str(result.get("time") or datetime.now(KST).isoformat(timespec="seconds"))

    for a in result.get("actions", []) or []:
        if not isinstance(a, dict):
            continue

        order_status = str(a.get("status", ""))
        if order_status not in ("FILLED", "ORDERED", "REJECT", "ERROR"):
            continue

        action = str(a.get("action", ""))
        side = "매수" if action.startswith("BUY") else "매도"
        ts = result_time

        # 같은 시각/종목/액션/수량/상태 중복 방지
        key = f"{market}|{ts}|{a.get('symbol')}|{action}|{a.get('qty')}|{order_status}"
        if any(isinstance(x, dict) and x.get("_key") == key for x in journal):
            continue

        # 기존 자동매매 일지에 새 항목이 들어갈 때 딱 한 번 블랙박스에도 기록
        _blackbox_record_action(
            market=market,
            result_time=result_time,
            action_row=a,
        )

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
            "이유코드": (
                str(a.get("reason_code", "") or "").strip().upper()
                or _normalized_reason_code(
                    action,
                    str(a.get("reason", "") or ""),
                    order_status,
                )
            ),
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
        "blackbox",
    )
    return {key: status.get(key) for key in allowed if key in status}


def _assistant_status_payload(status: dict) -> dict:
    """
    ChatGPT 확인용 최소 읽기전용 상태.
    API 키, 계좌번호, 주문 원본 응답, 저장경로는 포함하지 않습니다.
    """
    blackbox = dict(status.get("blackbox", {}) or {})
    # 외부 읽기 화면에는 내부 DB 절대경로는 제외
    blackbox.pop("db_path", None)

    return {
        "running": status.get("running"),
        "status": status.get("status"),
        "stage": status.get("stage"),
        "stage_message": status.get("stage_message"),
        "heartbeat_at": status.get("heartbeat_at"),
        "updated_at": status.get("updated_at"),
        "env": status.get("env"),
        "execute_orders": status.get("execute_orders"),
        "provider": status.get("provider"),
        "version": status.get("version"),
        "last_error": status.get("last_error"),
        "kr_scan_error": status.get("kr_scan_error"),
        "us_scan_error": status.get("us_scan_error"),
        "kr_top5": status.get("kr_top5", []),
        "us_top5": status.get("us_top5", []),
        "kr_holdings": status.get("kr_holdings", []),
        "us_holdings": status.get("us_holdings", []),
        "kr_holdings_sync": status.get("kr_holdings_sync", {}),
        "us_holdings_sync": status.get("us_holdings_sync", {}),
        "kr_last_result": status.get("kr_last_result"),
        "us_last_result": status.get("us_last_result"),
        "kr_journal": status.get("kr_journal", []),
        "us_journal": status.get("us_journal", []),
        "config": status.get("config", {}),
        "blackbox": blackbox,
        "ai_committee": status.get("ai_committee", {}),
    }


def _result_has_order_activity(result: dict | None) -> bool:
    if not isinstance(result, dict):
        return False
    return any(str(a.get("status", "")) in ("FILLED", "ORDERED") for a in result.get("actions", []) or [])


class Handler(BaseHTTPRequestHandler):
    def _send_bytes(self, status_code: int, body: bytes, content_type: str) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _authorized(self, query: dict) -> bool:
        if not STATUS_TOKEN:
            return False

        header_token = str(
            self.headers.get("X-Song-Status-Token", "") or ""
        ).strip()

        auth = str(
            self.headers.get("Authorization", "") or ""
        ).strip()
        bearer_token = ""
        if auth.lower().startswith("bearer "):
            bearer_token = auth[7:].strip()

        query_token = ""
        values = query.get("token", [])
        if values:
            query_token = str(values[0] or "").strip()

        return any(
            candidate
            and secrets.compare_digest(candidate, STATUS_TOKEN)
            for candidate in (
                header_token,
                bearer_token,
                query_token,
            )
        )

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = parse_qs(
            parsed.query,
            keep_blank_values=True,
        )

        if path == "/ping":
            self._send_bytes(
                200,
                b"song-trader-worker-ok",
                "text/plain; charset=utf-8",
            )
            return

        if path == "/health":
            status = load_status()
            payload = {
                "ok": True,
                "running": bool(status.get("running")),
                "heartbeat_at": status.get("heartbeat_at"),
                "version": status.get("version"),
                "status_protected": True,
                "blackbox_ok": bool((status.get("blackbox") or {}).get("ok")),
            }
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
            return

        if path == "/":
            body = json.dumps(
                {
                    "ok": True,
                    "service": "song-trader-worker",
                    "status_protected": True,
                },
                ensure_ascii=False,
            ).encode("utf-8")
            self._send_bytes(
                200,
                body,
                "application/json; charset=utf-8",
            )
            return

        if path == "/replay-kr-ai-committee":
            values = query.get("view", [])
            supplied = str(values[0] or "").strip() if values else ""

            if not CHATGPT_VIEW_TOKEN:
                self._send_bytes(
                    503,
                    b'{"ok":false,"error":"assistant view not configured"}',
                    "application/json; charset=utf-8",
                )
                return

            if not (
                supplied
                and secrets.compare_digest(
                    supplied.encode("utf-8"),
                    CHATGPT_VIEW_TOKEN.encode("utf-8"),
                )
            ):
                self._send_bytes(
                    401,
                    b'{"ok":false,"error":"unauthorized"}',
                    "application/json; charset=utf-8",
                )
                return

            kr_now = datetime.now(KST)
            us_now = datetime.now(ET)
            kr_live = (
                kr_now.weekday() < 5
                and dtime(8, 20) <= kr_now.time() < dtime(15, 40)
            )
            us_live = (
                us_now.weekday() < 5
                and dtime(9, 20) <= us_now.time() < dtime(16, 10)
            )
            if kr_live or us_live:
                body = json.dumps(
                    {
                        "ok": False,
                        "error": "AI replay disabled during live trading windows",
                        "message": (
                            "실시간 Worker 보호를 위해 한국장/미국장 운영시간에는 "
                            "AI 투자위원회 리플레이를 실행하지 않습니다."
                        ),
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                self._send_bytes(423, body, "application/json; charset=utf-8")
                return

            if not _KR_REPLAY_LOCK.acquire(blocking=False):
                body = json.dumps(
                    {
                        "ok": False,
                        "error": "replay already running",
                        "message": "다른 국내 리플레이가 실행 중입니다. 잠시 후 다시 시도하세요.",
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                self._send_bytes(409, body, "application/json; charset=utf-8")
                return

            try:
                date_values = query.get("date", [])
                date_text = (
                    str(date_values[0] or "").strip()
                    if date_values else "2026-08-10"
                )
                refresh_values = query.get("refresh", [])
                refresh = bool(
                    refresh_values
                    and str(refresh_values[0] or "").strip().lower()
                    in ("1", "true", "yes", "on")
                )
                code_values = query.get("codes", [])
                if code_values:
                    codes = [
                        x.strip().zfill(6)
                        for x in str(code_values[0] or "").split(",")
                        if x.strip()
                    ][:80]
                else:
                    codes = None

                try:
                    replay = run_kr_ai_committee_replay(
                        date_text=date_text,
                        codes=codes,
                        refresh=refresh,
                    )
                    body = json.dumps(
                        replay,
                        ensure_ascii=False,
                        default=str,
                    ).encode("utf-8")
                    self._send_bytes(200, body, "application/json; charset=utf-8")
                except Exception as e:
                    log(
                        "KR AI 위원회 리플레이 오류: "
                        f"{type(e).__name__}: {e}"
                    )
                    body = json.dumps(
                        {
                            "ok": False,
                            "error": f"{type(e).__name__}: {e}",
                        },
                        ensure_ascii=False,
                    ).encode("utf-8")
                    self._send_bytes(500, body, "application/json; charset=utf-8")
            finally:
                _KR_REPLAY_LOCK.release()
            return

        if path == "/replay-kr-ai-committee-v2":
            values = query.get("view", [])
            supplied = str(values[0] or "").strip() if values else ""

            if not CHATGPT_VIEW_TOKEN:
                self._send_bytes(
                    503,
                    b'{"ok":false,"error":"assistant view not configured"}',
                    "application/json; charset=utf-8",
                )
                return

            if not (
                supplied
                and secrets.compare_digest(
                    supplied.encode("utf-8"),
                    CHATGPT_VIEW_TOKEN.encode("utf-8"),
                )
            ):
                self._send_bytes(
                    401,
                    b'{"ok":false,"error":"unauthorized"}',
                    "application/json; charset=utf-8",
                )
                return

            kr_now = datetime.now(KST)
            us_now = datetime.now(ET)
            kr_live = (
                kr_now.weekday() < 5
                and dtime(8, 20) <= kr_now.time() < dtime(15, 40)
            )
            us_live = (
                us_now.weekday() < 5
                and dtime(9, 20) <= us_now.time() < dtime(16, 10)
            )
            if kr_live or us_live:
                body = json.dumps(
                    {
                        "ok": False,
                        "error": "AI V2 replay disabled during live trading windows",
                        "message": (
                            "실시간 Worker 보호를 위해 한국장/미국장 운영시간에는 "
                            "AI 투자위원회 V2 리플레이를 실행하지 않습니다."
                        ),
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                self._send_bytes(423, body, "application/json; charset=utf-8")
                return

            if not _KR_REPLAY_LOCK.acquire(blocking=False):
                body = json.dumps(
                    {
                        "ok": False,
                        "error": "replay already running",
                        "message": "다른 국내 리플레이가 실행 중입니다. 잠시 후 다시 시도하세요.",
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                self._send_bytes(409, body, "application/json; charset=utf-8")
                return

            try:
                date_values = query.get("date", [])
                date_text = (
                    str(date_values[0] or "").strip()
                    if date_values else "2026-08-10"
                )
                refresh_values = query.get("refresh", [])
                refresh = bool(
                    refresh_values
                    and str(refresh_values[0] or "").strip().lower()
                    in ("1", "true", "yes", "on")
                )
                code_values = query.get("codes", [])
                if code_values:
                    codes = [
                        x.strip().zfill(6)
                        for x in str(code_values[0] or "").split(",")
                        if x.strip()
                    ][:80]
                else:
                    codes = None

                try:
                    replay = run_kr_ai_committee_replay_v2(
                        date_text=date_text,
                        codes=codes,
                        refresh=refresh,
                    )
                    body = json.dumps(
                        replay,
                        ensure_ascii=False,
                        default=str,
                    ).encode("utf-8")
                    self._send_bytes(200, body, "application/json; charset=utf-8")
                except Exception as e:
                    log(
                        "KR AI 위원회 V2 리플레이 오류: "
                        f"{type(e).__name__}: {e}"
                    )
                    body = json.dumps(
                        {
                            "ok": False,
                            "error": f"{type(e).__name__}: {e}",
                        },
                        ensure_ascii=False,
                    ).encode("utf-8")
                    self._send_bytes(500, body, "application/json; charset=utf-8")
            finally:
                _KR_REPLAY_LOCK.release()
            return

        if path == "/replay-kr-ai-committee-v2-summary":
            values = query.get("view", [])
            supplied = str(values[0] or "").strip() if values else ""

            if not CHATGPT_VIEW_TOKEN:
                self._send_bytes(
                    503,
                    b'{"ok":false,"error":"assistant view not configured"}',
                    "application/json; charset=utf-8",
                )
                return

            if not (
                supplied
                and secrets.compare_digest(
                    supplied.encode("utf-8"),
                    CHATGPT_VIEW_TOKEN.encode("utf-8"),
                )
            ):
                self._send_bytes(
                    401,
                    b'{"ok":false,"error":"unauthorized"}',
                    "application/json; charset=utf-8",
                )
                return

            date_values = query.get("dates", [])
            if date_values:
                dates = [
                    x.strip()
                    for x in str(date_values[0] or "").split(",")
                    if x.strip()
                ][:20]
            else:
                dates = None

            summary = summarize_kr_ai_committee_replays_v2(dates)
            body = json.dumps(
                summary,
                ensure_ascii=False,
                default=str,
            ).encode("utf-8")
            self._send_bytes(200, body, "application/json; charset=utf-8")
            return

        if path == "/replay-kr-ai-committee-summary":
            values = query.get("view", [])
            supplied = str(values[0] or "").strip() if values else ""

            if not CHATGPT_VIEW_TOKEN:
                self._send_bytes(
                    503,
                    b'{"ok":false,"error":"assistant view not configured"}',
                    "application/json; charset=utf-8",
                )
                return

            if not (
                supplied
                and secrets.compare_digest(
                    supplied.encode("utf-8"),
                    CHATGPT_VIEW_TOKEN.encode("utf-8"),
                )
            ):
                self._send_bytes(
                    401,
                    b'{"ok":false,"error":"unauthorized"}',
                    "application/json; charset=utf-8",
                )
                return

            date_values = query.get("dates", [])
            if date_values:
                dates = [
                    x.strip()
                    for x in str(date_values[0] or "").split(",")
                    if x.strip()
                ][:20]
            else:
                dates = None

            summary = summarize_kr_ai_committee_replays(dates)
            body = json.dumps(
                summary,
                ensure_ascii=False,
                default=str,
            ).encode("utf-8")
            self._send_bytes(200, body, "application/json; charset=utf-8")
            return

        if path == "/replay-kr-trades":
            values = query.get("view", [])
            supplied = str(values[0] or "").strip() if values else ""

            if not CHATGPT_VIEW_TOKEN:
                self._send_bytes(
                    503,
                    b'{"ok":false,"error":"assistant view not configured"}',
                    "application/json; charset=utf-8",
                )
                return

            if not (
                supplied
                and secrets.compare_digest(
                    supplied.encode("utf-8"),
                    CHATGPT_VIEW_TOKEN.encode("utf-8"),
                )
            ):
                self._send_bytes(
                    401,
                    b'{"ok":false,"error":"unauthorized"}',
                    "application/json; charset=utf-8",
                )
                return

            # 하나의 Worker가 실제 주문도 담당하므로 한국장/미국장 실시간 시간에는
            # 리플레이를 막아 CPU/네트워크 경합을 피한다.
            kr_now = datetime.now(KST)
            us_now = datetime.now(ET)
            kr_live = (
                kr_now.weekday() < 5
                and dtime(8, 20) <= kr_now.time() < dtime(15, 40)
            )
            us_live = (
                us_now.weekday() < 5
                and dtime(9, 20) <= us_now.time() < dtime(16, 10)
            )
            if kr_live or us_live:
                body = json.dumps(
                    {
                        "ok": False,
                        "error": "KR replay disabled during live trading windows",
                        "message": (
                            "실시간 자동매매 Worker 보호를 위해 한국장/미국장 "
                            "운영시간에는 국내 리플레이를 실행하지 않습니다."
                        ),
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                self._send_bytes(423, body, "application/json; charset=utf-8")
                return

            if not _KR_REPLAY_LOCK.acquire(blocking=False):
                body = json.dumps(
                    {
                        "ok": False,
                        "error": "replay already running",
                        "message": "국내 리플레이가 이미 실행 중입니다. 첫 실행은 데이터 준비 때문에 잠시 걸릴 수 있습니다.",
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                self._send_bytes(409, body, "application/json; charset=utf-8")
                return

            try:
                date_values = query.get("date", [])
                date_text = (
                    str(date_values[0] or "").strip()
                    if date_values else "2026-08-14"
                )

                refresh_values = query.get("refresh", [])
                refresh = bool(
                    refresh_values
                    and str(refresh_values[0] or "").strip().lower()
                    in ("1", "true", "yes", "on")
                )

                code_values = query.get("codes", [])
                if code_values:
                    codes = [
                        x.strip().zfill(6)
                        for x in str(code_values[0] or "").split(",")
                        if x.strip()
                    ][:80]
                else:
                    codes = None

                try:
                    replay = run_kr_trade_replay(
                        date_text=date_text,
                        codes=codes,
                        use_cache=not refresh,
                    )
                    body = json.dumps(
                        replay,
                        ensure_ascii=False,
                        default=str,
                    ).encode("utf-8")
                    self._send_bytes(
                        200,
                        body,
                        "application/json; charset=utf-8",
                    )
                except Exception as e:
                    log(
                        "KR 거래 리플레이 오류: "
                        f"{type(e).__name__}: {e}"
                    )
                    body = json.dumps(
                        {
                            "ok": False,
                            "error": f"{type(e).__name__}: {e}",
                        },
                        ensure_ascii=False,
                    ).encode("utf-8")
                    self._send_bytes(
                        500,
                        body,
                        "application/json; charset=utf-8",
                    )
            finally:
                _KR_REPLAY_LOCK.release()
            return

        if path == "/replay-kr-buy2-abc":
            values = query.get("view", [])
            supplied = str(values[0] or "").strip() if values else ""

            if not CHATGPT_VIEW_TOKEN:
                self._send_bytes(
                    503,
                    b'{"ok":false,"error":"assistant view not configured"}',
                    "application/json; charset=utf-8",
                )
                return

            if not (
                supplied
                and secrets.compare_digest(
                    supplied.encode("utf-8"),
                    CHATGPT_VIEW_TOKEN.encode("utf-8"),
                )
            ):
                self._send_bytes(
                    401,
                    b'{"ok":false,"error":"unauthorized"}',
                    "application/json; charset=utf-8",
                )
                return

            kr_now = datetime.now(KST)
            us_now = datetime.now(ET)
            kr_live = (
                kr_now.weekday() < 5
                and dtime(8, 20) <= kr_now.time() < dtime(15, 40)
            )
            us_live = (
                us_now.weekday() < 5
                and dtime(9, 20) <= us_now.time() < dtime(16, 10)
            )
            if kr_live or us_live:
                body = json.dumps(
                    {
                        "ok": False,
                        "error": "KR replay disabled during live trading windows",
                        "message": (
                            "실시간 자동매매 Worker 보호를 위해 한국장/미국장 "
                            "운영시간에는 국내 BUY2 비교를 실행하지 않습니다."
                        ),
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                self._send_bytes(
                    423, body, "application/json; charset=utf-8"
                )
                return

            if not _KR_REPLAY_LOCK.acquire(blocking=False):
                body = json.dumps(
                    {
                        "ok": False,
                        "error": "replay already running",
                        "message": "국내 리플레이가 이미 실행 중입니다. 잠시 후 다시 시도하세요.",
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                self._send_bytes(
                    409, body, "application/json; charset=utf-8"
                )
                return

            try:
                date_values = query.get("date", [])
                date_text = (
                    str(date_values[0] or "").strip()
                    if date_values else "2026-08-10"
                )

                code_values = query.get("codes", [])
                if code_values:
                    codes = [
                        x.strip().zfill(6)
                        for x in str(code_values[0] or "").split(",")
                        if x.strip()
                    ][:80]
                else:
                    codes = None

                try:
                    replay = compare_kr_buy2_strategies(
                        date_text=date_text,
                        codes=codes,
                    )
                    body = json.dumps(
                        replay,
                        ensure_ascii=False,
                        default=str,
                    ).encode("utf-8")
                    self._send_bytes(
                        200,
                        body,
                        "application/json; charset=utf-8",
                    )
                except Exception as e:
                    log(
                        "KR BUY2 A/B/C 리플레이 오류: "
                        f"{type(e).__name__}: {e}"
                    )
                    body = json.dumps(
                        {
                            "ok": False,
                            "error": f"{type(e).__name__}: {e}",
                        },
                        ensure_ascii=False,
                    ).encode("utf-8")
                    self._send_bytes(
                        500,
                        body,
                        "application/json; charset=utf-8",
                    )
            finally:
                _KR_REPLAY_LOCK.release()
            return

        if path == "/replay-us-reentry-abc":
            values = query.get("view", [])
            supplied = (
                str(values[0] or "").strip()
                if values else ""
            )

            if not CHATGPT_VIEW_TOKEN:
                self._send_bytes(
                    503,
                    b'{"ok":false,"error":"assistant view not configured"}',
                    "application/json; charset=utf-8",
                )
                return

            if not (
                supplied
                and secrets.compare_digest(
                    supplied.encode("utf-8"),
                    CHATGPT_VIEW_TOKEN.encode("utf-8"),
                )
            ):
                self._send_bytes(
                    401,
                    b'{"ok":false,"error":"unauthorized"}',
                    "application/json; charset=utf-8",
                )
                return

            us_now = datetime.now(ET)
            if (
                us_now.weekday() < 5
                and dtime(9, 20)
                <= us_now.time()
                < dtime(16, 10)
            ):
                body = json.dumps(
                    {
                        "ok": False,
                        "error": "reentry ABC replay disabled during US trading window",
                        "message": (
                            "미국장 실시간 Worker 보호를 위해 "
                            "09:20~16:10 ET에는 재진입 A/B/C를 실행하지 않습니다."
                        ),
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                self._send_bytes(
                    423,
                    body,
                    "application/json; charset=utf-8",
                )
                return

            if not _REPLAY_LOCK.acquire(blocking=False):
                body = json.dumps(
                    {
                        "ok": False,
                        "error": "replay already running",
                        "message": (
                            "이미 리플레이가 실행 중입니다. "
                            "잠시 후 다시 시도하세요."
                        ),
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                self._send_bytes(
                    409,
                    body,
                    "application/json; charset=utf-8",
                )
                return

            try:
                date_values = query.get("dates", [])
                if date_values:
                    dates = [
                        x.strip()
                        for x in str(
                            date_values[0] or ""
                        ).split(",")
                        if x.strip()
                    ][:10]
                else:
                    dates = ["2026-08-12"]

                symbol_values = query.get("symbols", [])
                if symbol_values:
                    symbols = [
                        x.strip().upper()
                        for x in str(
                            symbol_values[0] or ""
                        ).split(",")
                        if x.strip()
                    ][:40]
                else:
                    symbols = list(US_UNIVERSE)[:40]

                try:
                    replay = compare_reentry_control_strategies(
                        dates=dates,
                        symbols=symbols,
                    )

                    payload = {
                        "ok": replay.get("ok", True),
                        "version": replay.get("version"),
                        "dates": replay.get("dates", dates),
                        "universe_count": replay.get(
                            "universe_count",
                            len(symbols),
                        ),
                        "entry_fixed": replay.get("entry_fixed", ""),
                        "buy2_fixed": replay.get("buy2_fixed", ""),
                        "exit_fixed": replay.get("exit_fixed", ""),
                        "strategies": replay.get("strategies", {}),
                        "aggregate": replay.get("aggregate", []),
                        "daily": replay.get("daily", []),
                        "recommended_by_replay": replay.get(
                            "recommended_by_replay",
                            "",
                        ),
                        "warning": replay.get("warning", ""),
                    }

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

                except Exception as e:
                    log(
                        "재진입 제어 A/B/C 리플레이 오류: "
                        f"{type(e).__name__}: {e}"
                    )
                    body = json.dumps(
                        {
                            "ok": False,
                            "error": f"{type(e).__name__}: {e}",
                        },
                        ensure_ascii=False,
                    ).encode("utf-8")
                    self._send_bytes(
                        500,
                        body,
                        "application/json; charset=utf-8",
                    )
            finally:
                _REPLAY_LOCK.release()

            return

        if path == "/replay-us-exit-abc":
            values = query.get("view", [])
            supplied = (
                str(values[0] or "").strip()
                if values else ""
            )

            if not CHATGPT_VIEW_TOKEN:
                self._send_bytes(
                    503,
                    b'{"ok":false,"error":"assistant view not configured"}',
                    "application/json; charset=utf-8",
                )
                return

            if not (
                supplied
                and secrets.compare_digest(
                    supplied.encode("utf-8"),
                    CHATGPT_VIEW_TOKEN.encode("utf-8"),
                )
            ):
                self._send_bytes(
                    401,
                    b'{"ok":false,"error":"unauthorized"}',
                    "application/json; charset=utf-8",
                )
                return

            us_now = datetime.now(ET)
            if (
                us_now.weekday() < 5
                and dtime(9, 20)
                <= us_now.time()
                < dtime(16, 10)
            ):
                body = json.dumps(
                    {
                        "ok": False,
                        "error": "exit ABC replay disabled during US trading window",
                        "message": (
                            "미국장 실시간 Worker 보호를 위해 "
                            "09:20~16:10 ET에는 조기청산 A/B/C를 실행하지 않습니다."
                        ),
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                self._send_bytes(
                    423,
                    body,
                    "application/json; charset=utf-8",
                )
                return

            if not _REPLAY_LOCK.acquire(
                blocking=False
            ):
                body = json.dumps(
                    {
                        "ok": False,
                        "error": "replay already running",
                        "message": (
                            "이미 리플레이가 실행 중입니다. "
                            "잠시 후 다시 시도하세요."
                        ),
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                self._send_bytes(
                    409,
                    body,
                    "application/json; charset=utf-8",
                )
                return

            try:
                date_values = query.get(
                    "dates",
                    [],
                )
                if date_values:
                    dates = [
                        x.strip()
                        for x in str(
                            date_values[0] or ""
                        ).split(",")
                        if x.strip()
                    ][:10]
                else:
                    dates = ["2026-08-14"]

                symbol_values = query.get(
                    "symbols",
                    [],
                )
                if symbol_values:
                    symbols = [
                        x.strip().upper()
                        for x in str(
                            symbol_values[0] or ""
                        ).split(",")
                        if x.strip()
                    ][:40]
                else:
                    symbols = list(
                        US_UNIVERSE
                    )[:40]

                try:
                    replay = compare_exit_strategies(
                        dates=dates,
                        symbols=symbols,
                    )

                    payload = {
                        "ok": replay.get(
                            "ok",
                            True,
                        ),
                        "version": replay.get(
                            "version"
                        ),
                        "dates": replay.get(
                            "dates",
                            dates,
                        ),
                        "universe_count": replay.get(
                            "universe_count",
                            len(symbols),
                        ),
                        "entry_fixed": replay.get(
                            "entry_fixed",
                            "",
                        ),
                        "buy2_fixed": replay.get(
                            "buy2_fixed",
                            "",
                        ),
                        "strategies": replay.get(
                            "strategies",
                            {},
                        ),
                        "aggregate": replay.get(
                            "aggregate",
                            [],
                        ),
                        "daily": replay.get(
                            "daily",
                            [],
                        ),
                        "recommended_by_replay": (
                            replay.get(
                                "recommended_by_replay",
                                "",
                            )
                        ),
                        "warning": replay.get(
                            "warning",
                            "",
                        ),
                    }

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

                except Exception as e:
                    log(
                        "조기청산 A/B/C 리플레이 오류: "
                        f"{type(e).__name__}: {e}"
                    )
                    body = json.dumps(
                        {
                            "ok": False,
                            "error": (
                                f"{type(e).__name__}: {e}"
                            ),
                        },
                        ensure_ascii=False,
                    ).encode("utf-8")
                    self._send_bytes(
                        500,
                        body,
                        "application/json; charset=utf-8",
                    )
            finally:
                _REPLAY_LOCK.release()

            return

        if path == "/replay-us-exit-diagnose":
            values = query.get("view", [])
            supplied = (
                str(values[0] or "").strip()
                if values else ""
            )

            if not CHATGPT_VIEW_TOKEN:
                self._send_bytes(
                    503,
                    b'{"ok":false,"error":"assistant view not configured"}',
                    "application/json; charset=utf-8",
                )
                return

            if not (
                supplied
                and secrets.compare_digest(
                    supplied.encode("utf-8"),
                    CHATGPT_VIEW_TOKEN.encode("utf-8"),
                )
            ):
                self._send_bytes(
                    401,
                    b'{"ok":false,"error":"unauthorized"}',
                    "application/json; charset=utf-8",
                )
                return

            us_now = datetime.now(ET)
            if (
                us_now.weekday() < 5
                and dtime(9, 20) <= us_now.time() < dtime(16, 10)
            ):
                body = json.dumps(
                    {
                        "ok": False,
                        "error": "exit diagnose disabled during US trading window",
                        "message": (
                            "미국장 실시간 Worker 보호를 위해 "
                            "09:20~16:10 ET에는 조기청산 진단을 실행하지 않습니다."
                        ),
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                self._send_bytes(
                    423,
                    body,
                    "application/json; charset=utf-8",
                )
                return

            if not _REPLAY_LOCK.acquire(blocking=False):
                body = json.dumps(
                    {
                        "ok": False,
                        "error": "replay already running",
                        "message": "이미 리플레이가 실행 중입니다. 잠시 후 다시 시도하세요.",
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                self._send_bytes(
                    409,
                    body,
                    "application/json; charset=utf-8",
                )
                return

            try:
                date_values = query.get("date", [])
                date_text = (
                    str(date_values[0] or "").strip()
                    if date_values else "2026-08-14"
                )
                if not date_text:
                    date_text = "2026-08-14"

                symbol_values = query.get("symbols", [])
                if symbol_values:
                    symbols = [
                        x.strip().upper()
                        for x in str(symbol_values[0] or "").split(",")
                        if x.strip()
                    ][:40]
                else:
                    symbols = list(US_UNIVERSE)[:40]

                try:
                    result = diagnose_exit_state_day(
                        date_text=date_text,
                        symbols=symbols,
                    )

                    payload = {
                        "ok": result.get("ok", True),
                        "version": result.get("version"),
                        "date": result.get("date", date_text),
                        "config": result.get("config", {}),
                        "summary": result.get("summary", {}),
                        "episodes": result.get("episodes", []),
                        "note": result.get("note", ""),
                    }

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

                except Exception as e:
                    log(
                        "US 조기청산 진단 오류: "
                        f"{type(e).__name__}: {e}"
                    )
                    body = json.dumps(
                        {
                            "ok": False,
                            "error": f"{type(e).__name__}: {e}",
                        },
                        ensure_ascii=False,
                    ).encode("utf-8")
                    self._send_bytes(
                        500,
                        body,
                        "application/json; charset=utf-8",
                    )
            finally:
                _REPLAY_LOCK.release()

            return

        if path == "/replay-us-diagnose":
            values = query.get("view", [])
            supplied = str(values[0] or "").strip() if values else ""

            if not CHATGPT_VIEW_TOKEN:
                self._send_bytes(
                    503,
                    b'{"ok":false,"error":"assistant view not configured"}',
                    "application/json; charset=utf-8",
                )
                return

            if not (
                supplied
                and secrets.compare_digest(
                    supplied.encode("utf-8"),
                    CHATGPT_VIEW_TOKEN.encode("utf-8"),
                )
            ):
                self._send_bytes(
                    401,
                    b'{"ok":false,"error":"unauthorized"}',
                    "application/json; charset=utf-8",
                )
                return

            us_now = datetime.now(ET)
            if (
                us_now.weekday() < 5
                and dtime(9, 20) <= us_now.time() < dtime(16, 10)
            ):
                body = json.dumps(
                    {
                        "ok": False,
                        "error": "diagnose disabled during US trading window",
                        "message": (
                            "미국장 실시간 Worker 보호를 위해 "
                            "09:20~16:10 ET에는 손실원인 진단 리플레이를 실행하지 않습니다."
                        ),
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                self._send_bytes(423, body, "application/json; charset=utf-8")
                return

            if not _REPLAY_LOCK.acquire(blocking=False):
                body = json.dumps(
                    {
                        "ok": False,
                        "error": "replay already running",
                        "message": "이미 리플레이가 실행 중입니다. 잠시 후 다시 시도하세요.",
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                self._send_bytes(409, body, "application/json; charset=utf-8")
                return

            try:
                date_values = query.get("date", [])
                date_text = (
                    str(date_values[0] or "").strip()
                    if date_values else "2026-08-14"
                )
                if not date_text:
                    date_text = "2026-08-14"

                symbol_values = query.get("symbols", [])
                if symbol_values:
                    symbols = [
                        x.strip().upper()
                        for x in str(symbol_values[0] or "").split(",")
                        if x.strip()
                    ][:40]
                else:
                    symbols = list(US_UNIVERSE)[:40]

                try:
                    result = diagnose_trade_day(
                        date_text=date_text,
                        symbols=symbols,
                    )
                    payload = {
                        "ok": result.get("ok", True),
                        "version": result.get("version"),
                        "date": result.get("date", date_text),
                        "config": result.get("config", {}),
                        "summary": result.get("summary", {}),
                        "episodes": result.get("episodes", []),
                        "note": result.get("note", ""),
                    }
                    body = json.dumps(
                        payload,
                        ensure_ascii=False,
                        default=str,
                    ).encode("utf-8")
                    self._send_bytes(200, body, "application/json; charset=utf-8")
                except Exception as e:
                    log(
                        "US 손실원인 진단 오류: "
                        f"{type(e).__name__}: {e}"
                    )
                    body = json.dumps(
                        {
                            "ok": False,
                            "error": f"{type(e).__name__}: {e}",
                        },
                        ensure_ascii=False,
                    ).encode("utf-8")
                    self._send_bytes(500, body, "application/json; charset=utf-8")
            finally:
                _REPLAY_LOCK.release()

            return

        if path == "/replay-us-sizing-abc":
            values = query.get("view", [])
            supplied = (
                str(values[0] or "").strip()
                if values else ""
            )

            if not CHATGPT_VIEW_TOKEN:
                self._send_bytes(
                    503,
                    b'{"ok":false,"error":"assistant view not configured"}',
                    "application/json; charset=utf-8",
                )
                return

            if not (
                supplied
                and secrets.compare_digest(
                    supplied.encode("utf-8"),
                    CHATGPT_VIEW_TOKEN.encode("utf-8"),
                )
            ):
                self._send_bytes(
                    401,
                    b'{"ok":false,"error":"unauthorized"}',
                    "application/json; charset=utf-8",
                )
                return

            us_now = datetime.now(ET)
            if (
                us_now.weekday() < 5
                and dtime(9, 20)
                <= us_now.time()
                < dtime(16, 10)
            ):
                body = json.dumps(
                    {
                        "ok": False,
                        "error": "sizing ABC replay disabled during US trading window",
                        "message": (
                            "미국장 실시간 Worker 보호를 위해 "
                            "09:20~16:10 ET에는 선진입 크기 리플레이를 실행하지 않습니다."
                        ),
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                self._send_bytes(
                    423,
                    body,
                    "application/json; charset=utf-8",
                )
                return

            if not _REPLAY_LOCK.acquire(
                blocking=False
            ):
                body = json.dumps(
                    {
                        "ok": False,
                        "error": "replay already running",
                        "message": (
                            "이미 리플레이가 실행 중입니다. "
                            "잠시 후 다시 시도하세요."
                        ),
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                self._send_bytes(
                    409,
                    body,
                    "application/json; charset=utf-8",
                )
                return

            try:
                date_values = query.get(
                    "dates",
                    [],
                )
                if date_values:
                    dates = [
                        x.strip()
                        for x in str(
                            date_values[0] or ""
                        ).split(",")
                        if x.strip()
                    ][:10]
                else:
                    dates = ["2026-08-14"]

                symbol_values = query.get(
                    "symbols",
                    [],
                )
                if symbol_values:
                    symbols = [
                        x.strip().upper()
                        for x in str(
                            symbol_values[0] or ""
                        ).split(",")
                        if x.strip()
                    ][:40]
                else:
                    symbols = list(
                        US_UNIVERSE
                    )[:40]

                try:
                    replay = (
                        compare_entry_sizing_strategies(
                            dates=dates,
                            symbols=symbols,
                        )
                    )

                    payload = {
                        "ok": replay.get(
                            "ok",
                            True,
                        ),
                        "version": replay.get(
                            "version"
                        ),
                        "dates": replay.get(
                            "dates",
                            dates,
                        ),
                        "universe_count": replay.get(
                            "universe_count",
                            len(symbols),
                        ),
                        "buy2_fixed": replay.get(
                            "buy2_fixed",
                            "B_STRICT",
                        ),
                        "market_filter": replay.get(
                            "market_filter",
                            "NONE",
                        ),
                        "strategies": replay.get(
                            "strategies",
                            {},
                        ),
                        "aggregate": replay.get(
                            "aggregate",
                            [],
                        ),
                        "daily": replay.get(
                            "daily",
                            [],
                        ),
                        "recommended_by_replay": (
                            replay.get(
                                "recommended_by_replay",
                                "",
                            )
                        ),
                        "warning": replay.get(
                            "warning",
                            "",
                        ),
                    }

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

                except Exception as e:
                    log(
                        "선진입 크기 A/B/C 리플레이 오류: "
                        f"{type(e).__name__}: {e}"
                    )

                    body = json.dumps(
                        {
                            "ok": False,
                            "error": (
                                f"{type(e).__name__}: {e}"
                            ),
                        },
                        ensure_ascii=False,
                    ).encode("utf-8")

                    self._send_bytes(
                        500,
                        body,
                        "application/json; charset=utf-8",
                    )
            finally:
                _REPLAY_LOCK.release()

            return

        if path == "/replay-us-confirm-abc":
            values = query.get("view", [])
            supplied = str(values[0] or "").strip() if values else ""

            if not CHATGPT_VIEW_TOKEN:
                self._send_bytes(
                    503,
                    b'{"ok":false,"error":"assistant view not configured"}',
                    "application/json; charset=utf-8",
                )
                return

            if not (
                supplied
                and secrets.compare_digest(
                    supplied.encode("utf-8"),
                    CHATGPT_VIEW_TOKEN.encode("utf-8"),
                )
            ):
                self._send_bytes(
                    401,
                    b'{"ok":false,"error":"unauthorized"}',
                    "application/json; charset=utf-8",
                )
                return

            us_now = datetime.now(ET)
            if (
                us_now.weekday() < 5
                and dtime(9, 20) <= us_now.time() < dtime(16, 10)
            ):
                body = json.dumps(
                    {
                        "ok": False,
                        "error": "confirmation ABC replay disabled during US trading window",
                        "message": (
                            "미국장 실시간 Worker 보호를 위해 "
                            "09:20~16:10 ET에는 BUY1 연속신호 리플레이를 실행하지 않습니다."
                        ),
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                self._send_bytes(
                    423,
                    body,
                    "application/json; charset=utf-8",
                )
                return

            if not _REPLAY_LOCK.acquire(blocking=False):
                body = json.dumps(
                    {
                        "ok": False,
                        "error": "replay already running",
                        "message": "이미 리플레이가 실행 중입니다. 잠시 후 다시 시도하세요.",
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                self._send_bytes(
                    409,
                    body,
                    "application/json; charset=utf-8",
                )
                return

            try:
                date_values = query.get("dates", [])
                if date_values:
                    dates = [
                        x.strip()
                        for x in str(date_values[0] or "").split(",")
                        if x.strip()
                    ][:10]
                else:
                    dates = ["2026-08-14"]

                symbol_values = query.get("symbols", [])
                if symbol_values:
                    symbols = [
                        x.strip().upper()
                        for x in str(symbol_values[0] or "").split(",")
                        if x.strip()
                    ][:40]
                else:
                    symbols = list(US_UNIVERSE)[:40]

                try:
                    replay = compare_buy1_confirmation_strategies(
                        dates=dates,
                        symbols=symbols,
                    )

                    payload = {
                        "ok": replay.get("ok", True),
                        "version": replay.get("version"),
                        "dates": replay.get("dates", dates),
                        "universe_count": replay.get("universe_count", len(symbols)),
                        "buy2_fixed": replay.get("buy2_fixed", "B_STRICT"),
                        "market_filter": replay.get("market_filter", "NONE"),
                        "aggregate": replay.get("aggregate", []),
                        "daily": replay.get("daily", []),
                        "recommended_by_replay": replay.get("recommended_by_replay", ""),
                        "warning": replay.get("warning", ""),
                    }

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
                except Exception as e:
                    log(
                        f"BUY1 연속신호 A/B/C 리플레이 오류: "
                        f"{type(e).__name__}: {e}"
                    )

                    body = json.dumps(
                        {
                            "ok": False,
                            "error": f"{type(e).__name__}: {e}",
                        },
                        ensure_ascii=False,
                    ).encode("utf-8")

                    self._send_bytes(
                        500,
                        body,
                        "application/json; charset=utf-8",
                    )
            finally:
                _REPLAY_LOCK.release()

            return

        if path == "/replay-us-entry-abc":
            values = query.get("view", [])
            supplied = str(values[0] or "").strip() if values else ""

            if not CHATGPT_VIEW_TOKEN:
                self._send_bytes(
                    503,
                    b'{"ok":false,"error":"assistant view not configured"}',
                    "application/json; charset=utf-8",
                )
                return

            if not (
                supplied
                and secrets.compare_digest(
                    supplied.encode("utf-8"),
                    CHATGPT_VIEW_TOKEN.encode("utf-8"),
                )
            ):
                self._send_bytes(
                    401,
                    b'{"ok":false,"error":"unauthorized"}',
                    "application/json; charset=utf-8",
                )
                return

            us_now = datetime.now(ET)
            if (
                us_now.weekday() < 5
                and dtime(9, 20) <= us_now.time() < dtime(16, 10)
            ):
                body = json.dumps(
                    {
                        "ok": False,
                        "error": "entry ABC replay disabled during US trading window",
                        "message": (
                            "미국장 실시간 Worker 보호를 위해 "
                            "09:20~16:10 ET에는 BUY1 시장필터 리플레이를 실행하지 않습니다."
                        ),
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                self._send_bytes(
                    423,
                    body,
                    "application/json; charset=utf-8",
                )
                return

            if not _REPLAY_LOCK.acquire(blocking=False):
                body = json.dumps(
                    {
                        "ok": False,
                        "error": "replay already running",
                        "message": "이미 리플레이가 실행 중입니다. 잠시 후 다시 시도하세요.",
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                self._send_bytes(
                    409,
                    body,
                    "application/json; charset=utf-8",
                )
                return

            try:
                date_values = query.get("dates", [])
                if date_values:
                    dates = [
                        x.strip()
                        for x in str(date_values[0] or "").split(",")
                        if x.strip()
                    ][:10]
                else:
                    dates = ["2026-08-14"]

                symbol_values = query.get("symbols", [])
                if symbol_values:
                    symbols = [
                        x.strip().upper()
                        for x in str(symbol_values[0] or "").split(",")
                        if x.strip()
                    ][:40]
                else:
                    symbols = list(US_UNIVERSE)[:40]

                try:
                    replay = compare_buy1_market_strategies(
                        dates=dates,
                        symbols=symbols,
                    )

                    payload = {
                        "ok": replay.get("ok", True),
                        "version": replay.get("version"),
                        "dates": replay.get("dates", dates),
                        "universe_count": replay.get("universe_count", len(symbols)),
                        "buy2_fixed": replay.get("buy2_fixed", "B_STRICT"),
                        "aggregate": replay.get("aggregate", []),
                        "daily": replay.get("daily", []),
                        "recommended_by_replay": replay.get("recommended_by_replay", ""),
                        "warning": replay.get("warning", ""),
                    }

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
                except Exception as e:
                    log(
                        f"BUY1 시장필터 A/B/C 리플레이 오류: "
                        f"{type(e).__name__}: {e}"
                    )

                    body = json.dumps(
                        {
                            "ok": False,
                            "error": f"{type(e).__name__}: {e}",
                        },
                        ensure_ascii=False,
                    ).encode("utf-8")

                    self._send_bytes(
                        500,
                        body,
                        "application/json; charset=utf-8",
                    )
            finally:
                _REPLAY_LOCK.release()

            return

        if path == "/replay-us-abc":
            values = query.get("view", [])
            supplied = str(values[0] or "").strip() if values else ""

            if not CHATGPT_VIEW_TOKEN:
                self._send_bytes(
                    503,
                    b'{"ok":false,"error":"assistant view not configured"}',
                    "application/json; charset=utf-8",
                )
                return

            if not (
                supplied
                and secrets.compare_digest(
                    supplied.encode("utf-8"),
                    CHATGPT_VIEW_TOKEN.encode("utf-8"),
                )
            ):
                self._send_bytes(
                    401,
                    b'{"ok":false,"error":"unauthorized"}',
                    "application/json; charset=utf-8",
                )
                return

            us_now = datetime.now(ET)
            if (
                us_now.weekday() < 5
                and dtime(9, 20) <= us_now.time() < dtime(16, 10)
            ):
                body = json.dumps(
                    {
                        "ok": False,
                        "error": "ABC replay disabled during US trading window",
                        "message": (
                            "미국장 실시간 Worker 보호를 위해 "
                            "09:20~16:10 ET에는 A/B/C 리플레이를 실행하지 않습니다."
                        ),
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                self._send_bytes(
                    423,
                    body,
                    "application/json; charset=utf-8",
                )
                return

            if not _REPLAY_LOCK.acquire(blocking=False):
                body = json.dumps(
                    {
                        "ok": False,
                        "error": "replay already running",
                        "message": "이미 리플레이가 실행 중입니다. 잠시 후 다시 시도하세요.",
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                self._send_bytes(
                    409,
                    body,
                    "application/json; charset=utf-8",
                )
                return

            try:
                date_values = query.get("dates", [])
                if date_values:
                    dates = [
                        x.strip()
                        for x in str(date_values[0] or "").split(",")
                        if x.strip()
                    ][:10]
                else:
                    dates = [
                        "2026-08-10",
                        "2026-08-11",
                        "2026-08-12",
                        "2026-08-13",
                        "2026-08-14",
                    ]

                symbol_values = query.get("symbols", [])
                if symbol_values:
                    symbols = [
                        x.strip().upper()
                        for x in str(symbol_values[0] or "").split(",")
                        if x.strip()
                    ][:40]
                else:
                    symbols = list(US_UNIVERSE)[:40]

                try:
                    replay = compare_buy2_strategies(
                        dates=dates,
                        symbols=symbols,
                    )

                    payload = {
                        "ok": replay.get("ok", True),
                        "version": replay.get("version"),
                        "dates": replay.get("dates", dates),
                        "universe_count": replay.get("universe_count", len(symbols)),
                        "aggregate": replay.get("aggregate", []),
                        "daily": replay.get("daily", []),
                        "recommended_by_replay": replay.get("recommended_by_replay", ""),
                        "warning": replay.get("warning", ""),
                    }

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
                except Exception as e:
                    log(
                        f"BUY2 A/B/C 리플레이 오류: "
                        f"{type(e).__name__}: {e}"
                    )

                    body = json.dumps(
                        {
                            "ok": False,
                            "error": f"{type(e).__name__}: {e}",
                        },
                        ensure_ascii=False,
                    ).encode("utf-8")

                    self._send_bytes(
                        500,
                        body,
                        "application/json; charset=utf-8",
                    )
            finally:
                _REPLAY_LOCK.release()

            return

        if path == "/replay-us-trades":
            values = query.get("view", [])
            supplied = str(values[0] or "").strip() if values else ""

            if not CHATGPT_VIEW_TOKEN:
                self._send_bytes(
                    503,
                    b'{"ok":false,"error":"assistant view not configured"}',
                    "application/json; charset=utf-8",
                )
                return

            if not (
                supplied
                and secrets.compare_digest(
                    supplied.encode("utf-8"),
                    CHATGPT_VIEW_TOKEN.encode("utf-8"),
                )
            ):
                self._send_bytes(
                    401,
                    b'{"ok":false,"error":"unauthorized"}',
                    "application/json; charset=utf-8",
                )
                return

            # 실시간 미국장 보호
            us_now = datetime.now(ET)
            if (
                us_now.weekday() < 5
                and dtime(9, 20) <= us_now.time() < dtime(16, 10)
            ):
                body = json.dumps(
                    {
                        "ok": False,
                        "error": "trade replay disabled during US trading window",
                        "message": (
                            "미국장 실시간 Worker 보호를 위해 "
                            "09:20~16:10 ET에는 실제매매 리플레이를 실행하지 않습니다."
                        ),
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                self._send_bytes(
                    423,
                    body,
                    "application/json; charset=utf-8",
                )
                return

            if not _REPLAY_LOCK.acquire(blocking=False):
                body = json.dumps(
                    {
                        "ok": False,
                        "error": "replay already running",
                        "message": "이미 리플레이가 실행 중입니다. 잠시 후 다시 시도하세요.",
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                self._send_bytes(
                    409,
                    body,
                    "application/json; charset=utf-8",
                )
                return

            try:
                date_values = query.get("date", [])
                date_text = (
                    str(date_values[0] or "").strip()
                    if date_values else "2026-08-14"
                )
                if not date_text:
                    date_text = "2026-08-14"

                symbol_values = query.get("symbols", [])
                if symbol_values:
                    symbol_text = str(symbol_values[0] or "").strip()
                    symbols = [
                        x.strip().upper()
                        for x in symbol_text.split(",")
                        if x.strip()
                    ][:40]
                else:
                    symbols = list(US_UNIVERSE)[:40]

                try:
                    replay = run_trade_replay(
                        date_text=date_text,
                        symbols=symbols,
                    )

                    payload = {
                        "ok": bool(replay.get("ok", True)),
                        "version": replay.get("version"),
                        "date": replay.get("date", date_text),
                        "universe_count": replay.get("universe_count", len(symbols)),
                        "summary": replay.get("summary", {}),
                        "symbols": replay.get("symbols", []),
                        "events": replay.get("events", []),
                        "assumptions": replay.get("assumptions", {}),
                    }

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
                except Exception as e:
                    log(
                        f"US 실제매매 리플레이 오류: "
                        f"{type(e).__name__}: {e}"
                    )

                    body = json.dumps(
                        {
                            "ok": False,
                            "error": f"{type(e).__name__}: {e}",
                        },
                        ensure_ascii=False,
                    ).encode("utf-8")

                    self._send_bytes(
                        500,
                        body,
                        "application/json; charset=utf-8",
                    )
            finally:
                _REPLAY_LOCK.release()

            return

        if path == "/replay-us":
            values = query.get("view", [])
            supplied = str(values[0] or "").strip() if values else ""

            if not CHATGPT_VIEW_TOKEN:
                self._send_bytes(
                    503,
                    b'{"ok":false,"error":"assistant view not configured"}',
                    "application/json; charset=utf-8",
                )
                return

            if not (
                supplied
                and secrets.compare_digest(
                    supplied.encode("utf-8"),
                    CHATGPT_VIEW_TOKEN.encode("utf-8"),
                )
            ):
                self._send_bytes(
                    401,
                    b'{"ok":false,"error":"unauthorized"}',
                    "application/json; charset=utf-8",
                )
                return

            # 실제 미국 정규장 근처에는 리플레이를 막아
            # yfinance 다운로드가 실시간 Worker 속도에 영향을 주지 않게 한다.
            us_now = datetime.now(ET)
            if (
                us_now.weekday() < 5
                and dtime(9, 20) <= us_now.time() < dtime(16, 10)
            ):
                body = json.dumps(
                    {
                        "ok": False,
                        "error": "replay disabled during US trading window",
                        "message": "미국장 실시간 Worker 보호를 위해 09:20~16:10 ET에는 리플레이를 실행하지 않습니다.",
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                self._send_bytes(
                    423,
                    body,
                    "application/json; charset=utf-8",
                )
                return

            if not _REPLAY_LOCK.acquire(blocking=False):
                body = json.dumps(
                    {
                        "ok": False,
                        "error": "replay already running",
                        "message": "이미 리플레이가 실행 중입니다. 잠시 후 다시 시도하세요.",
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                self._send_bytes(
                    409,
                    body,
                    "application/json; charset=utf-8",
                )
                return

            try:
                date_values = query.get("date", [])
                date_text = (
                    str(date_values[0] or "").strip()
                    if date_values else "2026-08-14"
                )
                if not date_text:
                    date_text = "2026-08-14"

                symbol_values = query.get("symbols", [])
                if symbol_values:
                    symbol_text = str(symbol_values[0] or "").strip()
                    symbols = [
                        x.strip().upper()
                        for x in symbol_text.split(",")
                        if x.strip()
                    ][:40]
                else:
                    # 별도 symbols 파라미터가 없으면
                    # 실제 Worker가 감시하는 미국 전체 유니버스를 그대로 리플레이한다.
                    symbols = list(US_UNIVERSE)[:40]

                if not symbols:
                    symbols = ["TSLA", "NVDA", "AMAT"]

                step_values = query.get("step", [])
                try:
                    step_minutes = int(
                        str(step_values[0] or "5")
                        if step_values else "5"
                    )
                except Exception:
                    step_minutes = 5
                step_minutes = max(1, min(30, step_minutes))

                try:
                    replay = run_replay(
                        date_text=date_text,
                        symbols=symbols,
                        step_minutes=step_minutes,
                    )
                    payload = {
                        "ok": bool(replay.get("ok", True)),
                        "date": replay.get("date", date_text),
                        "symbols": replay.get("symbols", symbols),
                        "universe_count": len(replay.get("symbols", symbols) or []),
                        "step_minutes": replay.get("step_minutes", step_minutes),
                        "generated_at": replay.get("generated_at"),
                        "summary": replay.get("summary", []),
                    }
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
                except Exception as e:
                    log(
                        f"US 리플레이 오류: "
                        f"{type(e).__name__}: {e}"
                    )
                    body = json.dumps(
                        {
                            "ok": False,
                            "error": f"{type(e).__name__}: {e}",
                        },
                        ensure_ascii=False,
                    ).encode("utf-8")
                    self._send_bytes(
                        500,
                        body,
                        "application/json; charset=utf-8",
                    )
            finally:
                _REPLAY_LOCK.release()
            return

        if path not in ("/status", "/public-status", "/assistant-status"):
            self._send_bytes(
                404,
                b'{"ok":false,"error":"not found"}',
                "application/json; charset=utf-8",
            )
            return

        if path == "/assistant-status":
            values = query.get("view", [])
            supplied = str(values[0] or "").strip() if values else ""

            if not CHATGPT_VIEW_TOKEN:
                self._send_bytes(
                    503,
                    b'{"ok":false,"error":"assistant view not configured"}',
                    "application/json; charset=utf-8",
                )
                return

            if not (
                supplied
                and secrets.compare_digest(
                    supplied.encode("utf-8"),
                    CHATGPT_VIEW_TOKEN.encode("utf-8"),
                )
            ):
                self._send_bytes(
                    401,
                    b'{"ok":false,"error":"unauthorized"}',
                    "application/json; charset=utf-8",
                )
                return

            status = load_status()
            payload = _assistant_status_payload(status)
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
            return

        if not STATUS_TOKEN:
            self._send_bytes(
                503,
                b'{"ok":false,"error":"status auth not configured"}',
                "application/json; charset=utf-8",
            )
            return

        if not self._authorized(query):
            self._send_bytes(
                401,
                b'{"ok":false,"error":"unauthorized"}',
                "application/json; charset=utf-8",
            )
            return

        status = load_status()
        payload = _public_status_payload(status)

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
_REPLAY_LOCK = threading.Lock()
_KR_REPLAY_LOCK = threading.Lock()


def _start_http_server(port: int):
    try:
        srv = ThreadingHTTPServer(
            ("0.0.0.0", int(port)),
            Handler,
        )
        threading.Thread(
            target=srv.serve_forever,
            daemon=True,
            name=f"status-http-{port}",
        ).start()
        _HTTP_SERVERS.append(srv)
        log(f"상태 HTTP 서버 시작: 0.0.0.0:{port}")
        return srv
    except OSError as e:
        log(
            f"상태 HTTP 서버 포트 {port} 시작 실패: "
            f"{type(e).__name__}: {e}"
        )
        return None


def start_server():
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
        raise RuntimeError(
            f"상태 HTTP 서버 시작 실패: 시도 포트 {ports}"
        )
    return started


def stop_handler(signum, frame):
    global RUNNING
    RUNNING = False


def _wait_for_initial_kis_token(client: KISClient) -> bool:
    """KIS 토큰 발급이 일시적으로 실패해도 Worker 프로세스를 죽이지 않습니다.

    HTTP 상태 서버는 이미 떠 있으므로, 인증이 회복될 때까지 STARTING 상태로
    남아 있으면서 천천히 재시도합니다. 실제 주문/스캔은 토큰 성공 뒤에만 시작합니다.
    """
    retry_seconds = max(30, int(os.getenv("KIS_TOKEN_STARTUP_RETRY_SECONDS", "60")))
    attempt = 0

    while RUNNING:
        attempt += 1
        try:
            client.get_token()
            if attempt > 1:
                log(f"✅ KIS 토큰 인증 회복 완료 · {attempt}번째 시도")
            return True
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            # 토큰/키 원문은 기록하지 않고 오류 유형과 메시지만 상태에 남깁니다.
            log(
                f"⚠️ KIS 토큰 인증 대기 · {attempt}번째 실패 · "
                f"{retry_seconds}초 후 재시도 · {err}"
            )
            save_status(
                running=True,
                status="starting",
                stage="KIS_AUTH_WAIT",
                stage_message=(
                    "🟡 KIS 인증이 일시적으로 지연 중입니다. "
                    f"Worker는 종료하지 않고 {retry_seconds}초 후 자동 재시도합니다."
                ),
                last_error=err[:1200],
            )

            # SIGTERM/SIGINT에 바로 반응할 수 있도록 1초 단위로 쉽니다.
            for _ in range(retry_seconds):
                if not RUNNING:
                    return False
                time.sleep(1)

    return False


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
    if not _wait_for_initial_kis_token(client):
        save_status(
            running=False,
            status="stopped",
            stage="STOPPED",
            stage_message="⏹️ Worker 종료",
        )
        return 0

    old = load_status()
    journal_store = _load_journal_store()
    kr_journal = _merge_journals(old.get("kr_journal", []) or [], journal_store.get("kr_journal", []) or [])
    us_journal = _merge_journals(old.get("us_journal", []) or [], journal_store.get("us_journal", []) or [])
    kr_top5 = pd.DataFrame(old.get("kr_top5", []) or [])
    us_top5 = pd.DataFrame(old.get("us_top5", []) or [])
    # 시작 직후 첫 재스캔 전에는 TOP5만 가지고 있어도 안전하게 관리하고,
    # 첫 미국 스캔부터는 전체 30종목 순위를 보유종목 관리에 사용한다.
    us_ranked = us_top5.copy()

    kr_last_scan = 0.0
    us_last_scan = 0.0
    last_balance_sync = 0.0

    log(
        f"V2.11.2a Worker 시작 env={ENV}, "
        f"KR_BUY2={'ON' if CFG.kr_buy2_enabled else 'OFF'}, "
        f"주문요청={'ON' if EXECUTE else 'OFF'}, 실제주문={'ON' if EFFECTIVE_EXECUTE else 'DRY'}, "
        f"PRIMARY={'YES' if PRIMARY else 'NO'}, 계좌끝4자리={client.account_no[-4:]}"
    )

    bb = _blackbox_status_safe()
    if bb.get("ok"):
        log(
            f"✅ 거래 블랙박스 준비 완료 "
            f"판단={bb.get('decision_count', 0)}건 "
            f"주문={bb.get('order_count', 0)}건"
        )
    else:
        log(f"⚠️ 거래 블랙박스 상태 이상: {bb.get('error', '')}")

    ai_status = ai_committee_status()
    if ai_status.get("ok") and ai_status.get("configured"):
        log(
            "🧠 AI 투자위원회 V1 준비 완료 "
            f"mode=SHADOW model={ai_status.get('model')} "
            "주문영향=NONE"
        )
    elif AI_COMMITTEE_IMPORT_ERROR:
        log(
            "⚠️ AI 투자위원회 import 오류 — 자동매매는 계속 동작: "
            + AI_COMMITTEE_IMPORT_ERROR
        )
    else:
        log(
            "🧠 AI 투자위원회 V1 SHADOW 대기 "
            "OPENAI_API_KEY 미설정 — 자동매매 영향 없음"
        )

    if not AI_COMMITTEE_REPLAY_IMPORT_ERROR:
        log("🧪 AI 투자위원회 국내 리플레이 V1 준비 완료 주문영향=NONE")
    else:
        log(
            "⚠️ AI 투자위원회 리플레이 import 오류 — 자동매매 영향 없음: "
            + AI_COMMITTEE_REPLAY_IMPORT_ERROR
        )

    initial_updates = {
        "running": True,
        "status": "running",
        "kr_journal": kr_journal,
        "us_journal": us_journal,
        "account": {"last4": client.account_no[-4:], "product_code": client.product_code},
        "ai_committee": ai_committee_status(),
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
                        try:
                            append_kr_top5_snapshot(kr_top5, kr_now)
                        except Exception as snap_e:
                            log(f"KR TOP5 snapshot 저장 경고: {type(snap_e).__name__}: {snap_e}")

                        # 그림자 AI 투자위원회: background 평가만 수행.
                        # 실제 run_kr_cycle의 주문 판단에는 절대 연결하지 않는다.
                        try:
                            prev_status = load_status()
                            ai_submit_shadow_scan(
                                "KR",
                                kr_top5,
                                kr_now,
                                portfolio_context={
                                    "env": ENV,
                                    "current_positions": len(
                                        prev_status.get("kr_holdings", []) or []
                                    ),
                                    "max_positions": int(CFG.max_positions),
                                    "daily_budget": int(CFG.kr_daily_budget),
                                    "buy2_enabled": bool(CFG.kr_buy2_enabled),
                                },
                            )
                        except Exception as ai_e:
                            log(
                                "KR AI 위원회 경고(주문 영향 없음): "
                                f"{type(ai_e).__name__}: {ai_e}"
                            )
                    except Exception as e:
                        # 오래된 후보로 뒤늦게 진입하지 않도록 즉시 비운다.
                        kr_top5 = pd.DataFrame()
                        kr_last_scan = time.time()
                        updates["kr_scan_error"] = f"{type(e).__name__}: {e}"
                        log(f"KR TOP5 오류(신규매수 차단): {type(e).__name__}: {e}")

                kr_result = run_kr_cycle(client, kr_top5, CFG, EFFECTIVE_EXECUTE, source="WORKER")
                try:
                    ai_record_worker_result("KR", kr_result)
                except Exception as ai_e:
                    log(
                        "KR AI 결과기록 경고(주문 영향 없음): "
                        f"{type(ai_e).__name__}: {ai_e}"
                    )
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
                    # 주문 직후 한 번 더 조회. 실제 체결확정은 다음 KIS 잔고 동기화 진단으로 확인.
                    time.sleep(max(0, min(3, int(CFG.confirm_wait_seconds))))
                    _apply_balance_sync(updates, "kr", _kr_balance_snapshot(client))

            if us_now.weekday() < 5 and dtime(9, 30) <= us_now.time() < dtime(16, 0):
                if (
                    us_now.time() < _clock_env("US_FORCE_EXIT_TIME", "15:50")
                    and ((time.time() - us_last_scan) >= US_RESCAN_SECONDS or us_top5.empty)
                ):
                    try:
                        us_ranked = build_us_ranked(US_UNIVERSE)
                        us_top5 = us_ranked.head(5).reset_index(drop=True)
                        us_last_scan = time.time()
                        updates["us_scan_error"] = ""

                        # 미국도 같은 그림자 위원회로 평가하되,
                        # 실제 C_PAPER 주문/청산 규칙에는 관여하지 않는다.
                        try:
                            prev_status = load_status()
                            ai_submit_shadow_scan(
                                "US",
                                us_ranked,
                                us_now,
                                portfolio_context={
                                    "env": ENV,
                                    "current_positions": len(
                                        prev_status.get("us_holdings", []) or []
                                    ),
                                    "max_positions": int(CFG.max_positions),
                                    "daily_budget_usd": float(CFG.us_daily_budget_usd),
                                    "strategy": "C_PAPER",
                                },
                            )
                        except Exception as ai_e:
                            log(
                                "US AI 위원회 경고(주문 영향 없음): "
                                f"{type(ai_e).__name__}: {ai_e}"
                            )
                    except Exception as e:
                        # 오래된 후보로 잘못 진입/조기청산하지 않도록 둘 다 비운다.
                        us_ranked = pd.DataFrame()
                        us_top5 = pd.DataFrame()
                        us_last_scan = time.time()
                        updates["us_scan_error"] = f"{type(e).__name__}: {e}"
                        log(f"US 전체순위 오류(신규매수·C전략 조기청산 차단): {type(e).__name__}: {e}")

                us_result = run_us_cycle(client, us_ranked, CFG, EFFECTIVE_EXECUTE, source="WORKER")
                try:
                    ai_record_worker_result("US", us_result)
                except Exception as ai_e:
                    log(
                        "US AI 결과기록 경고(주문 영향 없음): "
                        f"{type(ai_e).__name__}: {ai_e}"
                    )
                us_journal = _journal_append(us_journal, us_result, "US")
                updates["us_last_result"] = _safe_result(us_result)

                if _result_has_order_activity(us_result):
                    _apply_balance_sync(updates, "us", _us_balance_snapshot(client))

            updates.update({
                "ai_committee": ai_committee_status(),
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
                    "us_strategy": "C_PAPER",
                    "ai_committee": "V1_SHADOW_NO_ORDER_CONTROL",
                    "us_buy2": (
                        f"B_STRICT +{CFG.us_buy2_strict_trigger_pct:.2f}% / "
                        f"{CFG.us_buy2_min_hold_minutes:.0f}분 / TOP{CFG.us_buy2_max_rank} / "
                        f"{CFG.us_buy2_min_score:.0f}점"
                    ),
                    "us_early_exit": (
                        f"{CFG.us_early_exit_min_hold_minutes:.0f}분 약화+정체 / "
                        f"조기청산 {CFG.us_pause_after_early_exits_count}회→"
                        f"{CFG.us_pause_new_entries_minutes:.0f}분 휴식"
                    ),
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

    try:
        ai_committee_shutdown()
    except Exception:
        pass
    save_status(running=False, status="stopped", stage="STOPPED", stage_message="⏹️ Worker 종료")
    return 0


if __name__ == "__main__":
    sys.exit(main())

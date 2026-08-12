from __future__ import annotations

import json
import os
import signal
import sys
import time
import traceback
from datetime import datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from trader_core import Settings, KISClient, discover_domestic_candidates, score_ticker
from auto_engine import AutoConfig, run_domestic_cycle, run_overseas_cycle
from trend_strategy import score_leader_trend


KST = ZoneInfo("Asia/Seoul")
ET = ZoneInfo("America/New_York")

STATE_DIR = Path(os.getenv("SONG_TRADER_STATE_DIR", "/tmp/song_trader"))
STATE_DIR.mkdir(parents=True, exist_ok=True)
WORKER_LOG = STATE_DIR / "worker.log"
WORKER_STATUS = STATE_DIR / "worker_status.json"

WORKER_ENV = os.getenv("SONG_WORKER_ENV", "demo").strip().lower()
if WORKER_ENV not in ("demo", "real"):
    WORKER_ENV = "demo"

ALLOW_REAL_WORKER = os.getenv("ALLOW_REAL_WORKER", "false").lower() in ("1", "true", "yes", "on")
REAL_CONFIRM = os.getenv("REAL_WORKER_CONFIRM", "") == "I-UNDERSTAND-LIVE-ORDERS"

EXECUTE_ORDERS = os.getenv("WORKER_EXECUTE_ORDERS", "false").lower() in ("1", "true", "yes", "on")
LOOP_SECONDS = max(30, int(os.getenv("WORKER_LOOP_SECONDS", "60")))
KR_RESCAN_SECONDS = max(120, int(os.getenv("KR_RESCAN_SECONDS", "300")))
US_RESCAN_SECONDS = max(120, int(os.getenv("US_RESCAN_SECONDS", "300")))

# 국내 모의 기본값: 하루 1,000만원 / 종목당 1,000만원
KR_DAILY_BUDGET = int(os.getenv("KR_DAILY_BUDGET", "10000000"))
KR_PER_STOCK_BUDGET = int(os.getenv("KR_PER_STOCK_BUDGET", "10000000"))

US_DAILY_BUDGET = float(os.getenv("US_DAILY_BUDGET", "1500"))
US_PER_STOCK_BUDGET = float(os.getenv("US_PER_STOCK_BUDGET", "600"))

MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "3"))
STOP_LOSS = float(os.getenv("STOP_LOSS_PCT", "3.0"))
TAKE1 = float(os.getenv("TAKE1_PCT", "3.0"))
TAKE2 = float(os.getenv("TAKE2_PCT", "5.0"))

MIN_SCORE = float(os.getenv("MIN_COMBINED_SCORE", "65"))
DEMO_MIN_SCORE = float(os.getenv("DEMO_MIN_COMBINED_SCORE", "40"))
LEADER_EXCEPTION_MIN_LEAD = float(os.getenv("LEADER_EXCEPTION_MIN_LEAD", "75"))
LEADER_EXCEPTION_MIN_COMBINED = float(os.getenv("LEADER_EXCEPTION_MIN_COMBINED", "60"))

US_UNIVERSE = [
    x.strip().upper()
    for x in os.getenv(
        "US_UNIVERSE",
        "AAPL,MSFT,NVDA,AMZN,META,TSLA,AMD,GOOGL,AVGO,NFLX"
    ).split(",")
    if x.strip()
]

RUNNING = True


def log(msg: str) -> None:
    now = datetime.now(KST).isoformat(timespec="seconds")
    line = f"[{now}] {msg}"
    print(line, flush=True)
    try:
        with WORKER_LOG.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def save_status(**kwargs) -> None:
    status = {
        "updated_at": datetime.now(KST).isoformat(timespec="seconds"),
        "env": WORKER_ENV,
        "execute_orders": EXECUTE_ORDERS,
        **kwargs,
    }
    try:
        WORKER_STATUS.write_text(
            json.dumps(status, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    except Exception:
        pass


def stop_handler(signum, frame):
    global RUNNING
    RUNNING = False
    log(f"종료 신호 수신: {signum}")


signal.signal(signal.SIGINT, stop_handler)
signal.signal(signal.SIGTERM, stop_handler)


def in_kr_monitor_window(now=None) -> bool:
    now = now or datetime.now(KST)
    if now.weekday() >= 5:
        return False
    return dtime(8, 30) <= now.time() < dtime(16, 0)


def in_us_monitor_window(now=None) -> bool:
    now = now or datetime.now(ET)
    if now.weekday() >= 5:
        return False
    return dtime(9, 20) <= now.time() < dtime(16, 10)


def current_worker_stage(kr_now=None, us_now=None) -> tuple[str, str]:
    """
    앱 화면에 표시할 worker 현재 단계.
    반환값: (stage_code, 사람이 읽는 메시지)
    """
    kr_now = kr_now or datetime.now(KST)
    us_now = us_now or datetime.now(ET)

    if in_kr_monitor_window(kr_now):
        return "KR_TRADING", "🇰🇷 국내장 자동매매 사이클 실행 중"

    if us_now.weekday() < 5 and dtime(9, 0) <= us_now.time() < dtime(9, 30):
        return "US_PREPARE", "🇺🇸 미국장 후보 준비 중"

    if in_us_monitor_window(us_now):
        return "US_TRADING", "🇺🇸 미국장 자동매매 사이클 실행 중"

    return "WAITING", "⏳ 다음 시장 대기 중"


def build_kr_leaders(client: KISClient) -> pd.DataFrame:
    candidates = discover_domestic_candidates(client, top_n=20)
    if candidates is None or candidates.empty:
        return pd.DataFrame()

    rows = []

    for _, r in candidates.head(12).iterrows():
        code = str(r.get("종목코드", "")).zfill(6)
        if not code:
            continue

        try:
            tech = score_ticker(code, "국내")
        except Exception:
            tech = None

        if not tech:
            continue

        lead = float(r.get("주도주점수", 0) or 0)
        net = int(tech.get("순점수", 0) or 0)

        try:
            from trader_core import _download_yf
            intraday_df = _download_yf(code, "국내")
            trend = score_leader_trend(intraday_df) or {}
        except Exception:
            trend = {}

        trend_score = float(trend.get("추세점수", 0) or 0)
        combined = lead * 0.45 + trend_score * 0.55
        signal_text = str(trend.get("추세판정", "⚪ 추세약함"))

        rows.append({
            "종목코드": code,
            "종목명": r.get("종목명", ""),
            "현재가": r.get("현재가", ""),
            "등락률": r.get("등락률", ""),
            "주도주점수": round(lead, 1),
            "기술순점수": net,
            "추세점수": round(trend_score, 1),
            "종합점수": round(combined, 1),
            "판정": signal_text,
            "진입근거": trend.get("추세이유", ""),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values(
        ["종합점수", "주도주점수", "기술순점수"],
        ascending=[False, False, False],
    ).head(5).reset_index(drop=True)

    labels = ["👑 1위", "🥈 2위", "🥉 3위", "4위", "5위"]
    df.insert(0, "순위", labels[:len(df)])
    return df


def build_us_leaders() -> pd.DataFrame:
    rows = []

    for symbol in US_UNIVERSE:
        try:
            row = score_ticker(symbol, market="미국")
            if row:
                rows.append(row)
        except Exception as e:
            log(f"US score 오류 {symbol}: {type(e).__name__}: {e}")

    if not rows:
        return pd.DataFrame()

    score_df = pd.DataFrame(rows).sort_values(
        ["순점수", "거래량배수"],
        ascending=[False, False],
    )

    top = score_df.head(5).copy().reset_index(drop=True)

    tech100 = ((top["순점수"].clip(-6, 6) + 6) / 12 * 100).astype(float)
    vol_bonus = (top["거래량배수"].clip(0, 2.0) / 2.0 * 10).astype(float)
    top["종합점수"] = (tech100 * 0.9 + vol_bonus).round(1)

    labels = ["⭐ 1위", "⭐ 2위", "⭐ 3위", "4위", "5위"]
    top.insert(0, "순위", labels[:len(top)])
    top["종목코드"] = top["종목"].astype(str)
    top["종목명"] = top["종목"].astype(str)
    top["판정"] = top["종합신호"]
    return top


def make_config() -> AutoConfig:
    cfg = AutoConfig(
        daily_budget=KR_DAILY_BUDGET,
        per_stock_budget=KR_PER_STOCK_BUDGET,
        max_positions=MAX_POSITIONS,
        buy1_pct=50,
        buy2_pct=30,
        buy3_pct=20,
        stop_loss_pct=STOP_LOSS,
        take1_pct=TAKE1,
        take2_pct=TAKE2,
        min_combined_score=MIN_SCORE,
        require_green_signal=True,
    )

    cfg.demo_relaxed_entry_enabled = True
    cfg.demo_min_combined_score = DEMO_MIN_SCORE

    cfg.leader_exception_enabled = True
    cfg.leader_exception_min_lead_score = LEADER_EXCEPTION_MIN_LEAD
    cfg.leader_exception_min_combined_score = LEADER_EXCEPTION_MIN_COMBINED

    cfg.us_daily_budget_usd = US_DAILY_BUDGET
    cfg.us_per_stock_budget_usd = US_PER_STOCK_BUDGET
    cfg.us_last_entry_time = "15:30"
    cfg.us_force_exit_time = "15:50"

    cfg.last_entry_time = "14:50"
    cfg.force_exit_time = "15:15"

    return cfg


def main() -> int:
    if WORKER_ENV == "real" and not (ALLOW_REAL_WORKER and REAL_CONFIRM):
        log(
            "실전 worker 잠금 상태입니다. "
            "ALLOW_REAL_WORKER=true 및 REAL_WORKER_CONFIRM=I-UNDERSTAND-LIVE-ORDERS "
            "둘 다 없으면 시작하지 않습니다."
        )
        return 2

    settings = Settings.from_env()
    client = KISClient(settings=settings, env=WORKER_ENV)
    cfg = make_config()

    client.get_token()

    real_execute = bool(EXECUTE_ORDERS)

    if WORKER_ENV == "demo":
        log(
            "🇰🇷🇺🇸 국내+미국 모의 자동매매 worker 시작 "
            f"(주문전송={'ON' if real_execute else 'DRY'}, 반복 {LOOP_SECONDS}초)"
        )
    else:
        log(
            "⚠️ 국내+미국 실전 worker 시작 "
            f"(주문전송={'ON' if real_execute else 'DRY'})"
        )

    log(
        f"💰 KR 한도: 하루 {KR_DAILY_BUDGET:,}원 / 종목당 {KR_PER_STOCK_BUDGET:,}원 "
        f"(1차 {int(KR_PER_STOCK_BUDGET*0.5):,} / "
        f"2차 {int(KR_PER_STOCK_BUDGET*0.3):,} / "
        f"3차 {int(KR_PER_STOCK_BUDGET*0.2):,})"
    )

    kr_leaders = pd.DataFrame()
    us_leaders = pd.DataFrame()
    kr_last_scan = 0.0
    us_last_scan = 0.0

    while RUNNING:
        cycle_started = time.time()
        kr_now = datetime.now(KST)
        us_now = datetime.now(ET)

        kr_result = None
        us_result = None

        stage_code, stage_message = current_worker_stage(kr_now, us_now)
        save_status(
            running=True,
            status="running",
            stage=stage_code,
            stage_message=stage_message,
            heartbeat_at=datetime.now(KST).isoformat(timespec="seconds"),
            kr_monitor=in_kr_monitor_window(kr_now),
            us_monitor=in_us_monitor_window(us_now),
        )

        try:
            if in_kr_monitor_window(kr_now):
                if (time.time() - kr_last_scan >= KR_RESCAN_SECONDS) or kr_leaders.empty:
                    try:
                        kr_leaders = build_kr_leaders(client)
                        kr_last_scan = time.time()

                        if kr_leaders.empty:
                            log("KR 후보 스캔: TOP5 없음")
                        else:
                            top = kr_leaders.iloc[0]
                            log(
                                f"KR 재스캔 완료: TOP1 "
                                f"{top.get('종목명','')}({top.get('종목코드','')}) "
                                f"점수 {top.get('종합점수','')} / {top.get('판정','')}"
                            )
                    except Exception as e:
                        log(f"KR 후보 스캔 오류: {type(e).__name__}: {e}")

                try:
                    kr_result = run_domestic_cycle(
                        client=client,
                        leader_df=kr_leaders,
                        config=cfg,
                        execute_orders=real_execute,
                    )

                    actions = kr_result.get("actions", []) if kr_result else []
                    if actions:
                        log(f"KR actions: {actions}")

                except Exception as e:
                    log(f"KR cycle 오류: {type(e).__name__}: {e}")
                    log(traceback.format_exc())

            if in_us_monitor_window(us_now):
                if (time.time() - us_last_scan >= US_RESCAN_SECONDS) or us_leaders.empty:
                    try:
                        us_leaders = build_us_leaders()
                        us_last_scan = time.time()

                        if us_leaders.empty:
                            log("US 후보 스캔: TOP5 없음")
                        else:
                            top = us_leaders.iloc[0]
                            log(
                                f"US 재스캔 완료: TOP1 "
                                f"{top.get('종목코드','')} "
                                f"점수 {top.get('종합점수','')} / {top.get('판정','')}"
                            )

                    except Exception as e:
                        log(f"US 후보 스캔 오류: {type(e).__name__}: {e}")

                try:
                    us_result = run_overseas_cycle(
                        client=client,
                        leader_df=us_leaders,
                        config=cfg,
                        execute_orders=real_execute,
                    )

                    actions = us_result.get("actions", []) if us_result else []
                    if actions:
                        log(f"US actions: {actions}")

                except Exception as e:
                    log(f"US cycle 오류: {type(e).__name__}: {e}")
                    log(traceback.format_exc())

            stage_code, stage_message = current_worker_stage(kr_now, us_now)
            save_status(
                running=True,
                status="running",
                stage=stage_code,
                stage_message=stage_message,
                heartbeat_at=datetime.now(KST).isoformat(timespec="seconds"),
                kr_monitor=in_kr_monitor_window(kr_now),
                us_monitor=in_us_monitor_window(us_now),
                kr_top5=kr_leaders.to_dict("records") if not kr_leaders.empty else [],
                us_top5=us_leaders.to_dict("records") if not us_leaders.empty else [],
                kr_last_result=kr_result,
                us_last_result=us_result,
            )

        except Exception as e:
            log(f"worker 메인 루프 오류: {type(e).__name__}: {e}")
            log(traceback.format_exc())

        elapsed = time.time() - cycle_started
        sleep_for = max(1.0, LOOP_SECONDS - elapsed)
        time.sleep(sleep_for)

    save_status(running=False)
    log("worker 종료")
    return 0


if __name__ == "__main__":
    sys.exit(main())

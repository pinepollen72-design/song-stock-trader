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
from auto_engine import AutoConfig, run_domestic_cycle
from trend_strategy import score_leader_trend

KST = ZoneInfo("Asia/Seoul")

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

KR_DAILY_BUDGET = int(os.getenv("KR_DAILY_BUDGET", "10000000"))
KR_PER_STOCK_BUDGET = int(os.getenv("KR_PER_STOCK_BUDGET", "10000000"))

MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "3"))

STOP_LOSS = float(os.getenv("STOP_LOSS_PCT", "3.0"))
TAKE1 = float(os.getenv("TAKE1_PCT", "3.0"))
TAKE2 = float(os.getenv("TAKE2_PCT", "5.0"))

MIN_SCORE = float(os.getenv("MIN_COMBINED_SCORE", "65"))
DEMO_MIN_SCORE = float(os.getenv("DEMO_MIN_COMBINED_SCORE", "40"))

LEADER_EXCEPTION_MIN_LEAD = float(os.getenv("LEADER_EXCEPTION_MIN_LEAD", "75"))
LEADER_EXCEPTION_MIN_COMBINED = float(os.getenv("LEADER_EXCEPTION_MIN_COMBINED", "60"))

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
            json.dumps(
                status,
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
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


def build_kr_leaders(client: KISClient) -> pd.DataFrame:
    """
    핵심 수정:
    - Yahoo/yfinance 데이터가 없는 종목은 '그 종목만' 건너뜁니다.
    - 나머지 후보 분석을 계속합니다.
    """
    candidates = discover_domestic_candidates(
        client,
        top_n=20,
    )

    if candidates is None or candidates.empty:
        return pd.DataFrame()

    rows = []

    for _, r in candidates.head(12).iterrows():
        code = str(r.get("종목코드", "")).zfill(6)
        name = str(r.get("종목명", ""))

        if not (len(code) == 6 and code.isdigit()):
            log(f"KR 후보 SKIP 잘못된 코드: {code} {name}")
            continue

        try:
            tech = score_ticker(
                code,
                "국내",
            )
        except Exception as e:
            log(
                f"KR 기술분석 SKIP {name}({code}): "
                f"{type(e).__name__}: {e}"
            )
            continue

        if not tech:
            log(
                f"KR 기술분석 SKIP {name}({code}): "
                "Yahoo 데이터 없음/부족"
            )
            continue

        lead = float(r.get("주도주점수", 0) or 0)
        net = int(tech.get("순점수", 0) or 0)

        try:
            from trader_core import _download_yf

            intraday_df = _download_yf(
                code,
                "국내",
            )

            if intraday_df is None or intraday_df.empty:
                log(
                    f"KR 추세분석 SKIP {name}({code}): "
                    "Yahoo 5분봉 데이터 없음"
                )
                continue

            trend = score_leader_trend(
                intraday_df
            ) or {}

        except Exception as e:
            log(
                f"KR 추세분석 SKIP {name}({code}): "
                f"{type(e).__name__}: {e}"
            )
            continue

        trend_score = float(
            trend.get("추세점수", 0) or 0
        )

        combined = (
            lead * 0.45
            + trend_score * 0.55
        )

        signal_text = str(
            trend.get(
                "추세판정",
                "⚪ 추세약함",
            )
        )

        rows.append({
            "종목코드": code,
            "종목명": name,
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

    df = (
        pd.DataFrame(rows)
        .sort_values(
            ["종합점수", "주도주점수", "기술순점수"],
            ascending=[False, False, False],
        )
        .head(5)
        .reset_index(drop=True)
    )

    labels = [
        "👑 1위",
        "🥈 2위",
        "🥉 3위",
        "4위",
        "5위",
    ]

    df.insert(
        0,
        "순위",
        labels[:len(df)],
    )

    return df


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

    cfg.last_entry_time = "14:50"
    cfg.force_exit_time = "15:15"

    return cfg


def main() -> int:
    if WORKER_ENV == "real" and not (
        ALLOW_REAL_WORKER
        and REAL_CONFIRM
    ):
        log(
            "실전 worker 잠금 상태입니다. "
            "ALLOW_REAL_WORKER=true 및 "
            "REAL_WORKER_CONFIRM=I-UNDERSTAND-LIVE-ORDERS "
            "둘 다 없으면 시작하지 않습니다."
        )
        return 2

    settings = Settings.from_env()

    client = KISClient(
        settings=settings,
        env=WORKER_ENV,
    )

    cfg = make_config()

    try:
        client.get_token()
    except Exception as e:
        log(
            f"KIS 시작 토큰 오류: "
            f"{type(e).__name__}: {e}"
        )
        return 3

    real_execute = bool(EXECUTE_ORDERS)

    log(
        "🇰🇷 국내 자동매매 worker 시작 "
        f"(환경={WORKER_ENV}, "
        f"주문전송={'ON' if real_execute else 'DRY'}, "
        f"반복={LOOP_SECONDS}초)"
    )

    log(
        f"💰 KR 한도: 하루 {KR_DAILY_BUDGET:,}원 / "
        f"종목당 {KR_PER_STOCK_BUDGET:,}원 "
        f"(1차 {int(KR_PER_STOCK_BUDGET*0.5):,} / "
        f"2차 {int(KR_PER_STOCK_BUDGET*0.3):,} / "
        f"3차 {int(KR_PER_STOCK_BUDGET*0.2):,})"
    )

    kr_leaders = pd.DataFrame()
    kr_last_scan = 0.0

    while RUNNING:
        started = time.time()
        now = datetime.now(KST)

        last_result = None

        try:
            if in_kr_monitor_window(now):
                if (
                    time.time() - kr_last_scan >= KR_RESCAN_SECONDS
                    or kr_leaders.empty
                ):
                    try:
                        kr_leaders = build_kr_leaders(
                            client
                        )
                        kr_last_scan = time.time()

                        if kr_leaders.empty:
                            log(
                                "KR 후보 스캔: TOP5 없음 "
                                "(데이터 없는 종목은 자동 제외)"
                            )
                        else:
                            top = kr_leaders.iloc[0]

                            log(
                                f"KR 재스캔 완료: TOP1 "
                                f"{top.get('종목명','')}"
                                f"({top.get('종목코드','')}) "
                                f"종합 {top.get('종합점수','')} / "
                                f"{top.get('판정','')}"
                            )

                    except Exception as e:
                        log(
                            f"KR 후보 스캔 오류: "
                            f"{type(e).__name__}: {e}"
                        )
                        log(traceback.format_exc())

                try:
                    last_result = run_domestic_cycle(
                        client=client,
                        leader_df=kr_leaders,
                        config=cfg,
                        execute_orders=real_execute,
                    )

                    actions = (
                        last_result.get("actions", [])
                        if last_result else []
                    )

                    diagnostics = (
                        last_result.get("diagnostics", [])
                        if last_result else []
                    )

                    if actions:
                        log(
                            f"KR actions: {actions}"
                        )

                    if diagnostics:
                        # 종목별로 왜 건너뛰었는지 항상 로그에 남김
                        log(
                            f"KR diagnostics: {diagnostics}"
                        )

                    if last_result and last_result.get("message"):
                        log(
                            f"KR message: "
                            f"{last_result.get('message')}"
                        )

                except Exception as e:
                    log(
                        f"KR cycle 오류: "
                        f"{type(e).__name__}: {e}"
                    )
                    log(traceback.format_exc())

            save_status(
                running=True,
                kr_monitor=in_kr_monitor_window(now),
                kr_top5=(
                    kr_leaders.to_dict("records")
                    if not kr_leaders.empty
                    else []
                ),
                kr_last_result=last_result,
            )

        except Exception as e:
            log(
                f"worker 메인 루프 오류: "
                f"{type(e).__name__}: {e}"
            )
            log(traceback.format_exc())

        elapsed = time.time() - started
        time.sleep(
            max(
                1.0,
                LOOP_SECONDS - elapsed,
            )
        )

    save_status(running=False)
    log("worker 종료")
    return 0


if __name__ == "__main__":
    sys.exit(main())

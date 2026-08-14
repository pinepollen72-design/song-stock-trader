from __future__ import annotations

import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

KST = ZoneInfo("Asia/Seoul")

# 단일 Worker 원칙:
# Streamlit 화면은 "주문을 실제로 보내는 단 하나의 Worker"만 봅니다.
# 현재 운영 기준은 Railway Worker입니다.
DEFAULT_WORKER_STATUS_URL = (
    "https://song-stock-trader-production-4d5f.up.railway.app"
)
WORKER_STATUS_URL = (
    os.getenv("WORKER_STATUS_URL")
    or os.getenv("SONG_WORKER_STATUS_URL")
    or DEFAULT_WORKER_STATUS_URL
).rstrip("/")

st.set_page_config(
    page_title="쏭 자동매매 V2",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 쏭 국내·미국 자동매매 V2")
st.caption(
    "TOP5 선정 → 단일 Worker 주문 → 한국투자 실제 잔고 확인 "
    "→ 실행결과/일지 표시. 한국투자 계좌를 최종 기준으로 봅니다."
)


def fetch_status() -> dict:
    try:
        r = requests.get(
            WORKER_STATUS_URL + "/status",
            timeout=8,
            headers={
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
            },
        )
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, dict):
            raise ValueError("Worker /status 응답이 JSON 객체가 아닙니다.")
        return data
    except Exception as e:
        st.error(
            f"Worker 상태조회 실패: {type(e).__name__}: {e}"
        )
        st.caption(f"조회 주소: {WORKER_STATUS_URL}/status")
        return {}


def _age_seconds(raw: str | None):
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=KST)
        return max(
            0,
            int(
                (
                    datetime.now(KST)
                    - dt.astimezone(KST)
                ).total_seconds()
            ),
        )
    except Exception:
        return None


def _fmt_sync_time(raw):
    if not raw:
        return "-"
    try:
        dt = datetime.fromisoformat(str(raw))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=KST)
        return dt.astimezone(KST).strftime("%H:%M:%S")
    except Exception:
        return str(raw)


if st.button(
    "🔄 Worker / 한국투자 상태 새로고침",
    use_container_width=True,
):
    st.rerun()

status = fetch_status()

st.divider()
st.subheader("🤖 자동매매 Worker 상태")

if not status:
    st.error("🔴 Worker 상태 확인 불가")
else:
    age = _age_seconds(status.get("heartbeat_at"))

    if status.get("running") and (
        age is None or age <= 180
    ):
        st.success("🟢 Worker 정상 작동 중")
    else:
        st.error("🔴 Worker 중지 또는 응답 지연")

    st.info(
        status.get(
            "stage_message",
            "상태 메시지 없음",
        )
    )

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "heartbeat",
        f"{age}초 전" if age is not None else "-",
    )
    c2.metric(
        "모드",
        "모의투자"
        if status.get("env") == "demo"
        else "실전투자",
    )
    c3.metric(
        "주문",
        "ON"
        if status.get("execute_orders")
        else "DRY",
    )

    acct = status.get("account", {}) or {}
    if acct.get("last4"):
        st.caption(
            f"KIS 연결 계좌: ****{acct.get('last4')}"
            f"-{acct.get('product_code', '')} · "
            f"Worker {status.get('version', '')}"
        )

    provider = status.get("provider", "")
    if provider:
        st.caption(f"실행 서버: {provider}")

market = st.radio(
    "시장",
    ["🇰🇷 국내", "🇺🇸 미국"],
    horizontal=True,
)
is_kr = market.startswith("🇰🇷")

prefix = "kr" if is_kr else "us"
top_key = f"{prefix}_top5"
result_key = f"{prefix}_last_result"
hold_key = f"{prefix}_holdings"
journal_key = f"{prefix}_journal"
sync_key = f"{prefix}_holdings_sync"

st.divider()
st.subheader("🏆 자동매매 후보 TOP5")

top = pd.DataFrame(status.get(top_key, []) or [])

if top.empty:
    if is_kr:
        st.caption(
            "현재 TOP5가 없습니다. 국내장 종료 후 새 Worker를 "
            "배포했다면 다음 국내장 스캔 때 생성됩니다."
        )
    else:
        st.caption(
            "현재 TOP5가 없습니다. 미국 정규장 스캔 때 생성됩니다."
        )
else:
    st.dataframe(
        top,
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "신규매수는 이 TOP5에 실제 표시된 종목만 주문할 수 있습니다."
    )

st.divider()
st.subheader("💼 한국투자 실제 보유잔고")

sync = status.get(sync_key, {}) or {}
holdings = pd.DataFrame(status.get(hold_key, []) or [])

if sync.get("ok") is False:
    st.error("🔴 한국투자 실제 잔고 조회 실패")
    st.caption(
        f"마지막 시도: {_fmt_sync_time(sync.get('at'))} · "
        f"{sync.get('error', '원인 미상')}"
    )

    if not holdings.empty:
        st.warning(
            "아래 표는 마지막으로 성공했던 잔고입니다. "
            "현재 잔고로 단정하면 안 됩니다."
        )
        st.dataframe(
            holdings,
            use_container_width=True,
            hide_index=True,
        )

elif sync.get("ok") is True:
    count = int(sync.get("count", len(holdings)) or 0)

    st.caption(
        f"한국투자 API 동기화: "
        f"{_fmt_sync_time(sync.get('at'))} · {count}종목"
    )

    if holdings.empty:
        st.success("현재 한국투자 보유잔고 0종목")
    else:
        st.dataframe(
            holdings,
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "이 표는 Worker가 한국투자 API에서 직접 읽은 실제 잔고입니다."
        )

else:
    st.warning(
        "🟡 아직 한국투자 잔고 동기화 결과가 없습니다. "
        "'보유잔고 없음'으로 판단하지 않습니다."
    )

st.divider()
st.subheader("🚀 자동매매 실행 결과")

result = status.get(result_key) or {}
actions = result.get("actions", []) or []
diagnostics = result.get("diagnostics", []) or []

# ---------------------------------------------------------
# 화면 상단 숫자는 "최근 1회 사이클"이 아니라 "오늘 누적 주문 일지" 기준.
# 이렇게 해야 오전에 발생한 매수/매도가 이후 사이클에서 0으로 사라지지 않습니다.
# SKIP은 주문 일지에 저장되지 않으므로 최근 사이클 진단 건수를 표시합니다.
# ---------------------------------------------------------
journal_raw = pd.DataFrame(status.get(journal_key, []) or [])
today_kst = datetime.now(KST).date()

def _today_journal(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()

    # 같은 주문이 상태 변경 과정에서 중복 저장돼 있으면 최신 1건만 집계
    if "_key" in out.columns:
        out = out.drop_duplicates(subset=["_key"], keep="last")

    time_col = next(
        (c for c in ["시간", "time", "timestamp", "at"] if c in out.columns),
        None,
    )
    if time_col is not None:
        parsed = pd.to_datetime(out[time_col], errors="coerce", utc=True)
        local_dates = parsed.dt.tz_convert(KST).dt.date
        out = out.loc[local_dates == today_kst].copy()

    return out


def _norm(value) -> str:
    return str(value or "").strip().upper()


today_journal = _today_journal(journal_raw)

side_col = next(
    (c for c in ["구분", "side", "action"] if c in today_journal.columns),
    None,
)
status_col = next(
    (c for c in ["상태", "status"] if c in today_journal.columns),
    None,
)

today_buy = 0
today_sell = 0
today_fail = 0

if not today_journal.empty and side_col is not None:
    for _, row in today_journal.iterrows():
        side = _norm(row.get(side_col))
        order_status = _norm(row.get(status_col)) if status_col else ""

        is_fail = (
            "실패" in order_status
            or "거절" in order_status
            or order_status in ("REJECT", "REJECTED", "ERROR", "FAILED")
        )

        if is_fail:
            today_fail += 1
            continue

        if side in ("매수", "BUY") or side.startswith("BUY"):
            today_buy += 1
        elif side in ("매도", "SELL") or side.startswith("SELL"):
            today_sell += 1

# 일지가 아직 없는 구버전 Worker에 대한 안전한 fallback
if journal_raw.empty and actions:
    today_buy = sum(
        1
        for a in actions
        if str(a.get("action", "")).upper().startswith("BUY")
        and str(a.get("status", "")).upper() in ("FILLED", "ORDERED")
    )
    today_sell = sum(
        1
        for a in actions
        if str(a.get("action", "")).upper().startswith("SELL")
        and str(a.get("status", "")).upper() in ("FILLED", "ORDERED")
    )
    today_fail = sum(
        1
        for a in actions
        if str(a.get("status", "")).upper() in ("REJECT", "ERROR", "FAILED")
    )

recent_skip = len(diagnostics)

c1, c2, c3, c4 = st.columns(4)
c1.metric("오늘 매수", today_buy)
c2.metric("오늘 매도", today_sell)
c3.metric("최근 SKIP", recent_skip)
c4.metric("오늘 오류", today_fail)

st.caption(
    "매수·매도·오류는 오늘 자동매매 일지 누적 기준입니다. "
    "SKIP은 가장 최근 Worker 사이클 기준입니다."
)

if not result:
    stage = str(status.get("stage", ""))

    if is_kr and stage not in ("KR", "KR_EXIT"):
        st.info(
            "국내장은 현재 매매 사이클을 실행하지 않는 시간입니다. "
            "잔고 동기화만 계속 수행합니다."
        )
    elif (not is_kr) and stage not in ("US", "US_EXIT"):
        st.info(
            "미국 정규장이 열리면 첫 자동매매 실행결과가 표시됩니다."
        )
    else:
        st.caption("아직 첫 자동매매 사이클 결과가 없습니다.")

else:
    if result.get("message"):
        st.info(result.get("message"))

    if actions:
        st.markdown("#### 최근 사이클 주문/액션")
        st.dataframe(
            pd.DataFrame(actions),
            use_container_width=True,
            hide_index=True,
        )

    if diagnostics:
        with st.expander("SKIP 사유 보기"):
            st.dataframe(
                pd.DataFrame(diagnostics),
                use_container_width=True,
                hide_index=True,
            )

    if not actions and not diagnostics:
        st.caption(
            "이번 사이클에는 주문이나 오류가 없었습니다."
        )

st.divider()
st.subheader("📒 자동매매 일지")

journal = pd.DataFrame(
    status.get(journal_key, []) or []
)

if journal.empty:
    st.info(
        "현재 Worker가 한국투자에 보낸 주문 기록이 아직 없습니다."
    )
    st.caption(
        "다음 주문부터 주문접수/체결확인/주문실패가 여기에 기록됩니다."
    )
else:
    if "_key" in journal.columns:
        journal = journal.drop(columns=["_key"])

    st.dataframe(
        journal.iloc[::-1],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "체결확인 = 주문 후 한국투자 실제 잔고 수량 변화까지 확인. "
        "주문접수 = KIS가 주문을 접수했으나 잔고 반영 확인 전."
    )

st.divider()

config = status.get("config", {}) or {}

st.caption(
    f"현재 기준: 종합점수 {config.get('min_score', 50)} 이상 · "
    f"손절 -{config.get('stop_loss_pct', 3)}% · "
    f"1차 익절 +{config.get('take1_pct', 3)}% · "
    f"2차 익절 +{config.get('take2_pct', 5)}% · "
    f"국내 {config.get('kr_force_exit_time', '15:15')} 강제청산 · "
    f"미국 {config.get('us_force_exit_time', '15:50')} ET 강제청산"
)

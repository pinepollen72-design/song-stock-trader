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

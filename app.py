from __future__ import annotations

import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

KST = ZoneInfo("Asia/Seoul")
WORKER_STATUS_URL = os.getenv("WORKER_STATUS_URL", "").rstrip("/")

st.set_page_config(page_title="쏭 자동매매 V2", page_icon="🤖", layout="wide")
st.title("🤖 쏭 국내·미국 자동매매 V2")
st.caption("TOP5 선정 → Worker 주문 → 한국투자 잔고 확인 → 화면/일지 반영. 한국투자 계좌를 최종 기준으로 봅니다.")


def fetch_status() -> dict:
    if not WORKER_STATUS_URL:
        return {}
    try:
        r = requests.get(WORKER_STATUS_URL + "/status", timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Railway Worker 상태조회 실패: {type(e).__name__}: {e}")
        return {}


if st.button("🔄 Railway 상태 새로고침", use_container_width=True):
    st.rerun()

status = fetch_status()

st.divider()
st.subheader("🤖 자동매매 Worker 상태")
if not status:
    st.error("🔴 Worker 상태 확인 불가")
else:
    heartbeat_raw = status.get("heartbeat_at")
    age = None
    if heartbeat_raw:
        try:
            hb = datetime.fromisoformat(heartbeat_raw)
            if hb.tzinfo is None:
                hb = hb.replace(tzinfo=KST)
            age = int((datetime.now(KST) - hb.astimezone(KST)).total_seconds())
        except Exception:
            pass
    if status.get("running") and (age is None or age <= 180):
        st.success("🟢 Worker 정상 작동 중")
    else:
        st.error("🔴 Worker 중지 또는 응답 지연")
    st.info(status.get("stage_message", "상태 메시지 없음"))
    c1, c2, c3 = st.columns(3)
    c1.metric("마지막 heartbeat", f"{age}초 전" if age is not None else "-")
    c2.metric("Worker 모드", "모의투자" if status.get("env") == "demo" else "실전투자")
    c3.metric("주문전송", "ON" if status.get("execute_orders") else "DRY")

market = st.radio("시장", ["🇰🇷 국내", "🇺🇸 미국"], horizontal=True)
is_kr = market.startswith("🇰🇷")
top_key = "kr_top5" if is_kr else "us_top5"
result_key = "kr_last_result" if is_kr else "us_last_result"
hold_key = "kr_holdings" if is_kr else "us_holdings"
journal_key = "kr_journal" if is_kr else "us_journal"

st.divider()
st.subheader("🏆 자동매매 후보 TOP5")
top = pd.DataFrame(status.get(top_key, []) or [])
if top.empty:
    st.caption("현재 TOP5가 없습니다.")
else:
    st.dataframe(top, use_container_width=True, hide_index=True)
    st.caption("중요: 신규매수는 이 TOP5에 실제로 표시된 종목만 주문할 수 있습니다.")

st.divider()
st.subheader("💼 한국투자 실제 보유잔고")
holdings = pd.DataFrame(status.get(hold_key, []) or [])
if holdings.empty:
    # 최신 cycle 안에 잔고가 있으면 대체 표시
    latest = status.get(result_key) or {}
    holdings = pd.DataFrame(latest.get("holdings", []) or [])
if holdings.empty:
    st.success("현재 보유잔고 없음")
else:
    st.dataframe(holdings, use_container_width=True, hide_index=True)
    st.caption("Worker가 마지막으로 한국투자 API에서 조회한 실제 잔고입니다.")

st.divider()
st.subheader("🚀 자동매매 실행 결과")
result = status.get(result_key) or {}
actions = result.get("actions", []) or []
if not result:
    st.caption("아직 실행결과가 없습니다.")
else:
    buy = sum(1 for a in actions if str(a.get("action", "")).startswith("BUY") and a.get("status") in ("FILLED", "ORDERED"))
    sell = sum(1 for a in actions if not str(a.get("action", "")).startswith("BUY") and a.get("status") in ("FILLED", "ORDERED"))
    fail = sum(1 for a in actions if a.get("status") in ("REJECT", "ERROR"))
    skip = len(result.get("diagnostics", []) or [])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("매수", buy); c2.metric("매도", sell); c3.metric("SKIP", skip); c4.metric("오류", fail)
    if result.get("message"):
        st.info(result.get("message"))
    if actions:
        st.dataframe(pd.DataFrame(actions), use_container_width=True, hide_index=True)
    elif result.get("diagnostics"):
        with st.expander("SKIP 사유 보기"):
            st.dataframe(pd.DataFrame(result.get("diagnostics")), use_container_width=True, hide_index=True)

st.divider()
st.subheader("📒 자동매매 일지")
journal = pd.DataFrame(status.get(journal_key, []) or [])
if journal.empty:
    st.caption("새 V2 Worker에서 발생한 주문이 아직 없습니다.")
else:
    if "_key" in journal.columns:
        journal = journal.drop(columns=["_key"])
    st.dataframe(journal.iloc[::-1], use_container_width=True, hide_index=True)
    st.caption("'체결확인'은 주문 후 한국투자 실제 잔고 수량 변화까지 확인된 경우입니다. '주문접수'는 KIS가 주문을 접수했지만 잔고 반영 확인이 아직 안 된 경우입니다.")

st.divider()
config = status.get("config", {}) or {}
st.caption(
    f"현재 기준: 종합점수 {config.get('min_score', 50)} 이상 · 손절 -{config.get('stop_loss_pct', 3)}% · "
    f"1차 익절 +{config.get('take1_pct', 3)}% · 2차 익절 +{config.get('take2_pct', 5)}% · "
    f"국내 {config.get('kr_force_exit_time', '15:15')} 강제청산 · 미국 {config.get('us_force_exit_time', '15:50')} ET 강제청산"
)

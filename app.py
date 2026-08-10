import streamlit as st
import pandas as pd

from trader_core import (
    Settings, KISClient, discover_domestic_candidates,
    score_ticker, split_budget, is_market_open, market_force_exit_time,
    load_trade_log
)

st.set_page_config(page_title="쏭 자동매매", page_icon="🤖", layout="wide")
st.title("🤖 쏭 국내·미국 자동매매")
st.caption("모의/실전 공용 대시보드입니다. 기본값은 모의투자이며 실전은 별도 잠금 해제가 필요합니다.")

settings = Settings.from_streamlit(st.secrets)

with st.sidebar:
    st.header("⚙️ 운용 설정")
    mode = st.radio("운용 모드", ["모의투자", "실전투자"], index=0)
    market = st.radio("시장", ["국내", "미국"], horizontal=True)
    auto_on = st.toggle("🤖 자동매매 ON", value=False)

    budget = st.number_input("1일 최대 신규매수 금액(원)", min_value=10000, value=300000, step=10000)
    per_stock = st.number_input("종목당 최대 금액(원)", min_value=10000, value=100000, step=10000)
    max_positions = st.number_input("최대 동시 보유 종목", min_value=1, max_value=10, value=3)

    st.markdown("**분할매수 비율**")
    b1 = st.number_input("1차 %", 0, 100, 40)
    b2 = st.number_input("2차 %", 0, 100, 30)
    b3 = st.number_input("3차 %", 0, 100, 30)

    stop_loss = st.number_input("손절 %", min_value=0.1, max_value=20.0, value=3.0, step=0.1)
    take1 = st.number_input("1차 익절 %", min_value=0.1, max_value=50.0, value=3.0, step=0.1)
    take2 = st.number_input("2차 익절 %", min_value=0.1, max_value=100.0, value=5.0, step=0.1)

    st.markdown("**실전 잠금**")
    live_phrase = st.text_input("실전 확인문구", type="password", placeholder="실전 운용 시에만 입력")

env = "demo" if mode == "모의투자" else "real"
live_unlocked = (env == "demo") or (
    live_phrase == settings.live_unlock_phrase and settings.allow_live
)

if env == "real":
    if live_unlocked:
        st.error("🔴 실전투자 모드 잠금 해제 상태입니다. 실제 주문이 발생할 수 있습니다.")
    else:
        st.warning("🔒 실전투자 모드는 잠겨 있습니다. 주문은 실행되지 않습니다.")

client = KISClient(settings=settings, env=env)

c1, c2, c3, c4 = st.columns(4)
c1.metric("운용 모드", "모의" if env == "demo" else "실전")
c2.metric("시장", market)
c3.metric("자동매매", "ON" if auto_on else "OFF")
c4.metric("실전 잠금", "해당 없음" if env == "demo" else ("해제" if live_unlocked else "잠김"))

st.divider()
st.subheader("🔑 API 상태")
if st.button("API 연결 확인"):
    try:
        token = client.get_token()
        st.success("✅ 토큰 자동 발급/재사용 정상")
        st.caption(f"토큰 앞부분: {token[:8]}…")
    except Exception as e:
        st.error(f"API 연결 실패: {e}")

st.divider()
st.subheader("🔥 오늘의 후보 탐색")

manual_text = st.text_input(
    "직접 추가할 종목",
    placeholder="국내: 005930,000660 / 미국: AAPL,NVDA,TSLA"
)
manual_symbols = [x.strip().upper() for x in manual_text.split(",") if x.strip()]

if market == "국내":
    if st.button("📊 거래량·거래대금 기반 국내 후보 찾기"):
        try:
            candidates = discover_domestic_candidates(client, top_n=20)
            st.session_state["candidates_kr"] = candidates
            st.session_state.pop("leader_df", None)
        except Exception as e:
            st.error(f"후보 탐색 실패: {e}")

    candidates = st.session_state.get("candidates_kr", pd.DataFrame())

    if not candidates.empty:
        st.success(f"✅ 일반 개별주식 후보 {len(candidates)}개")
        st.dataframe(candidates, use_container_width=True, hide_index=True)

        st.markdown("### 👑 오늘의 대장주 후보 TOP 5")
        st.caption("거래대금·거래량·등락률의 주도주점수와 5분봉 기술점수를 함께 비교합니다.")

        if st.button("👑 대장주 TOP 5 계산"):
            rows = []
            progress = st.progress(0, text="대장주 후보를 분석하고 있어요...")

            # API 순위 상위 12개까지만 5분봉 기술분석하여 모바일 속도/API 부담 완화
            scan = candidates.head(12).copy()
            total = max(len(scan), 1)

            for i, (_, r) in enumerate(scan.iterrows()):
                code = str(r["종목코드"]).zfill(6)
                try:
                    tech = score_ticker(code, "국내")
                except Exception:
                    tech = None

                if tech:
                    lead = float(r.get("주도주점수", 0))
                    net = int(tech.get("순점수", 0))

                    # 기술 순점수 -6~+6을 0~100 범위로 보수적으로 환산
                    tech100 = max(0.0, min(100.0, ((net + 6) / 12) * 100))
                    combined = lead * 0.65 + tech100 * 0.35

                    rows.append({
                        "종목코드": code,
                        "종목명": r.get("종목명", ""),
                        "현재가": r.get("현재가", ""),
                        "등락률": r.get("등락률", ""),
                        "주도주점수": round(lead, 1),
                        "RSI": tech.get("RSI"),
                        "거래량배수": tech.get("거래량배수"),
                        "매수점수": tech.get("매수점수"),
                        "매도점수": tech.get("매도점수"),
                        "기술순점수": net,
                        "종합점수": round(combined, 1),
                        "판정": tech.get("종합신호"),
                    })

                progress.progress((i + 1) / total, text=f"{i+1}/{total} 분석 중")

            progress.empty()

            if rows:
                leader_df = pd.DataFrame(rows).sort_values(
                    ["종합점수", "주도주점수", "기술순점수"],
                    ascending=[False, False, False]
                ).head(5).reset_index(drop=True)

                labels = ["👑 1위", "🥈 2위", "🥉 3위", "4위", "5위"]
                leader_df.insert(0, "순위", labels[:len(leader_df)])
                st.session_state["leader_df"] = leader_df
            else:
                st.warning("5분봉 기술분석 가능한 후보가 없었습니다.")

        leader_df = st.session_state.get("leader_df", pd.DataFrame())
        if not leader_df.empty:
            st.dataframe(leader_df, use_container_width=True, hide_index=True)

            top = leader_df.iloc[0]
            st.info(
                f"👑 현재 1순위 대장주 후보: {top['종목명']} ({top['종목코드']}) · "
                f"종합점수 {top['종합점수']} · 기술판정 {top['판정']}"
            )
            st.caption("대장주 후보는 자동 계산 결과이며 매수 추천이나 수익 보장이 아닙니다.")

        auto_symbols = candidates["종목코드"].astype(str).tolist()
    else:
        auto_symbols = []
else:
    st.info("미국 v1은 직접 관심종목 + 기본 모멘텀 유니버스를 기술점수로 분석합니다.")
    default_us = ["AAPL","MSFT","NVDA","AMZN","META","TSLA","AMD","GOOGL","AVGO","NFLX"]
    auto_symbols = default_us
    st.write("기본 미국 후보:", ", ".join(default_us))

symbols = list(dict.fromkeys(manual_symbols + auto_symbols))

st.divider()
st.subheader("🧠 기술점수")

score_rows = []
if st.button("후보 기술점수 계산"):
    with st.spinner("후보 기술점수를 계산하고 있어요..."):
        for symbol in symbols[:30]:
            try:
                row = score_ticker(symbol, market=market)
                if row:
                    score_rows.append(row)
            except Exception:
                pass

    if score_rows:
        score_df = pd.DataFrame(score_rows).sort_values(
            ["순점수", "거래량배수"], ascending=[False, False]
        )
        st.session_state["score_df"] = score_df

score_df = st.session_state.get("score_df", pd.DataFrame())
if not score_df.empty:
    st.dataframe(score_df, use_container_width=True, hide_index=True)
    st.subheader("⭐ 기술점수 TOP 5")
    st.dataframe(score_df.head(5), use_container_width=True, hide_index=True)

st.divider()
st.subheader("🧪 주문 계산 미리보기")
parts = split_budget(per_stock, [b1, b2, b3])
st.write(f"종목당 {per_stock:,}원 기준 → 1차 {parts[0]:,}원 / 2차 {parts[1]:,}원 / 3차 {parts[2]:,}원")
st.write(f"손절 -{stop_loss:.1f}% / 1차 익절 +{take1:.1f}% / 2차 익절 +{take2:.1f}%")
st.write("당일매매 규칙: 해당 시장 마감 전 남은 당일 포지션 전량 청산")

open_now = is_market_open("KR" if market == "국내" else "US")
st.info(
    f"현재 시장 상태: {'장중' if open_now else '장외'} · "
    f"강제청산 기준시각: {market_force_exit_time('KR' if market == '국내' else 'US')}"
)

st.divider()
st.subheader("🤖 자동매매 엔진")
if auto_on:
    if env == "real" and not live_unlocked:
        st.error("실전 잠금이 해제되지 않아 자동주문을 실행하지 않습니다.")
    else:
        st.success("자동매매 설정이 ON입니다.")
        st.warning("Streamlit 화면만으로는 24시간 상시 실행을 보장하지 않습니다. "
                   "`worker.py`를 항상 켜진 서버에서 실행해야 24시간 감시가 됩니다.")
else:
    st.caption("자동매매는 OFF입니다. 분석/조회만 수행합니다.")

st.divider()
st.subheader("🧾 최근 주문 로그")
log = load_trade_log()
if log.empty:
    st.caption("아직 저장된 주문 로그가 없습니다.")
else:
    st.dataframe(log.tail(100), use_container_width=True, hide_index=True)

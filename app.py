import streamlit as st
import pandas as pd

from trader_core import (
    Settings, KISClient, discover_domestic_candidates,
    score_ticker, split_budget, is_market_open, market_force_exit_time,
    load_trade_log
)
from auto_engine import (
    AutoConfig, run_domestic_cycle, run_overseas_cycle,
    load_state, reset_today_state
)
from trend_strategy import score_leader_trend
from ai_judge import analyze_market_with_ai, merge_ai_filter

st.set_page_config(page_title="쏭 자동매매", page_icon="🤖", layout="wide")
st.title("🤖 쏭 국내·미국 자동매매")
st.caption("모의투자와 실전투자가 같은 전략 엔진을 사용합니다. 실전은 잠금 해제 전까지 주문되지 않습니다.")

settings = Settings.from_env()

with st.sidebar:
    st.header("⚙️ 운용 설정")

    mode = st.radio("운용 모드", ["모의투자", "실전투자"], index=0)
    market = st.radio("시장", ["국내", "미국"], horizontal=True)
    auto_on = st.toggle("🤖 자동매매 ON", value=False)

    strategy_mode = st.selectbox(
        "매매 전략",
        ["기존 기술지표 모드", "대장주 추세매매 모드"],
        index=1,
        help="RSI/볼린저 기반 기존 전략과 VWAP·돌파·눌림 중심 추세전략을 선택합니다."
    )

    ai_filter_on = st.toggle(
        "🧠 AI 시장판단 필터",
        value=False,
        help="최신 뉴스/이슈를 AI가 확인해 규칙 기반 후보에 추가 위험 필터를 적용합니다."
    )
    ai_score_min = st.slider("AI 최소점수", 0, 100, 60, 5)
    ai_conf_min = st.slider("AI 최소확신도", 0, 100, 55, 5)

    st.markdown("### 💰 자금 설정")
    if market == "국내":
        budget = st.number_input(
            "1일 최대 신규매수 금액(원)",
            min_value=10000,
            value=300000,
            step=10000,
        )
        per_stock = st.number_input(
            "종목당 최대 금액(원)",
            min_value=10000,
            value=100000,
            step=10000,
        )
        us_daily_budget = 0.0
        us_per_stock_budget = 0.0
    else:
        st.caption("미국도 모의·실전에서 같은 달러 예산과 분할매수 규칙을 사용합니다.")
        us_daily_budget = st.number_input(
            "미국 1일 최대 신규매수 금액(USD)",
            min_value=0.0,
            value=1500.0,
            step=100.0,
        )
        us_per_stock_budget = st.number_input(
            "미국 종목당 최대 금액(USD)",
            min_value=0.0,
            value=600.0,
            step=50.0,
        )
        budget = 300000
        per_stock = 100000

    max_positions = st.number_input(
        "최대 동시 보유 종목",
        min_value=1,
        max_value=10,
        value=3,
    )

    st.markdown("**분할매수 비율**")
    b1 = st.number_input("1차 %", 0, 100, 40)
    b2 = st.number_input("2차 %", 0, 100, 30)
    b3 = st.number_input("3차 %", 0, 100, 30)

    stop_loss = st.number_input(
        "손절 %",
        min_value=0.1,
        max_value=20.0,
        value=3.0,
        step=0.1,
    )
    take1 = st.number_input(
        "1차 익절 %",
        min_value=0.1,
        max_value=50.0,
        value=3.0,
        step=0.1,
    )
    take2 = st.number_input(
        "2차 익절 %",
        min_value=0.1,
        max_value=100.0,
        value=5.0,
        step=0.1,
    )

    st.markdown("**실전 잠금**")
    live_phrase = st.text_input(
        "실전 확인문구",
        type="password",
        placeholder="실전 운용 시에만 입력",
    )

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

cfg = AutoConfig(
    daily_budget=int(budget),
    per_stock_budget=int(per_stock),
    max_positions=int(max_positions),
    buy1_pct=int(b1),
    buy2_pct=int(b2),
    buy3_pct=int(b3),
    stop_loss_pct=float(stop_loss),
    take1_pct=float(take1),
    take2_pct=float(take2),
    min_combined_score=65.0 if strategy_mode == "기존 기술지표 모드" else 68.0,
    require_green_signal=True,
    us_daily_budget_usd=float(us_daily_budget),
    us_per_stock_budget_usd=float(us_per_stock_budget),
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("운용 모드", "모의" if env == "demo" else "실전")
c2.metric("시장", market)
c3.metric("자동매매", "ON" if auto_on else "OFF")
c4.metric(
    "실전 잠금",
    "해당 없음" if env == "demo" else ("해제" if live_unlocked else "잠김"),
)

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
    "추가 관심종목 입력 (선택사항)",
    placeholder="국내: 005930,000660 / 미국: AAPL,NVDA,TSLA",
)
manual_symbols = [x.strip().upper() for x in manual_text.split(",") if x.strip()]

if market == "국내":
    if st.button("📊 거래량·거래대금 기반 국내 후보 찾기"):
        try:
            candidates = discover_domestic_candidates(client, top_n=20)
            st.session_state["candidates_kr"] = candidates
            st.session_state.pop("leader_df_kr", None)
            st.session_state.pop("ai_result_국내", None)
            st.session_state.pop("ai_filtered_국내", None)
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
                    tech100 = max(0.0, min(100.0, ((net + 6) / 12) * 100))

                    try:
                        from trader_core import _download_yf
                        intraday_df = _download_yf(code, "국내")
                        trend = score_leader_trend(intraday_df)
                    except Exception:
                        trend = None

                    trend_score = float((trend or {}).get("추세점수", 0))

                    if strategy_mode == "대장주 추세매매 모드":
                        combined = lead * 0.45 + trend_score * 0.55
                        final_signal = (trend or {}).get("추세판정", "⚪ 추세약함")
                    else:
                        combined = lead * 0.65 + tech100 * 0.35
                        final_signal = tech.get("종합신호")

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
                        "추세점수": round(trend_score, 1),
                        "VWAP": (trend or {}).get("VWAP"),
                        "VWAP괴리율": (trend or {}).get("VWAP괴리율"),
                        "당일고가거리": (trend or {}).get("당일고가거리"),
                        "돌파": (trend or {}).get("돌파"),
                        "눌림재상승": (trend or {}).get("눌림재상승"),
                        "종합점수": round(combined, 1),
                        "판정": final_signal,
                        "진입근거": (
                            (trend or {}).get("추세이유", "")
                            if strategy_mode == "대장주 추세매매 모드"
                            else tech.get("종합신호")
                        ),
                    })

                progress.progress((i + 1) / total, text=f"{i+1}/{total} 분석 중")

            progress.empty()

            if rows:
                leader_df = pd.DataFrame(rows).sort_values(
                    ["종합점수", "주도주점수", "기술순점수"],
                    ascending=[False, False, False],
                ).head(5).reset_index(drop=True)

                labels = ["👑 1위", "🥈 2위", "🥉 3위", "4위", "5위"]
                leader_df.insert(0, "순위", labels[:len(leader_df)])
                st.session_state["leader_df_kr"] = leader_df
                st.session_state.pop("ai_result_국내", None)
                st.session_state.pop("ai_filtered_국내", None)
            else:
                st.warning("5분봉 기술분석 가능한 후보가 없었습니다.")

        leader_df = st.session_state.get("leader_df_kr", pd.DataFrame())
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
    st.info("미국은 관심종목 + 기본 모멘텀 유니버스를 기술점수로 분석합니다.")
    default_us = [
        "AAPL", "MSFT", "NVDA", "AMZN", "META",
        "TSLA", "AMD", "GOOGL", "AVGO", "NFLX",
    ]
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
            ["순점수", "거래량배수"],
            ascending=[False, False],
        )
        st.session_state["score_df"] = score_df

        if market == "미국":
            us_top = score_df.head(5).copy().reset_index(drop=True)
            tech100 = ((us_top["순점수"].clip(-6, 6) + 6) / 12 * 100).astype(float)
            vol_bonus = (
                us_top["거래량배수"].clip(0, 2.0) / 2.0 * 10
            ).astype(float)
            us_top["종합점수"] = (tech100 * 0.9 + vol_bonus).round(1)

            labels = ["⭐ 1위", "⭐ 2위", "⭐ 3위", "4위", "5위"]
            us_top.insert(0, "순위", labels[:len(us_top)])
            us_top["종목코드"] = us_top["종목"].astype(str)
            us_top["종목명"] = us_top["종목"].astype(str)
            us_top["판정"] = us_top["종합신호"]
            us_top["진입근거"] = (
                "RSI " + us_top["RSI"].astype(str)
                + " / 거래량배수 " + us_top["거래량배수"].astype(str)
                + " / 매수점수 " + us_top["매수점수"].astype(str)
                + " / 매도점수 " + us_top["매도점수"].astype(str)
            )

            st.session_state["leader_df_us"] = us_top
            st.session_state.pop("ai_result_미국", None)
            st.session_state.pop("ai_filtered_미국", None)

score_df = st.session_state.get("score_df", pd.DataFrame())
if not score_df.empty:
    st.dataframe(score_df, use_container_width=True, hide_index=True)

    if market == "미국":
        st.subheader("🇺🇸 미국 기술·모멘텀 TOP 5")
        us_show = st.session_state.get("leader_df_us", pd.DataFrame())
        if not us_show.empty:
            cols = [
                c for c in [
                    "순위", "종목", "현재가", "RSI", "거래량배수",
                    "매수점수", "매도점수", "순점수", "종합점수", "판정",
                ]
                if c in us_show.columns
            ]
            st.dataframe(
                us_show[cols],
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.subheader("⭐ 기술점수 TOP 5")
        st.dataframe(
            score_df.head(5),
            use_container_width=True,
            hide_index=True,
        )

st.divider()
st.subheader("🧪 주문 계산 미리보기")

parts = split_budget(int(per_stock), [int(b1), int(b2), int(b3)])

if market == "국내":
    st.write(
        f"종목당 {int(per_stock):,}원 기준 → "
        f"1차 {parts[0]:,}원 / 2차 {parts[1]:,}원 / 3차 {parts[2]:,}원"
    )
else:
    total_pct = max(1, int(b1) + int(b2) + int(b3))
    usd1 = float(us_per_stock_budget) * int(b1) / total_pct
    usd2 = float(us_per_stock_budget) * int(b2) / total_pct
    usd3 = float(us_per_stock_budget) * int(b3) / total_pct

    st.write(
        f"미국 종목당 ${float(us_per_stock_budget):,.0f} 기준 → "
        f"1차 약 ${usd1:,.0f} / 2차 약 ${usd2:,.0f} / 3차 약 ${usd3:,.0f}"
    )
    st.caption(
        "실제 주문수량은 주문 직전 한국투자 현재가를 다시 조회한 뒤 "
        "각 단계 예산으로 살 수 있는 정수 주식 수량으로 계산합니다."
    )

st.write(
    f"손절 -{float(stop_loss):.1f}% / "
    f"1차 익절 +{float(take1):.1f}% / "
    f"2차 익절 +{float(take2):.1f}%"
)
st.write("당일매매 규칙: 해당 시장 마감 전 남은 추적 포지션 전량 청산")

open_now = is_market_open("KR" if market == "국내" else "US")
st.info(
    f"현재 시장 상태: {'장중' if open_now else '장외'} · "
    f"강제청산 기준시각: "
    f"{market_force_exit_time('KR' if market == '국내' else 'US')}"
)

st.divider()
st.subheader("🧠 AI 시장판단 필터")
st.caption(
    "AI는 최신 공개 뉴스·이슈를 확인해 위험도를 평가하는 보조 필터입니다. "
    "AI 단독으로 주문을 만들지 않으며 손절·예산·보유한도 같은 숫자 규칙을 우회할 수 없습니다."
)

leader_for_ai = (
    st.session_state.get("leader_df_kr", pd.DataFrame())
    if market == "국내"
    else st.session_state.get("leader_df_us", pd.DataFrame())
)

ai_result_key = f"ai_result_{market}"
ai_filtered_key = f"ai_filtered_{market}"

if ai_filter_on:
    if leader_for_ai.empty:
        st.warning(
            "먼저 👑 대장주 TOP5를 계산해주세요."
            if market == "국내"
            else "먼저 미국 후보 기술점수를 계산해주세요."
        )
    else:
        if st.button("🧠 최신 뉴스·이슈 AI 판단", key=f"ai_button_{market}"):
            with st.spinner("AI가 최신 공개 뉴스와 후보 종목 이슈를 확인하고 있어요..."):
                try:
                    result = analyze_market_with_ai(
                        leader_for_ai,
                        st.secrets,
                        strategy_name=strategy_mode,
                        market=market,
                    )
                except Exception as e:
                    result = {
                        "ok": False,
                        "error": f"{type(e).__name__}: {e}",
                    }

                st.session_state[ai_result_key] = result

        ai_result = st.session_state.get(ai_result_key)

        if ai_result is None:
            st.caption("위 버튼을 누르면 AI 시장요약과 종목별 판정이 표시됩니다.")
        elif not ai_result.get("ok"):
            st.error(
                f"AI 판단 실패: "
                f"{ai_result.get('error', '알 수 없는 오류')}"
            )
        else:
            summary = str(ai_result.get("market_summary", "")).strip()

            if summary:
                st.info(summary)
            else:
                st.warning("AI 응답은 성공했지만 시장 요약이 비어 있습니다.")

            try:
                ai_filtered = merge_ai_filter(
                    leader_for_ai,
                    ai_result,
                    min_ai_score=int(ai_score_min),
                    min_confidence=int(ai_conf_min),
                )
                st.session_state[ai_filtered_key] = ai_filtered

                show_cols = [
                    c for c in [
                        "순위", "종목코드", "종목명", "종합점수", "판정",
                        "AI판정", "AI점수", "AI확신도", "AI통과", "AI테마",
                        "뉴스품질", "AI이유", "AI위험",
                    ]
                    if c in ai_filtered.columns
                ]

                if show_cols:
                    st.dataframe(
                        ai_filtered[show_cols],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.warning("AI 결과는 받았지만 표시 가능한 열을 찾지 못했습니다.")

                passed = (
                    int(ai_filtered["AI통과"].sum())
                    if "AI통과" in ai_filtered.columns
                    else 0
                )
                st.success(f"AI 추가 필터 통과: {passed}개")

            except Exception as e:
                st.error(
                    f"AI 결과 병합 실패: "
                    f"{type(e).__name__}: {e}"
                )
else:
    st.caption("AI 필터 OFF — 기존 숫자 규칙만 사용합니다.")

st.divider()
st.subheader("🤖 자동매매 엔진 v4")

leader_for_trade = (
    st.session_state.get("leader_df_kr", pd.DataFrame())
    if market == "국내"
    else st.session_state.get("leader_df_us", pd.DataFrame())
)

if ai_filter_on:
    ai_filtered_for_trade = st.session_state.get(
        f"ai_filtered_{market}",
        pd.DataFrame(),
    )

    if (
        ai_filtered_for_trade.empty
        or "AI통과" not in ai_filtered_for_trade.columns
    ):
        leader_for_trade = pd.DataFrame()
    else:
        leader_for_trade = ai_filtered_for_trade[
            ai_filtered_for_trade["AI통과"] == True
        ].copy()

if strategy_mode == "대장주 추세매매 모드":
    st.write(
        f"진입 기준: TOP5 + 🟢 매수/추세매수 후보 + "
        f"종합점수 {cfg.min_combined_score:.0f}점 이상"
    )
else:
    st.write(
        f"진입 기준: TOP5 + 🟢 매수 후보 + "
        f"종합점수 {cfg.min_combined_score:.0f}점 이상"
    )

st.write(
    f"추가매수: 평균단가 대비 +{cfg.add2_trigger_pct:.1f}% → 2차, "
    f"+{cfg.add3_trigger_pct:.1f}% → 3차"
)

if market == "국내":
    st.write(
        f"신규매수 마감 {cfg.last_entry_time} KST / "
        f"강제청산 {cfg.force_exit_time} KST"
    )
else:
    st.write(
        f"미국 신규매수 마감 {cfg.us_last_entry_time} ET / "
        f"강제청산 {cfg.us_force_exit_time} ET"
    )
    st.success(
        "🇺🇸 미국도 모의투자와 실전투자가 같은 자동매매 엔진을 사용합니다. "
        "전환 시 전략 코드를 다시 바꾸지 않습니다."
    )

if leader_for_trade.empty:
    if ai_filter_on:
        st.warning(
            "후보 계산 후 AI 필터까지 통과한 종목이 있어야 자동매매 후보가 생깁니다."
        )
    else:
        st.warning(
            "먼저 후보 TOP5를 계산해야 자동매매 후보가 생깁니다."
        )

if market == "국내":
    current_state = load_state()
    st.caption(
        f"오늘 국내 자동매매 상태: "
        f"추적 종목 {len(current_state.get('positions', {}))}개 · "
        f"누적 신규매수 약 "
        f"{int(current_state.get('daily_buy_amount', 0)):,}원 · "
        f"주문 {int(current_state.get('daily_orders', 0))}회"
    )

dry_run = st.toggle(
    "🧪 주문 없이 자동매매 판단만 보기",
    value=True,
    help="켜져 있으면 모의/실전 주문을 전송하지 않고 판단 결과만 기록합니다.",
)

if env == "real":
    st.error("실전 모드는 모의운용 검증 후에만 사용하세요.")
    real_confirm = st.checkbox(
        "실전 주문 위험을 이해했고 실제 주문 전송을 허용합니다."
    )
else:
    real_confirm = True

can_execute = auto_on and live_unlocked and real_confirm

if not auto_on:
    st.caption("자동매매는 OFF입니다. 분석/조회만 수행합니다.")
elif env == "real" and not live_unlocked:
    st.error("실전 잠금이 해제되지 않아 주문을 실행하지 않습니다.")
else:
    st.success("자동매매 설정 ON")

run_label = (
    "▶️ 국내 자동매매 1회 사이클 실행"
    if market == "국내"
    else "▶️ 미국 자동매매 1회 사이클 실행"
)

if st.button(run_label, type="primary"):
    if not auto_on:
        st.warning("왼쪽 설정에서 자동매매를 ON으로 먼저 바꿔주세요.")
    elif leader_for_trade.empty:
        st.warning(
            "자동매매 후보가 없습니다. 먼저 후보 TOP5를 계산해주세요."
        )
    elif env == "real" and not live_unlocked:
        st.error("실전 잠금 상태입니다.")
    else:
        try:
            if market == "국내":
                cycle = run_domestic_cycle(
                    client=client,
                    leader_df=leader_for_trade,
                    config=cfg,
                    execute_orders=(can_execute and not dry_run),
                )
                cycle_key = "last_cycle_kr"
            else:
                cycle = run_overseas_cycle(
                    client=client,
                    leader_df=leader_for_trade,
                    config=cfg,
                    execute_orders=(can_execute and not dry_run),
                )
                cycle_key = "last_cycle_us"

            st.session_state[cycle_key] = cycle

            if can_execute and not dry_run:
                st.success(
                    f"{'🇰🇷 국내' if market == '국내' else '🇺🇸 미국'} "
                    f"{'모의' if env == 'demo' else '실전'} "
                    "자동매매 1회 사이클 실행 완료"
                )
            else:
                st.info("주문 전송 없이 판단 사이클만 실행했습니다.")

        except Exception as e:
            st.error(f"자동매매 사이클 오류: {type(e).__name__}: {e}")

cycle_key = "last_cycle_kr" if market == "국내" else "last_cycle_us"
cycle = st.session_state.get(cycle_key)

if cycle:
    if cycle.get("message"):
        st.info(cycle["message"])

    if cycle.get("balance_warning"):
        st.warning(f"잔고조회 경고: {cycle['balance_warning']}")

    actions = cycle.get("actions", [])
    if actions:
        st.dataframe(
            pd.DataFrame(actions),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("이번 사이클에서 주문/가상주문 동작이 없었습니다.")

    state_after = cycle.get("state", {})
    if state_after:
        if market == "국내":
            st.caption(
                f"사이클 후 상태: 추적 종목 "
                f"{len(state_after.get('positions', {}))}개 · "
                f"주문 {int(state_after.get('daily_orders', 0))}회"
            )
        else:
            st.caption(
                f"사이클 후 미국 상태: 추적 종목 "
                f"{len(state_after.get('positions', {}))}개 · "
                f"누적 신규매수 약 "
                f"${float(state_after.get('daily_buy_amount_usd', 0)):.2f} · "
                f"주문 {int(state_after.get('daily_orders', 0))}회"
            )

if market == "국내":
    if st.button("♻️ 오늘 국내 자동매매 상태 초기화"):
        reset_today_state()
        st.session_state.pop("last_cycle_kr", None)
        st.success("오늘 국내 자동매매 상태를 초기화했습니다.")

st.warning(
    "Streamlit 화면은 24시간 워커가 아닙니다. "
    "현재 버튼은 한 번의 매매 판단 사이클만 실행합니다. "
    "상시 자동운용은 worker.py를 별도 상시 서버에서 실행해야 합니다."
)

st.divider()
st.subheader("📒 자동 매매일지 미리보기")

journal_df = (
    st.session_state.get("leader_df_kr", pd.DataFrame())
    if market == "국내"
    else st.session_state.get("leader_df_us", pd.DataFrame())
)

if journal_df.empty:
    st.caption("후보 TOP5를 계산하면 주요 진입근거가 여기에 표시됩니다.")
else:
    cols = [
        c for c in [
            "순위", "종목코드", "종목명", "주도주점수", "추세점수",
            "VWAP", "VWAP괴리율", "당일고가거리", "돌파", "눌림재상승",
            "종합점수", "판정", "진입근거",
        ]
        if c in journal_df.columns
    ]
    st.dataframe(
        journal_df[cols],
        use_container_width=True,
        hide_index=True,
    )

st.divider()
st.subheader("🧾 최근 주문 로그")

log = load_trade_log()
if log.empty:
    st.caption("아직 저장된 주문 로그가 없습니다.")
else:
    st.dataframe(
        log.tail(100),
        use_container_width=True,
        hide_index=True,
    )

import streamlit as st
import pandas as pd

from trader_core import (
    Settings,
    KISClient,
    discover_domestic_candidates,
    score_ticker,
    split_budget,
    is_market_open,
    market_force_exit_time,
    load_trade_log,
)

from auto_engine import (
    AutoConfig,
    run_domestic_cycle,
    run_overseas_cycle,
    load_state,
    reset_today_state,
)

from trend_strategy import score_leader_trend
from ai_judge import analyze_market_with_ai, merge_ai_filter


st.set_page_config(
    page_title="쏭 자동매매",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 쏭 국내·미국 자동매매")
st.caption(
    "후보 탐색 → 대장주 선정 → 기술분석 → AI 필터 → "
    "자동매매 판단/주문을 한 번에 실행합니다."
)

settings = Settings.from_env()


with st.sidebar:
    st.header("⚙️ 운용 설정")

    mode = st.radio(
        "운용 모드",
        ["모의투자", "실전투자"],
        index=0,
    )

    market = st.radio(
        "시장",
        ["국내", "미국"],
        horizontal=True,
    )

    strategy_mode = st.selectbox(
        "매매 전략",
        [
            "기존 기술지표 모드",
            "대장주 추세매매 모드",
        ],
        index=1,
    )

    st.markdown("### 🧠 AI 보조 필터")

    ai_filter_on = st.toggle(
        "AI 시장판단 사용",
        value=False,
    )

    ai_score_min = st.slider(
        "AI 최소점수",
        0, 100, 60, 5,
    )

    ai_conf_min = st.slider(
        "AI 최소확신도",
        0, 100, 55, 5,
    )

    st.markdown("### 💰 자금 설정")

    if market == "국내":
        budget = st.number_input(
            "1일 최대 신규매수 금액(원)",
            min_value=10_000,
            value=10_000_000,
            step=100_000,
        )

        per_stock = st.number_input(
            "종목당 최대 금액(원)",
            min_value=10_000,
            value=10_000_000,
            step=100_000,
        )

        us_daily_budget = 0.0
        us_per_stock_budget = 0.0

    else:
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

        budget = 10_000_000
        per_stock = 10_000_000

    max_positions = st.number_input(
        "최대 동시 보유 종목",
        min_value=1,
        max_value=10,
        value=3,
    )

    st.markdown("### 📦 분할매수")

    b1 = st.number_input("1차 %", 0, 100, 50)
    b2 = st.number_input("2차 %", 0, 100, 30)
    b3 = st.number_input("3차 %", 0, 100, 20)

    st.markdown("### 🛡️ 손절·익절")

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

    st.markdown("### 🔐 실전 잠금")

    live_phrase = st.text_input(
        "실전 확인문구",
        type="password",
        placeholder="실전투자일 때만 입력",
    )


env = "demo" if mode == "모의투자" else "real"

live_unlocked = (
    env == "demo"
    or (
        live_phrase == settings.live_unlock_phrase
        and settings.allow_live
    )
)

if env == "real":
    if live_unlocked:
        st.error("🔴 실전투자 잠금이 해제되었습니다.")
    else:
        st.warning("🔒 실전투자는 잠겨 있습니다.")

client = KISClient(
    settings=settings,
    env=env,
)


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
    min_combined_score=65.0,
    require_green_signal=True,
)

cfg.demo_relaxed_entry_enabled = True
cfg.demo_min_combined_score = 40.0

cfg.leader_exception_enabled = True
cfg.leader_exception_min_lead_score = 75.0
cfg.leader_exception_min_combined_score = 60.0

cfg.us_daily_budget_usd = float(us_daily_budget)
cfg.us_per_stock_budget_usd = float(us_per_stock_budget)
cfg.us_last_entry_time = "15:30"
cfg.us_force_exit_time = "15:50"


def build_domestic_leader_df(candidates: pd.DataFrame):
    rows = []

    if candidates is None or candidates.empty:
        return pd.DataFrame()

    scan = candidates.head(12).copy()
    total = max(len(scan), 1)

    progress = st.progress(
        0,
        text="📊 대장주 후보 기술분석 시작...",
    )

    for i, (_, r) in enumerate(scan.iterrows()):
        code = str(r["종목코드"]).zfill(6)

        try:
            tech = score_ticker(code, "국내")
        except Exception:
            tech = None

        if not tech:
            progress.progress((i + 1) / total, text=f"{i + 1}/{total} 분석 중")
            continue

        lead = float(r.get("주도주점수", 0))
        net = int(tech.get("순점수", 0))

        tech100 = max(
            0.0,
            min(
                100.0,
                ((net + 6) / 12) * 100,
            ),
        )

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

        progress.progress(
            (i + 1) / total,
            text=f"{i + 1}/{total} 분석 중",
        )

    progress.empty()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values(
        ["종합점수", "주도주점수", "기술순점수"],
        ascending=[False, False, False],
    ).head(5).reset_index(drop=True)

    labels = ["👑 1위", "🥈 2위", "🥉 3위", "4위", "5위"]
    df.insert(0, "순위", labels[:len(df)])
    return df


def build_us_leader_df(symbols):
    score_rows = []

    progress = st.progress(
        0,
        text="🇺🇸 미국 후보 분석 시작...",
    )

    total = max(len(symbols), 1)

    for i, symbol in enumerate(symbols):
        try:
            row = score_ticker(symbol, market="미국")
            if row:
                score_rows.append(row)
        except Exception:
            pass

        progress.progress(
            (i + 1) / total,
            text=f"{i + 1}/{total} 분석 중",
        )

    progress.empty()

    if not score_rows:
        return pd.DataFrame(), pd.DataFrame()

    score_df = pd.DataFrame(score_rows).sort_values(
        ["순점수", "거래량배수"],
        ascending=[False, False],
    )

    us_top = score_df.head(5).copy().reset_index(drop=True)

    tech100 = ((us_top["순점수"].clip(-6, 6) + 6) / 12 * 100).astype(float)
    vol_bonus = (us_top["거래량배수"].clip(0, 2.0) / 2.0 * 10).astype(float)

    us_top["종합점수"] = (tech100 * 0.9 + vol_bonus).round(1)

    labels = ["⭐ 1위", "⭐ 2위", "⭐ 3위", "4위", "5위"]
    us_top.insert(0, "순위", labels[:len(us_top)])

    us_top["종목코드"] = us_top["종목"].astype(str)
    us_top["종목명"] = us_top["종목"].astype(str)
    us_top["판정"] = us_top["종합신호"]

    return score_df, us_top


c1, c2, c3, c4 = st.columns(4)

c1.metric("운용 모드", "모의" if env == "demo" else "실전")
c2.metric("시장", market)
c3.metric(
    "전략",
    "대장주" if strategy_mode == "대장주 추세매매 모드" else "기술지표",
)
c4.metric("AI 필터", "ON" if ai_filter_on else "OFF")


st.divider()
st.subheader("🔑 API 상태")

try:
    client.get_token()
    st.success("✅ 한국투자 API 연결 정상")
except Exception as e:
    st.error(f"API 연결 실패: {e}")


st.divider()
st.subheader("🧪 주문 설정")

parts = split_budget(
    int(per_stock),
    [int(b1), int(b2), int(b3)],
)

if market == "국내":
    st.write(
        f"하루 최대 {int(budget):,}원 · "
        f"종목당 {int(per_stock):,}원 → "
        f"1차 {parts[0]:,}원 / "
        f"2차 {parts[1]:,}원 / "
        f"3차 {parts[2]:,}원"
    )
else:
    total_pct = max(1, int(b1) + int(b2) + int(b3))

    usd1 = float(us_per_stock_budget) * int(b1) / total_pct
    usd2 = float(us_per_stock_budget) * int(b2) / total_pct
    usd3 = float(us_per_stock_budget) * int(b3) / total_pct

    st.write(
        f"미국 종목당 ${float(us_per_stock_budget):,.0f} → "
        f"1차 약 ${usd1:,.0f} / "
        f"2차 약 ${usd2:,.0f} / "
        f"3차 약 ${usd3:,.0f}"
    )

st.write(
    f"손절 -{float(stop_loss):.1f}% / "
    f"1차 익절 +{float(take1):.1f}% / "
    f"2차 익절 +{float(take2):.1f}%"
)

if market == "국내" and env == "demo":
    st.info(
        "🧪 모의투자 완화 진입 ON: TOP5 중 종합점수 40 이상이면 "
        "녹색 신호가 아니어도 1차 진입을 허용합니다."
    )

open_now = is_market_open("KR" if market == "국내" else "US")

st.info(
    f"현재 시장 상태: {'장중' if open_now else '장외'} · "
    f"강제청산 기준시각: "
    f"{market_force_exit_time('KR' if market == '국내' else 'US')}"
)


dry_run = st.toggle(
    "🧪 주문 없이 판단만 보기",
    value=True,
    help=(
        "ON이면 주문을 보내지 않습니다. "
        "한국투자 모의계좌로 실제 모의주문을 보내려면 OFF로 바꾸세요."
    ),
)


if env == "real":
    st.error("⚠️ 현재 실전투자 모드입니다.")
    real_confirm = st.checkbox(
        "실제 주문 위험을 이해했고 주문 전송을 허용합니다."
    )
else:
    real_confirm = True

can_execute = live_unlocked and real_confirm


st.divider()
st.header("🚀 원터치 자동매매")

st.write(
    "후보 탐색 → 대장주 TOP5 → 기술분석 → "
    "AI 필터(선택) → 최종 후보 → 자동매매"
)

one_click_label = (
    "🚀 국내 자동매매 한 번에 실행"
    if market == "국내"
    else "🚀 미국 자동매매 한 번에 실행"
)

if st.button(
    one_click_label,
    type="primary",
    use_container_width=True,
):
    if env == "real" and not live_unlocked:
        st.error("🔒 실전투자 잠금이 해제되지 않았습니다.")
        st.stop()

    if env == "real" and not real_confirm:
        st.error("실전 주문 허용 체크가 필요합니다.")
        st.stop()

    try:
        if market == "국내":
            st.info("① 거래량·거래대금 기반 후보를 찾습니다.")

            candidates = discover_domestic_candidates(client, top_n=20)

            if candidates is None or candidates.empty:
                st.warning("오늘 조건에 맞는 국내 후보가 없습니다.")
                st.stop()

            st.session_state["candidates_kr"] = candidates
            st.success(f"✅ 1단계 완료: {len(candidates)}개 후보 발견")

            st.info("② 대장주 TOP5와 기술점수를 계산합니다.")

            leader_df = build_domestic_leader_df(candidates)

            if leader_df.empty:
                st.warning("기술분석 가능한 대장주 후보가 없습니다.")
                st.stop()

            st.session_state["leader_df_kr"] = leader_df
            st.success("✅ 2단계 완료: 대장주 TOP5 선정")

            leader_for_trade = leader_df.copy()

            if ai_filter_on:
                st.info("③ 최신 뉴스·이슈 AI 필터를 실행합니다.")

                ai_result = analyze_market_with_ai(
                    leader_df,
                    st.secrets,
                    strategy_name=strategy_mode,
                    market=market,
                )

                st.session_state["ai_result_국내"] = ai_result

                if not ai_result.get("ok"):
                    st.error(
                        "AI 판단에 실패했습니다. "
                        f"{ai_result.get('error', '')}"
                    )
                    st.stop()

                ai_filtered = merge_ai_filter(
                    leader_df,
                    ai_result,
                    min_ai_score=int(ai_score_min),
                    min_confidence=int(ai_conf_min),
                )

                st.session_state["ai_filtered_국내"] = ai_filtered

                if "AI통과" not in ai_filtered.columns:
                    st.warning("AI 통과 결과를 확인할 수 없습니다.")
                    st.stop()

                leader_for_trade = ai_filtered[
                    ai_filtered["AI통과"] == True
                ].copy()

                passed = len(leader_for_trade)
                st.success(f"✅ 3단계 완료: AI 필터 {passed}개 통과")

                if passed == 0:
                    st.warning("오늘은 AI 필터를 통과한 신규매수 후보가 없습니다.")

            else:
                st.info("③ AI 필터 OFF → 숫자·기술 규칙으로 진행합니다.")

            st.info("④ 자동매매 엔진을 실행합니다.")

            cycle = run_domestic_cycle(
                client=client,
                leader_df=leader_for_trade,
                config=cfg,
                execute_orders=(can_execute and not dry_run),
            )

            st.session_state["last_cycle_kr"] = cycle

            if dry_run:
                st.info("🧪 주문 없이 자동매매 판단만 완료했습니다.")
            else:
                st.success(
                    "✅ 한국투자 모의계좌 자동매매 사이클 완료"
                    if env == "demo"
                    else "🔴 실전 자동매매 사이클 완료"
                )

        else:
            default_us = [
                "AAPL", "MSFT", "NVDA", "AMZN", "META",
                "TSLA", "AMD", "GOOGL", "AVGO", "NFLX",
            ]

            score_df, leader_df = build_us_leader_df(default_us)

            if leader_df.empty:
                st.warning("미국 기술분석 가능한 후보가 없습니다.")
                st.stop()

            st.session_state["score_df"] = score_df
            st.session_state["leader_df_us"] = leader_df

            leader_for_trade = leader_df.copy()

            if ai_filter_on:
                ai_result = analyze_market_with_ai(
                    leader_df,
                    st.secrets,
                    strategy_name=strategy_mode,
                    market=market,
                )

                st.session_state["ai_result_미국"] = ai_result

                if not ai_result.get("ok"):
                    st.error(
                        f"AI 판단 실패: {ai_result.get('error', '')}"
                    )
                    st.stop()

                ai_filtered = merge_ai_filter(
                    leader_df,
                    ai_result,
                    min_ai_score=int(ai_score_min),
                    min_confidence=int(ai_conf_min),
                )

                st.session_state["ai_filtered_미국"] = ai_filtered

                if "AI통과" not in ai_filtered.columns:
                    st.warning("AI 통과 결과가 없습니다.")
                    st.stop()

                leader_for_trade = ai_filtered[
                    ai_filtered["AI통과"] == True
                ].copy()

            cycle = run_overseas_cycle(
                client=client,
                leader_df=leader_for_trade,
                config=cfg,
                execute_orders=(can_execute and not dry_run),
            )

            st.session_state["last_cycle_us"] = cycle

            if dry_run:
                st.info("🧪 미국 주문 없이 판단만 완료했습니다.")
            else:
                st.success(
                    "✅ 미국 모의 자동매매 완료"
                    if env == "demo"
                    else "🔴 미국 실전 자동매매 완료"
                )

    except Exception as e:
        st.error(
            f"원터치 자동매매 오류: "
            f"{type(e).__name__}: {e}"
        )


st.divider()

if market == "국내":
    st.subheader("👑 오늘의 대장주 후보 TOP 5")
    leader_show = st.session_state.get("leader_df_kr", pd.DataFrame())
else:
    st.subheader("🇺🇸 미국 자동매매 후보 TOP 5")
    leader_show = st.session_state.get("leader_df_us", pd.DataFrame())

if leader_show.empty:
    st.caption("🚀 원터치 자동매매 버튼을 누르면 자동으로 계산됩니다.")
else:
    st.dataframe(
        leader_show,
        use_container_width=True,
        hide_index=True,
    )

    top = leader_show.iloc[0]
    st.info(
        f"현재 1순위 후보: "
        f"{top.get('종목명', '')} "
        f"({top.get('종목코드', '')}) · "
        f"종합점수 {top.get('종합점수', '')} · "
        f"{top.get('판정', '')}"
    )


if ai_filter_on:
    st.divider()
    st.subheader("🧠 AI 시장판단 결과")

    ai_result = st.session_state.get(f"ai_result_{market}")
    ai_filtered = st.session_state.get(
        f"ai_filtered_{market}",
        pd.DataFrame(),
    )

    if ai_result:
        summary = str(
            ai_result.get("market_summary", "")
        ).strip()

        if summary:
            st.info(summary)

    if not ai_filtered.empty:
        show_cols = [
            c
            for c in [
                "순위", "종목코드", "종목명", "종합점수", "판정",
                "AI판정", "AI점수", "AI확신도", "AI통과",
                "AI테마", "뉴스품질", "AI이유", "AI위험",
            ]
            if c in ai_filtered.columns
        ]

        if show_cols:
            st.dataframe(
                ai_filtered[show_cols],
                use_container_width=True,
                hide_index=True,
            )


st.divider()
st.subheader("🤖 자동매매 실행 결과")

cycle_key = "last_cycle_kr" if market == "국내" else "last_cycle_us"
cycle = st.session_state.get(cycle_key)

if not cycle:
    st.caption("아직 실행된 자동매매 사이클이 없습니다.")
else:
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
        st.caption("이번 사이클에는 매수·매도 동작이 없었습니다.")

    state_after = cycle.get("state", {})

    if state_after:
        if market == "국내":
            st.caption(
                f"추적 종목 {len(state_after.get('positions', {}))}개 · "
                f"누적 신규매수 약 "
                f"{int(state_after.get('daily_buy_amount', 0)):,}원 · "
                f"주문 {int(state_after.get('daily_orders', 0))}회"
            )
        else:
            st.caption(
                f"미국 추적 종목 "
                f"{len(state_after.get('positions', {}))}개 · "
                f"누적 신규매수 약 "
                f"${float(state_after.get('daily_buy_amount_usd', 0)):.2f} · "
                f"주문 {int(state_after.get('daily_orders', 0))}회"
            )


if market == "국내":
    current_state = load_state()

    st.info(
        f"📌 오늘 추적 종목 "
        f"{len(current_state.get('positions', {}))}개 · "
        f"누적 신규매수 약 "
        f"{int(current_state.get('daily_buy_amount', 0)):,}원 · "
        f"주문 {int(current_state.get('daily_orders', 0))}회"
    )

    if st.button("♻️ 오늘 국내 자동매매 상태 초기화"):
        reset_today_state()
        st.session_state.pop("last_cycle_kr", None)
        st.success("오늘 국내 자동매매 상태를 초기화했습니다.")
        st.rerun()


st.warning(
    "🚀 버튼은 1회 실행용입니다. "
    "화면을 닫아도 반복 실행하려면 Render의 worker.py가 계속 실행되어야 합니다."
)


st.divider()
st.subheader("📒 자동 매매일지 미리보기")

journal_df = (
    st.session_state.get("leader_df_kr", pd.DataFrame())
    if market == "국내"
    else st.session_state.get("leader_df_us", pd.DataFrame())
)

if journal_df.empty:
    st.caption("원터치 자동매매를 실행하면 진입근거가 표시됩니다.")
else:
    cols = [
        c
        for c in [
            "순위", "종목코드", "종목명", "주도주점수", "추세점수",
            "VWAP", "VWAP괴리율", "당일고가거리",
            "돌파", "눌림재상승", "종합점수", "판정", "진입근거",
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

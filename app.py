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
    load_state,
    reset_today_state,
)

from trend_strategy import score_leader_trend


st.set_page_config(
    page_title="쏭 자동매매",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 쏭 국내 모의 자동매매")
st.caption(
    "데이터 없는 종목은 자동 건너뛰고, "
    "매수하지 않은 이유를 종목별로 표시하는 안정화 버전입니다."
)

settings = Settings.from_env()

with st.sidebar:
    st.header("⚙️ 운용 설정")

    mode = st.radio(
        "운용 모드",
        ["모의투자", "실전투자"],
        index=0,
    )

    strategy_mode = st.selectbox(
        "매매 전략",
        [
            "기존 기술지표 모드",
            "대장주 추세매매 모드",
        ],
        index=1,
    )

    st.markdown("### 💰 자금 설정")

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

    max_positions = st.number_input(
        "최대 동시 보유 종목",
        min_value=1,
        max_value=10,
        value=3,
    )

    st.markdown("### 📦 분할매수")

    b1 = st.number_input(
        "1차 %",
        0,
        100,
        50,
    )

    b2 = st.number_input(
        "2차 %",
        0,
        100,
        30,
    )

    b3 = st.number_input(
        "3차 %",
        0,
        100,
        20,
    )

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

    live_phrase = st.text_input(
        "실전 확인문구",
        type="password",
    )


env = "demo" if mode == "모의투자" else "real"

live_unlocked = (
    env == "demo"
    or (
        live_phrase == settings.live_unlock_phrase
        and settings.allow_live
    )
)

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

cfg.last_entry_time = "14:50"
cfg.force_exit_time = "15:15"


def build_domestic_leader_df(
    candidates: pd.DataFrame
) -> pd.DataFrame:
    rows = []

    if candidates is None or candidates.empty:
        return pd.DataFrame()

    scan = candidates.head(12).copy()
    total = max(len(scan), 1)

    progress = st.progress(
        0,
        text="📊 대장주 후보 분석 시작...",
    )

    skipped = []

    for i, (_, r) in enumerate(scan.iterrows()):
        code = str(
            r.get("종목코드", "")
        ).zfill(6)

        name = str(
            r.get("종목명", "")
        )

        if not (
            len(code) == 6
            and code.isdigit()
        ):
            skipped.append({
                "종목코드": code,
                "종목명": name,
                "이유": "잘못된 종목코드",
            })
            continue

        try:
            tech = score_ticker(
                code,
                "국내",
            )
        except Exception as e:
            tech = None
            skipped.append({
                "종목코드": code,
                "종목명": name,
                "이유": (
                    f"기술분석 예외 "
                    f"{type(e).__name__}: {e}"
                ),
            })

        if not tech:
            skipped.append({
                "종목코드": code,
                "종목명": name,
                "이유": (
                    "Yahoo 데이터 없음/부족 "
                    "→ 이 종목만 자동 제외"
                ),
            })

            progress.progress(
                (i + 1) / total,
                text=f"{i + 1}/{total} 분석 중",
            )
            continue

        lead = float(
            r.get("주도주점수", 0) or 0
        )

        net = int(
            tech.get("순점수", 0) or 0
        )

        tech100 = max(
            0.0,
            min(
                100.0,
                ((net + 6) / 12) * 100,
            ),
        )

        try:
            from trader_core import _download_yf

            intraday_df = _download_yf(
                code,
                "국내",
            )

            if (
                intraday_df is None
                or intraday_df.empty
            ):
                skipped.append({
                    "종목코드": code,
                    "종목명": name,
                    "이유": (
                        "Yahoo 5분봉 없음 "
                        "→ 이 종목만 자동 제외"
                    ),
                })

                progress.progress(
                    (i + 1) / total,
                    text=f"{i + 1}/{total} 분석 중",
                )
                continue

            trend = score_leader_trend(
                intraday_df
            ) or {}

        except Exception as e:
            skipped.append({
                "종목코드": code,
                "종목명": name,
                "이유": (
                    f"추세분석 예외 "
                    f"{type(e).__name__}: {e}"
                ),
            })

            progress.progress(
                (i + 1) / total,
                text=f"{i + 1}/{total} 분석 중",
            )
            continue

        trend_score = float(
            trend.get("추세점수", 0) or 0
        )

        if (
            strategy_mode
            == "대장주 추세매매 모드"
        ):
            combined = (
                lead * 0.45
                + trend_score * 0.55
            )

            final_signal = trend.get(
                "추세판정",
                "⚪ 추세약함",
            )
        else:
            combined = (
                lead * 0.65
                + tech100 * 0.35
            )

            final_signal = tech.get(
                "종합신호",
                "⚪ 중립",
            )

        rows.append({
            "종목코드": code,
            "종목명": name,
            "현재가": r.get("현재가", ""),
            "등락률": r.get("등락률", ""),
            "주도주점수": round(lead, 1),
            "RSI": tech.get("RSI"),
            "거래량배수": tech.get("거래량배수"),
            "기술순점수": net,
            "추세점수": round(trend_score, 1),
            "종합점수": round(combined, 1),
            "판정": final_signal,
            "진입근거": trend.get(
                "추세이유",
                "",
            ),
        })

        progress.progress(
            (i + 1) / total,
            text=f"{i + 1}/{total} 분석 중",
        )

    progress.empty()

    st.session_state[
        "scan_skipped"
    ] = pd.DataFrame(skipped)

    if not rows:
        return pd.DataFrame()

    df = (
        pd.DataFrame(rows)
        .sort_values(
            [
                "종합점수",
                "주도주점수",
                "기술순점수",
            ],
            ascending=[
                False,
                False,
                False,
            ],
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


st.subheader("🔑 API 상태")

try:
    client.get_token()
    st.success(
        "✅ 한국투자 API 연결 정상"
    )
except Exception as e:
    st.error(
        f"API 연결 실패: {e}"
    )


st.divider()
st.subheader("🧪 주문 설정")

parts = split_budget(
    int(per_stock),
    [
        int(b1),
        int(b2),
        int(b3),
    ],
)

st.write(
    f"하루 최대 {int(budget):,}원 · "
    f"종목당 {int(per_stock):,}원 → "
    f"1차 {parts[0]:,}원 / "
    f"2차 {parts[1]:,}원 / "
    f"3차 {parts[2]:,}원"
)

st.write(
    f"손절 -{float(stop_loss):.1f}% / "
    f"1차 익절 +{float(take1):.1f}% / "
    f"2차 익절 +{float(take2):.1f}%"
)

st.info(
    "🧪 모의투자 완화 진입: "
    "TOP5 중 종합점수 40 이상이면 "
    "녹색 신호가 아니어도 1차 진입 허용"
)

st.info(
    f"현재 시장 상태: "
    f"{'장중' if is_market_open('KR') else '장외'} · "
    f"강제청산 기준시각: "
    f"{market_force_exit_time('KR')}"
)


dry_run = st.toggle(
    "🧪 주문 없이 판단만 보기",
    value=True,
)

if env == "real":
    real_confirm = st.checkbox(
        "실제 주문 위험을 이해했고 주문 전송을 허용합니다."
    )
else:
    real_confirm = True

can_execute = (
    live_unlocked
    and real_confirm
)


st.divider()
st.header("🚀 원터치 자동매매")

if st.button(
    "🚀 국내 자동매매 한 번에 실행",
    type="primary",
    use_container_width=True,
):
    try:
        st.info(
            "① 거래량·거래대금 기반 후보 탐색"
        )

        candidates = discover_domestic_candidates(
            client,
            top_n=20,
        )

        if (
            candidates is None
            or candidates.empty
        ):
            st.warning(
                "국내 후보가 없습니다."
            )
            st.stop()

        st.success(
            f"✅ 후보 {len(candidates)}개 발견"
        )

        st.info(
            "② 데이터 없는 종목은 건너뛰며 TOP5 계산"
        )

        leader_df = build_domestic_leader_df(
            candidates
        )

        st.session_state[
            "leader_df_kr"
        ] = leader_df

        if leader_df.empty:
            st.warning(
                "분석 가능한 TOP5가 없습니다."
            )
            st.stop()

        st.success(
            "✅ TOP5 선정 완료"
        )

        cycle = run_domestic_cycle(
            client=client,
            leader_df=leader_df,
            config=cfg,
            execute_orders=(
                can_execute
                and not dry_run
            ),
            source="APP",
        )

        st.session_state[
            "last_cycle_kr"
        ] = cycle

    except Exception as e:
        st.error(
            f"자동매매 오류: "
            f"{type(e).__name__}: {e}"
        )


st.divider()
st.subheader(
    "👑 오늘의 대장주 후보 TOP 5"
)

leader_show = st.session_state.get(
    "leader_df_kr",
    pd.DataFrame(),
)

if leader_show.empty:
    st.caption(
        "원터치 버튼을 누르면 계산됩니다."
    )
else:
    st.dataframe(
        leader_show,
        use_container_width=True,
        hide_index=True,
    )


scan_skipped = st.session_state.get(
    "scan_skipped",
    pd.DataFrame(),
)

if not scan_skipped.empty:
    st.warning(
        "아래 종목은 Yahoo 데이터 문제로 자동 제외되었습니다. "
        "다른 종목 분석은 계속 진행합니다."
    )

    st.dataframe(
        scan_skipped,
        use_container_width=True,
        hide_index=True,
    )


st.divider()
st.subheader(
    "🤖 자동매매 실행 결과"
)

cycle = st.session_state.get(
    "last_cycle_kr"
)

if not cycle:
    st.caption(
        "아직 실행된 사이클이 없습니다."
    )
else:
    if cycle.get("order_gate_message"):
        st.warning(
            cycle["order_gate_message"]
        )

    if cycle.get("balance_warning"):
        st.warning(
            cycle["balance_warning"]
        )

    if cycle.get("message"):
        st.info(
            cycle["message"]
        )

    actions = cycle.get(
        "actions",
        [],
    )

    if actions:
        st.markdown(
            "### ✅ 주문/판단 액션"
        )

        st.dataframe(
            pd.DataFrame(actions),
            use_container_width=True,
            hide_index=True,
        )

    diagnostics = cycle.get(
        "diagnostics",
        [],
    )

    if diagnostics:
        st.markdown(
            "### 🔎 종목별 매수 안 된 이유"
        )

        st.dataframe(
            pd.DataFrame(diagnostics),
            use_container_width=True,
            hide_index=True,
        )

    state_after = cycle.get(
        "state",
        {},
    )

    st.caption(
        f"추적 종목 "
        f"{len(state_after.get('positions', {}))}개 · "
        f"누적 실제 신규매수 "
        f"{int(state_after.get('daily_buy_amount', 0)):,}원 · "
        f"실제 주문 "
        f"{int(state_after.get('daily_orders', 0))}회"
    )


current_state = load_state()

st.info(
    f"📌 오늘 추적 종목 "
    f"{len(current_state.get('positions', {}))}개 · "
    f"누적 실제 신규매수 "
    f"{int(current_state.get('daily_buy_amount', 0)):,}원 · "
    f"실제 주문 "
    f"{int(current_state.get('daily_orders', 0))}회"
)

if st.button(
    "♻️ 오늘 국내 자동매매 상태 초기화"
):
    reset_today_state()

    st.session_state.pop(
        "last_cycle_kr",
        None,
    )

    st.success(
        "오늘 상태를 초기화했습니다."
    )

    st.rerun()


st.warning(
    "Render worker가 주문전송=ON이면 화면을 닫아도 계속 반복합니다. "
    "앱 버튼과 worker를 동시에 주문전송 ON으로 두면 중복주문 가능성이 있으므로 "
    "모의테스트 초반에는 한쪽만 ON을 권장합니다."
)


st.divider()
st.subheader(
    "🧾 최근 주문 로그"
)

log = load_trade_log()

if log.empty:
    st.caption(
        "아직 저장된 주문 로그가 없습니다."
    )
else:
    st.dataframe(
        log.tail(100),
        use_container_width=True,
        hide_index=True,
    )

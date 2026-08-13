import streamlit as st
import pandas as pd
import json
import os
import requests
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from trader_core import (
    Settings,
    KISClient,
    discover_domestic_candidates,
    score_ticker,
    split_budget,
    is_market_open,
    market_force_exit_time,
)

from auto_engine import (
    AutoConfig,
    run_domestic_cycle,
    run_overseas_cycle,
    sync_us_budget_from_krw,
)

from trend_strategy import score_leader_trend


# =========================================================
# 기본 화면
# =========================================================
st.set_page_config(
    page_title="쏭 자동매매",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 쏭 국내·미국 자동매매")
st.caption(
    "후보 탐색 → 대장주 선정 → 기술분석 → 자동매매 판단/주문을 실행합니다."
)

settings = Settings.from_env()

# =========================================================
# Worker 상태 표시
# =========================================================
KST = ZoneInfo("Asia/Seoul")
WORKER_STATUS_FILE = Path(
    os.getenv(
        "SONG_WORKER_STATUS_FILE",
        "/tmp/song_trader/worker_status.json",
    )
)

WORKER_STATUS_URL = os.getenv(
    "SONG_WORKER_STATUS_URL",
    "",
).strip()


def load_worker_status():
    """
    1순위: Railway worker의 /status API
    2순위: 같은 서버에서 실행할 때의 로컬 worker_status.json
    """
    if WORKER_STATUS_URL:
        try:
            url = WORKER_STATUS_URL.rstrip("/")
            if not url.endswith("/status"):
                url = url + "/status"

            resp = requests.get(
                url,
                timeout=5,
                headers={
                    "User-Agent": "song-stock-trader-status/1.0",
                },
            )

            if resp.ok:
                data = resp.json()
                if isinstance(data, dict):
                    data["_source"] = "railway"
                    return data
        except Exception:
            pass

    if not WORKER_STATUS_FILE.exists():
        return None

    try:
        data = json.loads(
            WORKER_STATUS_FILE.read_text(encoding="utf-8")
        )
        if isinstance(data, dict):
            data["_source"] = "local"
            return data
    except Exception:
        pass

    return None


def render_worker_status():
    st.divider()
    st.subheader("🤖 자동매매 Worker 상태")

    status = load_worker_status()

    if not status:
        st.error("🔴 Worker 상태 확인 불가")
        if WORKER_STATUS_URL:
            st.caption(
                "Railway Worker 상태 주소에 연결하지 못했습니다. "
                "Railway 배포 상태와 SONG_WORKER_STATUS_URL을 확인하세요."
            )
        else:
            st.caption(
                "SONG_WORKER_STATUS_URL이 설정되지 않았거나 "
                "worker_status.json을 찾지 못했습니다."
            )
        return

    heartbeat_raw = (
        status.get("heartbeat_at")
        or status.get("updated_at")
    )

    seconds_ago = 999999.0

    if heartbeat_raw:
        try:
            heartbeat = datetime.fromisoformat(heartbeat_raw)
            if heartbeat.tzinfo is None:
                heartbeat = heartbeat.replace(tzinfo=KST)
            seconds_ago = max(
                0.0,
                (
                    datetime.now(KST)
                    - heartbeat.astimezone(KST)
                ).total_seconds(),
            )
        except Exception:
            pass

    running = bool(status.get("running"))
    stale = seconds_ago > 180

    if running and not stale:
        st.success("🟢 Worker 정상 작동 중")
        st.info(
            status.get(
                "stage_message",
                "🤖 자동매매 Worker 실행 중",
            )
        )
    elif running and stale:
        st.error("🔴 Worker 응답 지연")
        st.warning(
            "마지막 heartbeat가 3분 이상 갱신되지 않았습니다."
        )
    else:
        st.error("🔴 Worker 중지 상태")

    if seconds_ago < 999999:
        st.caption(
            f"마지막 heartbeat: {int(seconds_ago)}초 전"
        )

    env_text = (
        "모의투자"
        if str(status.get("env", "demo")) == "demo"
        else "실전투자"
    )
    order_text = (
        "주문전송 ON"
        if bool(status.get("execute_orders"))
        else "DRY"
    )

    st.caption(
        f"Worker 모드: {env_text} · {order_text}"
    )

    source_text = (
        "Railway"
        if status.get("_source") == "railway"
        else "로컬"
    )
    st.caption(f"상태 연결: {source_text}")


# =========================================================
# 사이드바 설정
# =========================================================
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
        help=(
            "RSI/볼린저 기반 기존 전략 또는 "
            "VWAP·돌파·눌림 중심 대장주 추세전략"
        ),
    )

    st.markdown("### 💰 자금 설정")

    if market == "국내":
        budget = st.number_input(
            "1일 최대 신규매수 금액(원)",
            min_value=10000,
            value=10000000,
            step=10000,
        )

        per_stock = st.number_input(
            "종목당 최대 금액(원)",
            min_value=10000,
            value=10000000,
            step=10000,
        )

        us_daily_budget = 0.0
        us_per_stock_budget = 0.0
        us_daily_budget_krw = 10000000
        us_per_stock_budget_krw = 10000000
        usd_krw_rate = 1400.0

    else:
        usd_krw_rate = st.number_input(
            "미국 환산 기준 1달러(원)",
            min_value=500.0,
            max_value=3000.0,
            value=1400.0,
            step=10.0,
            help=(
                "미국 모의테스트 예산을 원화 기준에서 USD로 환산합니다. "
                "실제 주문은 계좌의 주문가능금액이 우선입니다."
            ),
        )

        us_daily_budget_krw = st.number_input(
            "미국 1일 최대 신규매수 원화기준(원)",
            min_value=10000,
            value=10000000,
            step=100000,
        )

        us_per_stock_budget_krw = st.number_input(
            "미국 종목당 최대 원화기준(원)",
            min_value=10000,
            value=10000000,
            step=100000,
        )

        us_daily_budget = (
            float(us_daily_budget_krw)
            / float(usd_krw_rate)
        )
        us_per_stock_budget = (
            float(us_per_stock_budget_krw)
            / float(usd_krw_rate)
        )

        budget = 10000000
        per_stock = 10000000

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

    st.markdown("### 🔐 실전 잠금")

    live_phrase = st.text_input(
        "실전 확인문구",
        type="password",
        placeholder="실전투자일 때만 입력",
    )


# =========================================================
# 환경 / KIS
# =========================================================
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
        st.error(
            "🔴 실전투자 잠금이 해제되었습니다. "
            "실제 주문이 발생할 수 있습니다."
        )
    else:
        st.warning(
            "🔒 실전투자는 잠겨 있습니다."
        )

@st.cache_resource(show_spinner=False)
def _cached_kis_client(
    env_name: str,
    paper_app_key: str,
    paper_app_secret: str,
    paper_account_no: str,
    paper_product_code: str,
    live_app_key: str,
    live_app_secret: str,
    live_account_no: str,
    live_product_code: str,
):
    cached_settings = Settings(
        paper_app_key=paper_app_key,
        paper_app_secret=paper_app_secret,
        paper_account_no=paper_account_no,
        paper_account_product_code=paper_product_code,
        live_app_key=live_app_key,
        live_app_secret=live_app_secret,
        live_account_no=live_account_no,
        live_account_product_code=live_product_code,
        allow_live=settings.allow_live,
        live_unlock_phrase=settings.live_unlock_phrase,
    )
    return KISClient(
        settings=cached_settings,
        env=env_name,
    )


client = _cached_kis_client(
    env,
    settings.paper_app_key,
    settings.paper_app_secret,
    settings.paper_account_no,
    settings.paper_account_product_code,
    settings.live_app_key,
    settings.live_app_secret,
    settings.live_account_no,
    settings.live_account_product_code,
)


# =========================================================
# 자동매매 설정
# =========================================================
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

    min_combined_score=50.0,

    require_green_signal=True,
)

cfg.demo_relaxed_entry_enabled = True
cfg.demo_min_combined_score = 50.0
cfg.last_entry_time = "15:10"
cfg.force_exit_time = "15:20"

cfg.us_daily_budget_krw = int(us_daily_budget_krw)
cfg.us_per_stock_budget_krw = int(us_per_stock_budget_krw)
cfg.usd_krw_rate = float(usd_krw_rate)
sync_us_budget_from_krw(cfg)

if not hasattr(cfg, "us_default_stage_qty"):
    cfg.us_default_stage_qty = 1

if not hasattr(cfg, "us_last_entry_time"):
    cfg.us_last_entry_time = "15:30"

if not hasattr(cfg, "us_force_exit_time"):
    cfg.us_force_exit_time = "15:50"


# =========================================================
# 국내 대장주 TOP5 계산 함수
# =========================================================
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
            tech = score_ticker(
                code,
                "국내",
            )
        except Exception:
            tech = None

        if not tech:
            progress.progress(
                (i + 1) / total,
                text=f"{i + 1}/{total} 분석 중",
            )
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

            intraday_df = _download_yf(
                code,
                "국내",
            )

            trend = score_leader_trend(
                intraday_df
            )

        except Exception:
            trend = None

        trend_score = float(
            (trend or {}).get(
                "추세점수",
                0,
            )
        )

        if strategy_mode == "대장주 추세매매 모드":
            combined = (
                lead * 0.45
                + trend_score * 0.55
            )

            final_signal = (
                (trend or {}).get(
                    "추세판정",
                    "⚪ 추세약함",
                )
            )
        else:
            combined = (
                lead * 0.65
                + tech100 * 0.35
            )

            final_signal = tech.get(
                "종합신호"
            )

        rows.append(
            {
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
                    (trend or {}).get(
                        "추세이유",
                        "",
                    )
                    if strategy_mode == "대장주 추세매매 모드"
                    else tech.get("종합신호")
                ),
            }
        )

        progress.progress(
            (i + 1) / total,
            text=f"{i + 1}/{total} 분석 중",
        )

    progress.empty()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    df = df.sort_values(
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
    ).head(5)

    df = df.reset_index(drop=True)

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


# =========================================================
# 미국 TOP5 계산 함수
# =========================================================
def build_us_leader_df(symbols):
    score_rows = []

    progress = st.progress(
        0,
        text="🇺🇸 미국 후보 분석 시작...",
    )

    total = max(len(symbols), 1)

    for i, symbol in enumerate(symbols):
        try:
            row = score_ticker(
                symbol,
                market="미국",
            )

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
        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    score_df = pd.DataFrame(
        score_rows
    ).sort_values(
        [
            "순점수",
            "거래량배수",
        ],
        ascending=[
            False,
            False,
        ],
    )

    us_top = (
        score_df
        .head(5)
        .copy()
        .reset_index(drop=True)
    )

    tech100 = (
        (
            us_top["순점수"]
            .clip(-6, 6)
            + 6
        )
        / 12
        * 100
    ).astype(float)

    vol_bonus = (
        us_top["거래량배수"]
        .clip(0, 2.0)
        / 2.0
        * 10
    ).astype(float)

    us_top["종합점수"] = (
        tech100 * 0.9
        + vol_bonus
    ).round(1)

    labels = [
        "⭐ 1위",
        "⭐ 2위",
        "⭐ 3위",
        "4위",
        "5위",
    ]

    us_top.insert(
        0,
        "순위",
        labels[:len(us_top)],
    )

    us_top["종목코드"] = (
        us_top["종목"]
        .astype(str)
    )

    us_top["종목명"] = (
        us_top["종목"]
        .astype(str)
    )

    us_top["판정"] = (
        us_top["종합신호"]
    )

    us_top["진입근거"] = (
        "RSI "
        + us_top["RSI"].astype(str)
        + " / 거래량배수 "
        + us_top["거래량배수"].astype(str)
        + " / 매수점수 "
        + us_top["매수점수"].astype(str)
        + " / 매도점수 "
        + us_top["매도점수"].astype(str)
    )

    return score_df, us_top


# =========================================================
# 상태 표시
# =========================================================
c1, c2, c3 = st.columns(3)

c1.metric(
    "운용 모드",
    "모의"
    if env == "demo"
    else "실전",
)

c2.metric(
    "시장",
    market,
)

c3.metric(
    "전략",
    "대장주"
    if strategy_mode
    == "대장주 추세매매 모드"
    else "기술지표",
)


# =========================================================
# API 상태
# =========================================================
st.divider()
st.subheader("🔑 API 상태")
st.caption(
    "화면 새로고침마다 토큰을 다시 발급하지 않습니다. "
    "worker와 동시에 토큰을 요청하면 KIS 발급 제한에 걸릴 수 있어 "
    "연결 확인은 필요할 때만 누르세요."
)

if st.button("🔌 API 연결 확인", use_container_width=True):
    try:
        client.get_token()
        st.success("✅ 한국투자 API 연결 정상")
    except Exception as e:
        st.error(
            f"API 연결 실패: {type(e).__name__}: {e}"
        )


# =========================================================
# 주문 설정
# =========================================================
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

if market == "국내":
    st.write(
        f"종목당 {int(per_stock):,}원 → "
        f"1차 {parts[0]:,}원 / "
        f"2차 {parts[1]:,}원 / "
        f"3차 {parts[2]:,}원"
    )
else:
    total_pct = max(
        1,
        int(b1)
        + int(b2)
        + int(b3),
    )

    usd_total = float(cfg.us_per_stock_budget_usd)
    usd1 = usd_total * int(b1) / total_pct
    usd2 = usd_total * int(b2) / total_pct
    usd3 = usd_total * int(b3) / total_pct

    st.write(
        f"환산기준 1달러 = {float(cfg.usd_krw_rate):,.0f}원"
    )
    st.write(
        f"미국 하루 최대 "
        f"{int(cfg.us_daily_budget_krw):,}원 "
        f"→ 약 ${float(cfg.us_daily_budget_usd):,.2f}"
    )
    st.write(
        f"미국 종목당 "
        f"{int(cfg.us_per_stock_budget_krw):,}원 "
        f"→ 약 ${usd_total:,.2f} → "
        f"1차 약 ${usd1:,.2f} / "
        f"2차 약 ${usd2:,.2f} / "
        f"3차 약 ${usd3:,.2f}"
    )


st.write(
    f"손절 -{float(stop_loss):.1f}% / "
    f"1차 익절 +{float(take1):.1f}% / "
    f"2차 익절 +{float(take2):.1f}%"
)

open_now = is_market_open(
    "KR"
    if market == "국내"
    else "US"
)

st.info(
    f"현재 시장 상태: "
    f"{'장중' if open_now else '장외'} · "
    f"강제청산 기준시각: "
    f"{market_force_exit_time('KR' if market == '국내' else 'US')}"
)

if market == "국내":
    st.info(
        "⏰ 국내: 후보분석 08:30~16:00 · 신규/추가매수 15:10까지 · "
        "15:20부터 강제청산 · 실제 주문은 15:30 전에만 전송"
    )
    st.info(
        "💳 모의 매수는 목표금액보다 주문가능금액이 적으면 "
        "5% 안전여유를 남기고 가능한 수량으로 자동 축소합니다."
    )
else:
    st.info(
        "🇺🇸 미국: 09:00 ET부터 후보준비 · 09:30 ET부터 주문 · "
        "15:30 ET 신규/추가매수 종료 · 15:50 ET 강제청산"
    )
    st.info(
        "💵 미국 예산은 원화기준 금액을 설정한 환율로 USD 환산하며, "
        "실제 해외잔고에서 가용 USD가 확인되면 5% 안전여유를 적용합니다."
    )


render_worker_status()


# =========================================================
# 주문 실행 방식
# =========================================================
# 별도 DRY-RUN 토글은 제거했습니다.
# 모의투자에서는 원터치 실행 시 모의주문을 전송하고,
# 실전투자는 기존 잠금/확인 절차를 그대로 통과해야 합니다.
dry_run = False


# =========================================================
# 실전 추가 확인
# =========================================================
if env == "real":
    st.error("⚠️ 현재 실전투자 모드입니다.")

    real_confirm = st.checkbox(
        "실제 주문 위험을 이해했고 주문 전송을 허용합니다."
    )
else:
    real_confirm = True

can_execute = (
    live_unlocked
    and real_confirm
)


# =========================================================
# Railway 자동매매 상태 / 후보
# =========================================================
st.divider()
st.header("🚀 자동매매")
st.info(
    "실제 모의주문은 Railway worker 한 곳에서만 전송합니다. "
    "이 화면의 새로고침 버튼은 주문을 보내지 않습니다."
)

if st.button(
    "🔄 Railway 상태·후보 새로고침",
    use_container_width=True,
):
    st.rerun()

# 예전 앱 수동실행 결과가 남아 있으면 제거하여 혼동을 막습니다.
st.session_state.pop("last_cycle_kr", None)
st.session_state.pop("last_cycle_us", None)

worker_for_candidates = load_worker_status() or {}
remote_top5_key = "kr_top5" if market == "국내" else "us_top5"
remote_top5 = worker_for_candidates.get(remote_top5_key, []) or []

if remote_top5:
    leader_show = pd.DataFrame(remote_top5)
    if market == "국내":
        st.subheader("🇰🇷 Railway 국내 자동매매 후보 TOP 5")
    else:
        st.subheader("🇺🇸 Railway 미국 자동매매 후보 TOP 5")

    st.dataframe(
        leader_show,
        use_container_width=True,
        hide_index=True,
    )

    if not leader_show.empty:
        top = leader_show.iloc[0]
        st.info(
            f"현재 1순위 후보: "
            f"{top.get('종목명', top.get('종목', ''))} "
            f"({top.get('종목코드', top.get('종목', ''))}) · "
            f"종합점수 {top.get('종합점수', '')} · "
            f"{top.get('판정', top.get('종합신호', ''))}"
        )
else:
    st.caption("Railway worker에서 아직 TOP5 후보가 전달되지 않았습니다.")



# =========================================================
# 자동매매 실행 결과
# =========================================================
st.divider()
st.subheader("🚀 자동매매 실행 결과")

worker_snapshot = load_worker_status() or {}
remote_cycle_key = "kr_last_result" if market == "국내" else "us_last_result"
cycle = worker_snapshot.get(remote_cycle_key) or {}

if not cycle:
    st.caption("Railway worker에서 아직 완료된 자동매매 사이클 결과가 없습니다.")
else:
    actions = cycle.get("actions", []) or []
    diagnostics = cycle.get("diagnostics", []) or []

    buy_count = sum(1 for x in actions if str(x.get("action", "")).startswith("BUY") and x.get("status") == "ORDERED")
    sell_count = sum(1 for x in actions if not str(x.get("action", "")).startswith("BUY") and x.get("status") == "ORDERED")
    fail_count = sum(1 for x in actions if x.get("status") in ("ERROR", "REJECT"))
    skip_count = len(diagnostics)

    a, b, c, d = st.columns(4)
    a.metric("매수", buy_count)
    b.metric("매도", sell_count)
    c.metric("SKIP", skip_count)
    d.metric("오류", fail_count)

    if cycle.get("message"):
        st.info(str(cycle.get("message")))

    if cycle.get("balance_warning"):
        st.warning(f"잔고조회 경고: {cycle.get('balance_warning')}")

    important = []
    for x in actions:
        status = str(x.get("status", ""))
        if status in ("ORDERED", "ERROR", "REJECT"):
            important.append({
                "종목": x.get("symbol", ""),
                "동작": x.get("action", ""),
                "상태": status,
                "수량": x.get("qty", ""),
                "점수": x.get("combined_score", ""),
                "이유": x.get("reason", x.get("msg1", "")),
            })

    if important:
        st.dataframe(pd.DataFrame(important), use_container_width=True, hide_index=True)

    if fail_count == 0 and not important:
        st.caption("이번 사이클에는 주문 접수 또는 주문 오류가 없었습니다.")

    # 진단은 긴 원문 대신 중요한 SKIP 사유만 최대 5개 표시
    if diagnostics:
        simple_diag = []
        for x in diagnostics[:5]:
            simple_diag.append({
                "종목": x.get("symbol", x.get("code", "")),
                "사유": x.get("reason", x.get("detail", "")),
            })
        st.caption("최근 판단 요약")
        st.dataframe(pd.DataFrame(simple_diag), use_container_width=True, hide_index=True)


# =========================================================
# 자동매매 일지
# =========================================================
st.divider()
st.subheader("📒 자동매매 일지")

journal_key = "kr_journal" if market == "국내" else "us_journal"
journal = worker_snapshot.get(journal_key, []) or []

if not journal:
    st.caption("아직 기록된 자동매매 주문이 없습니다. 새 worker 배포 이후 주문부터 표시됩니다.")
else:
    journal_df = pd.DataFrame(journal)
    show_cols = [c for c in [
        "시간", "종목", "종목코드", "구분", "수량", "기준가",
        "종합점수", "손익률", "이유", "상태"
    ] if c in journal_df.columns]
    st.dataframe(journal_df[show_cols].tail(100).iloc[::-1], use_container_width=True, hide_index=True)
    st.caption("※ '주문접수'는 KIS API가 주문을 정상 접수했다는 뜻이며 실제 체결가와는 다를 수 있습니다.")

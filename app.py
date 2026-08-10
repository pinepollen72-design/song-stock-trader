import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
st.set_page_config(page_title="단기매매 신호 분석기", page_icon="📈", layout="wide")

st.title("📈 볼린저 밴드 + 거래량 + 캔들 + RSI 단기매매 신호 분석기")
st.caption("교육·분석용 도구입니다. 실제 매매의 수익을 보장하지 않으며 투자 판단은 본인이 하세요.")
# 한국투자증권 모의투자 API 인증 테스트
def get_kis_token():
    url = "https://openapivts.koreainvestment.com:29443/oauth2/tokenP"

    headers = {
        "content-type": "application/json"
    }

    body = {
        "grant_type": "client_credentials",
        "appkey": st.secrets["KIS_APP_KEY"],
        "appsecret": st.secrets["KIS_APP_SECRET"]
    }

    response = requests.post(url, headers=headers, data=json.dumps(body))

    if response.status_code == 200:
        return response.json().get("access_token")
    else:
        st.error("한국투자증권 API 인증 실패")
        st.write(response.text)
        return None


if st.button("🔐 모의투자 API 연결 테스트"):
    token = get_kis_token()

    if token:
        st.success("✅ 한국투자증권 모의투자 API 연결 성공!")

def get_mock_balance():
    token = get_kis_token()

    if not token:
        return None

    url = "https://openapivts.koreainvestment.com:29443/uapi/domestic-stock/v1/trading/inquire-balance"

    headers = {
        "content-type": "application/json; charset=utf-8",
        "authorization": f"Bearer {token}",
        "appkey": st.secrets["KIS_APP_KEY"],
        "appsecret": st.secrets["KIS_APP_SECRET"],
        "tr_id": "VTTC8434R",
        "custtype": "P",
    }

    params = {
        "CANO": st.secrets["KIS_ACCOUNT_NO"],
        "ACNT_PRDT_CD": st.secrets["KIS_ACCOUNT_PRODUCT_CODE"],
        "AFHR_FLPR_YN": "N",
        "OFL_YN": "",
        "INQR_DVSN": "02",
        "UNPR_DVSN": "01",
        "FUND_STTL_ICLD_YN": "N",
        "FNCG_AMT_AUTO_RDPT_YN": "N",
        "PRCS_DVSN": "01",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": "",
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        return response.json()

    st.error("모의투자 잔고 조회 실패")
    st.write(response.text)
    return None
if st.button("💰 모의투자 잔고 조회"):
    balance = get_mock_balance()

    if balance:
        st.success("✅ 모의투자 잔고 조회 성공!")
        st.json(balance)

# 한국투자증권 국내주식 현재가 조회
def get_kis_price(stock_code):
    token = get_kis_token()

    if not token:
        return None

    url = "https://openapivts.koreainvestment.com:29443/uapi/domestic-stock/v1/quotations/inquire-price"

    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {token}",
        "appkey": st.secrets["KIS_APP_KEY"],
        "appsecret": st.secrets["KIS_APP_SECRET"],
        "tr_id": "FHKST01010100"
    }

    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": stock_code
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        return response.json()

    st.error("현재가 조회 실패")
    st.write(response.text)
    return None

# 한국투자증권 모의투자 국내주식 매수
def mock_buy_stock(stock_code, qty=1):
    token = get_kis_token()

    if not token:
        return None

    url = "https://openapivts.koreainvestment.com:29443/uapi/domestic-stock/v1/trading/order-cash"

    headers = {
        "content-type": "application/json; charset=utf-8",
        "authorization": f"Bearer {token}",
        "appkey": st.secrets["KIS_APP_KEY"],
        "appsecret": st.secrets["KIS_APP_SECRET"],
        "tr_id": "VTTC0802U",
        "custtype": "P"
    }

    body = {
        "CANO": st.secrets["KIS_ACCOUNT_NO"],
        "ACNT_PRDT_CD": st.secrets["KIS_ACCOUNT_PRODUCT_CODE"],
        "PDNO": stock_code,
        "ORD_DVSN": "01",
        "ORD_QTY": str(qty),
        "ORD_UNPR": "0"
    }

    response = requests.post(
        url,
        headers=headers,
        data=json.dumps(body)
    )

    if response.status_code == 200:
        return response.json()

    st.error("모의매수 주문 요청 실패")
    st.write(response.text)
    return None
stock_list = {
    "삼성전자": "005930",
    "SK하이닉스": "000660",
    "현대차": "005380",
    "기아": "000270",
    "NAVER": "035420",
    "카카오": "035720",
    "LG에너지솔루션": "373220",
    "삼성바이오로직스": "207940",
    "셀트리온": "068270",
    "KB금융": "105560",
    "신한지주": "055550",
    "POSCO홀딩스": "005490",
    "삼성SDI": "006400",
    "LG화학": "051910",
    "현대모비스": "012330",
    "삼성물산": "028260",
    "한화에어로스페이스": "012450",
    "두산에너빌리티": "034020"

}

selected_stock = st.selectbox(
    "🔎 종목 선택",
    list(stock_list.keys())
)

stock_code_test = stock_list[selected_stock]

if st.button("📈 한국투자 현재가 조회"):
    price_data = get_kis_price(stock_code_test)
if st.button("🧪 선택 종목 모의매수 1주"):
    order_result = mock_buy_stock(stock_code_test, qty=1)

    if order_result:
        if order_result.get("rt_cd") == "0":
            st.success(f"✅ {selected_stock} 1주 모의매수 주문 성공!")
            st.json(order_result)
        else:
            st.error("모의매수 주문이 거절되었습니다.")
            st.json(order_result)
    if price_data and price_data.get("output"):
        output = price_data["output"]

        st.success("✅ 현재가 조회 성공!")
        st.metric(
            "현재가",
            f'{int(output["stck_prpr"]):,}원'
        )
    else:
        st.error("현재가 데이터를 가져오지 못했습니다.")
# -----------------------------
# Data
# -----------------------------
@st.cache_data(ttl=300)
def load_data(symbol, period="6mo", interval="1d"):
    try:
        import yfinance as yf
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna()
    except Exception as e:
        st.error(f"데이터를 불러오지 못했습니다: {e}")
        return None

def indicators(df, bb_period=20, bb_std=2, rsi_period=14, vol_period=20):
    d = df.copy()
    close = d["Close"].astype(float)
    high = d["High"].astype(float)
    low = d["Low"].astype(float)
    open_ = d["Open"].astype(float)
    volume = d["Volume"].astype(float)

    d["MA20"] = close.rolling(bb_period).mean()
    d["STD20"] = close.rolling(bb_period).std()
    d["BB_UPPER"] = d["MA20"] + bb_std * d["STD20"]
    d["BB_LOWER"] = d["MA20"] - bb_std * d["STD20"]

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(rsi_period).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    rs = gain / loss.replace(0, np.nan)
    d["RSI"] = 100 - (100 / (1 + rs))

    d["VOL_MA"] = volume.rolling(vol_period).mean()
    d["VOL_RATIO"] = volume / d["VOL_MA"]

    d["MA5"] = close.rolling(5).mean()
    d["MA20_SIMPLE"] = close.rolling(20).mean()

    # Candle features
    d["BODY"] = (close - open_).abs()
    d["RANGE"] = (high - low).replace(0, np.nan)
    d["LOWER_WICK"] = np.minimum(open_, close) - low
    d["UPPER_WICK"] = high - np.maximum(open_, close)
    d["BULL"] = close > open_
    d["BEAR"] = close < open_

    return d.dropna()

def score_latest(d):
    x = d.iloc[-1]
    prev = d.iloc[-2]

    buy = 0
    sell = 0
    reasons_buy = []
    reasons_sell = []

    # Bollinger
    if x["Close"] <= x["BB_LOWER"] * 1.01:
        buy += 2
        reasons_buy.append("볼린저 하단 근처")
    if prev["Close"] < prev["BB_LOWER"] and x["Close"] > x["BB_LOWER"]:
        buy += 2
        reasons_buy.append("볼린저 하단 이탈 후 재진입")

    if x["Close"] >= x["BB_UPPER"] * 0.99:
        sell += 2
        reasons_sell.append("볼린저 상단 근처")
    if prev["Close"] > prev["BB_UPPER"] and x["Close"] < x["BB_UPPER"]:
        sell += 2
        reasons_sell.append("볼린저 상단 이탈 후 재진입")

    # RSI
    if 30 <= x["RSI"] <= 40 and x["RSI"] > prev["RSI"]:
        buy += 2
        reasons_buy.append("RSI 과매도권 반등")
    elif x["RSI"] < 30:
        buy += 1
        reasons_buy.append("RSI 30 이하")

    if 60 <= x["RSI"] <= 70 and x["RSI"] < prev["RSI"]:
        sell += 2
        reasons_sell.append("RSI 과매수권 하락 전환")
    elif x["RSI"] > 70:
        sell += 1
        reasons_sell.append("RSI 70 이상")

    # Volume
    if x["VOL_RATIO"] >= 1.5 and x["BULL"]:
        buy += 2
        reasons_buy.append("거래량 20일 평균의 1.5배 이상 + 양봉")
    if x["VOL_RATIO"] >= 1.5 and x["BEAR"]:
        sell += 2
        reasons_sell.append("거래량 20일 평균의 1.5배 이상 + 음봉")

    # Candle
    if x["BULL"] and x["BODY"] / x["RANGE"] >= 0.6:
        buy += 1
        reasons_buy.append("몸통이 큰 양봉")
    if x["BEAR"] and x["BODY"] / x["RANGE"] >= 0.6:
        sell += 1
        reasons_sell.append("몸통이 큰 음봉")

    if x["LOWER_WICK"] > x["BODY"] * 1.2 and x["BULL"]:
        buy += 1
        reasons_buy.append("긴 아래꼬리 + 양봉")

    if x["UPPER_WICK"] > x["BODY"] * 1.2 and x["BEAR"]:
        sell += 1
        reasons_sell.append("긴 윗꼬리 + 음봉")

    # Trend
    if x["MA5"] > x["MA20_SIMPLE"]:
        buy += 1
        reasons_buy.append("5일선 > 20일선")
    if x["MA5"] < x["MA20_SIMPLE"]:
        sell += 1
        reasons_sell.append("5일선 < 20일선")

    net = buy - sell
    if net >= 4:
        signal = "🟢 매수 후보"
    elif net <= -4:
        signal = "🔴 매도 후보"
    else:
        signal = "🟡 관망"

    return buy, sell, net, signal, reasons_buy, reasons_sell

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("⚙️ 설정")
symbol = stock_code_test + ".KS"
period = st.sidebar.selectbox("조회 기간", ["3mo", "6mo", "1y", "2y"], index=1)
bb_period = st.sidebar.number_input("볼린저 기간", 5, 100, 20)
bb_std = st.sidebar.number_input("표준편차 배수", 0.5, 4.0, 2.0, 0.1)
rsi_period = st.sidebar.number_input("RSI 기간", 5, 50, 14)
st.sidebar.info("한국 주식은 .KS, 코스닥은 .KQ를 사용합니다.")

if st.sidebar.button("🔄 분석하기", type="primary"):
    st.cache_data.clear()

df = load_data(symbol, period=period)

if df is None or len(df) < 30:
    st.warning("데이터가 부족합니다. 종목 코드와 조회 기간을 확인해주세요.")
    st.stop()

d = indicators(df, bb_period, bb_std, rsi_period)
buy, sell, net, signal, rb, rs = score_latest(d)
x = d.iloc[-1]

# Header
c1, c2, c3, c4 = st.columns(4)
c1.metric("현재가", f"{x['Close']:,.0f}")
c2.metric("RSI", f"{x['RSI']:.1f}")
c3.metric("거래량 비율", f"{x['VOL_RATIO']:.2f}배")
c4.metric("종합 신호", signal)

st.divider()

# Scores
left, right = st.columns(2)
with left:
    st.subheader("🟢 매수 점수")
    st.metric("Buy Score", f"{buy}/10+")
    for r in rb:
        st.write("•", r)
    if not rb:
        st.write("특별한 매수 조건 없음")

with right:
    st.subheader("🔴 매도 점수")
    st.metric("Sell Score", f"{sell}/10+")
    for r in rs:
        st.write("•", r)
    if not rs:
        st.write("특별한 매도 조건 없음")

st.divider()

# Chart
st.subheader("📊 지표")
chart = d[["Close", "MA20", "BB_UPPER", "BB_LOWER"]].rename(
    columns={"Close":"주가", "MA20":"20일선", "BB_UPPER":"볼린저 상단", "BB_LOWER":"볼린저 하단"}
)
st.line_chart(chart)

st.subheader("📉 RSI")
st.line_chart(d[["RSI"]])

st.subheader("📦 거래량")
st.bar_chart(d[["Volume", "VOL_MA"]].rename(columns={"Volume":"거래량","VOL_MA":"20일 평균거래량"}))

st.divider()
st.subheader("🧾 최근 데이터")
show = d[["Open","High","Low","Close","Volume","BB_UPPER","BB_LOWER","RSI","VOL_RATIO"]].tail(10).copy()
show.columns = ["시가","고가","저가","종가","거래량","BB상단","BB하단","RSI","거래량배수"]
st.dataframe(show.style.format({
    "시가":"{:,.0f}", "고가":"{:,.0f}", "저가":"{:,.0f}", "종가":"{:,.0f}",
    "거래량":"{:,.0f}", "BB상단":"{:,.0f}", "BB하단":"{:,.0f}",
    "RSI":"{:.1f}", "거래량배수":"{:.2f}"
}), use_container_width=True)

st.caption("주의: 이 프로그램은 기술적 지표를 조합한 교육·분석용 도구입니다.")
st.subheader("🚀 자동 종목 스캐너")

if st.button("🔎 18개 종목 자동 분석"):
        results = []

        with st.spinner("18개 종목을 분석하고 있어요..."):
            for name, code in stock_list.items(): 
                try:
                    scan_symbol = code + ".KS"
                    scan_df = load_data(scan_symbol, period=period)

                    if scan_df is None or len(scan_df) < 30:
                        continue

                    scan_d = indicators(
                        scan_df,
                        bb_period,
                        bb_std,
                        rsi_period
                    )

                    if scan_d is None or len(scan_d) < 2:
                        continue

                    buy_s, sell_s, net_s, signal_s, _, _ = score_latest(scan_d)
                    latest = scan_d.iloc[-1]

                    results.append({
                        "종목": name,
                        "종목코드": code,
                        "현재가": int(latest["Close"]),
                        "RSI": round(float(latest["RSI"]), 1),
                        "거래량배수": round(float(latest["VOL_RATIO"]), 2),
                        "매수점수": buy_s,
                        "매도점수": sell_s,
                        "순점수": net_s,
                        "종합신호": signal_s
                    })

                except Exception:
                    continue
            if results:
                result_df = pd.DataFrame(results)

                result_df = result_df.sort_values(
                    by=["순점수", "매수점수", "거래량배수"],
                    ascending=[False, False, False]
                )

                st.success(f"✅ {len(result_df)}개 종목 분석 완료!")

                st.subheader("🏆 단기매매 후보 순위")

                st.dataframe(
                    result_df,
                    use_container_width=True,
                    hide_index=True
                )
                st.subheader("⭐ 오늘의 단기매매 TOP 5")

                top5_df = result_df.head(5)

                st.dataframe(top5_df, use_container_width=True, hide_index=True)

            else:
                st.warning("분석 가능한 종목 데이터를 가져오지 못했습니다.")

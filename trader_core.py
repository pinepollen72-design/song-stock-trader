from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

TOKEN_DIR = Path(os.getenv("SONG_TRADER_STATE_DIR", "/tmp/song_trader"))
TOKEN_DIR.mkdir(parents=True, exist_ok=True)
TRADE_LOG = TOKEN_DIR / "trade_log.csv"

@dataclass
class Settings:
    paper_app_key: str
    paper_app_secret: str
    paper_account_no: str
    paper_account_product_code: str
    live_app_key: str = ""
    live_app_secret: str = ""
    live_account_no: str = ""
    live_account_product_code: str = "01"
    allow_live: bool = False
    live_unlock_phrase: str = "LIVE-TRADING-UNLOCK"

    @classmethod
    def from_streamlit(cls, secrets):
        def g(name, default=""):
            try:
                return str(secrets.get(name, default))
            except Exception:
                return default
        def gb(name, default=False):
            v = g(name, str(default))
            return str(v).lower() in ("1","true","yes","on")
        return cls(
            paper_app_key=g("KIS_PAPER_APP_KEY", g("KIS_APP_KEY")),
            paper_app_secret=g("KIS_PAPER_APP_SECRET", g("KIS_APP_SECRET")),
            paper_account_no=g("KIS_PAPER_ACCOUNT_NO", g("KIS_ACCOUNT_NO")),
            paper_account_product_code=g("KIS_PAPER_ACCOUNT_PRODUCT_CODE", g("KIS_ACCOUNT_PRODUCT_CODE","01")),
            live_app_key=g("KIS_LIVE_APP_KEY"),
            live_app_secret=g("KIS_LIVE_APP_SECRET"),
            live_account_no=g("KIS_LIVE_ACCOUNT_NO"),
            live_account_product_code=g("KIS_LIVE_ACCOUNT_PRODUCT_CODE","01"),
            allow_live=gb("ALLOW_LIVE_TRADING", False),
            live_unlock_phrase=g("LIVE_UNLOCK_PHRASE","LIVE-TRADING-UNLOCK"),
        )

    @classmethod
    def from_env(cls):
        def g(name, default=""):
            return os.getenv(name, default)
        def gb(name, default=False):
            return g(name, str(default)).lower() in ("1","true","yes","on")
        return cls(
            paper_app_key=g("KIS_PAPER_APP_KEY", g("KIS_APP_KEY")),
            paper_app_secret=g("KIS_PAPER_APP_SECRET", g("KIS_APP_SECRET")),
            paper_account_no=g("KIS_PAPER_ACCOUNT_NO", g("KIS_ACCOUNT_NO")),
            paper_account_product_code=g("KIS_PAPER_ACCOUNT_PRODUCT_CODE", g("KIS_ACCOUNT_PRODUCT_CODE","01")),
            live_app_key=g("KIS_LIVE_APP_KEY"),
            live_app_secret=g("KIS_LIVE_APP_SECRET"),
            live_account_no=g("KIS_LIVE_ACCOUNT_NO"),
            live_account_product_code=g("KIS_LIVE_ACCOUNT_PRODUCT_CODE","01"),
            allow_live=gb("ALLOW_LIVE_TRADING", False),
            live_unlock_phrase=g("LIVE_UNLOCK_PHRASE","LIVE-TRADING-UNLOCK"),
        )

class KISClient:
    def __init__(self, settings: Settings, env: str = "demo"):
        if env not in ("demo", "real"):
            raise ValueError("env must be demo or real")
        self.settings = settings
        self.env = env
        if env == "demo":
            self.base_url = "https://openapivts.koreainvestment.com:29443"
            self.app_key = settings.paper_app_key
            self.app_secret = settings.paper_app_secret
            self.account_no = settings.paper_account_no
            self.product_code = settings.paper_account_product_code
        else:
            self.base_url = "https://openapi.koreainvestment.com:9443"
            self.app_key = settings.live_app_key
            self.app_secret = settings.live_app_secret
            self.account_no = settings.live_account_no
            self.product_code = settings.live_account_product_code

        if not self.app_key or not self.app_secret:
            raise ValueError("해당 운용모드의 App Key/App Secret이 없습니다.")

    @property
    def token_file(self) -> Path:
        return TOKEN_DIR / f"token_{self.env}.json"

    def get_token(self) -> str:
        now = time.time()
        if self.token_file.exists():
            try:
                saved = json.loads(self.token_file.read_text())
                if saved.get("token") and float(saved.get("expires_at",0)) > now + 300:
                    return saved["token"]
            except Exception:
                pass

        url = f"{self.base_url}/oauth2/tokenP"
        body = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
        }
        r = requests.post(url, headers={"content-type":"application/json"},
                          data=json.dumps(body), timeout=15)
        r.raise_for_status()
        data = r.json()
        token = data["access_token"]

        # 공식 응답 만료시각 문자열을 우선 사용하되, 파싱 실패 시 보수적으로 23시간 캐시
        expires_at = now + 23 * 3600
        raw_exp = data.get("access_token_token_expired")
        if raw_exp:
            try:
                dt = datetime.strptime(raw_exp, "%Y-%m-%d %H:%M:%S")
                expires_at = dt.timestamp()
            except Exception:
                pass

        self.token_file.write_text(json.dumps({
            "token": token, "expires_at": expires_at
        }))
        return token

    def _headers(self, tr_id: str) -> Dict[str,str]:
        return {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self.get_token()}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id,
            "custtype": "P",
        }

    def get(self, path: str, tr_id: str, params: Dict[str,Any]) -> Dict[str,Any]:
        r = requests.get(f"{self.base_url}{path}",
                         headers=self._headers(tr_id), params=params, timeout=15)
        r.raise_for_status()
        return r.json()

    def post(self, path: str, tr_id: str, body: Dict[str,Any]) -> Dict[str,Any]:
        r = requests.post(f"{self.base_url}{path}",
                          headers=self._headers(tr_id),
                          data=json.dumps(body), timeout=15)
        r.raise_for_status()
        return r.json()

    def domestic_price(self, code: str) -> Dict[str,Any]:
        return self.get(
            "/uapi/domestic-stock/v1/quotations/inquire-price",
            "FHKST01010100",
            {"FID_COND_MRKT_DIV_CODE":"J","FID_INPUT_ISCD":code}
        )

    def domestic_volume_rank(self) -> Dict[str,Any]:
        return self.get(
            "/uapi/domestic-stock/v1/quotations/volume-rank",
            "FHPST01710000",
            {
                "FID_COND_MRKT_DIV_CODE":"J",
                "FID_COND_SCR_DIV_CODE":"20171",
                "FID_INPUT_ISCD":"0000",
                "FID_DIV_CLS_CODE":"0",
                "FID_BLNG_CLS_CODE":"3",  # 거래금액순
                "FID_TRGT_CLS_CODE":"111111111",
                "FID_TRGT_EXLS_CLS_CODE":"0000000000",
                "FID_INPUT_PRICE_1":"0",
                "FID_INPUT_PRICE_2":"1000000",
                "FID_VOL_CNT":"100000",
                "FID_INPUT_DATE_1":"",
            }
        )

    def domestic_order(self, code: str, qty: int, side: str,
                       price: int = 0, market_order: bool = True) -> Dict[str,Any]:
        if side not in ("buy","sell"):
            raise ValueError("side must be buy/sell")
        if self.env == "demo":
            tr_id = "VTTC0012U" if side == "buy" else "VTTC0011U"
        else:
            tr_id = "TTTC0012U" if side == "buy" else "TTTC0011U"

        body = {
            "CANO": self.account_no,
            "ACNT_PRDT_CD": self.product_code,
            "PDNO": code,
            "ORD_DVSN": "01" if market_order else "00",
            "ORD_QTY": str(int(qty)),
            "ORD_UNPR": "0" if market_order else str(int(price)),
            "EXCG_ID_DVSN_CD":"KRX",
            "SLL_TYPE":"01" if side == "sell" else "",
            "CNDT_PRIC":"",
        }
        return self.post("/uapi/domestic-stock/v1/trading/order-cash", tr_id, body)

    def overseas_order_us(self, symbol: str, qty: int, side: str,
                          limit_price: float, exchange: str = "NASD") -> Dict[str,Any]:
        # 한국투자 공식 샘플 기준: 미국 모의주문은 지정가(00)만 사용
        if exchange not in ("NASD","NYSE","AMEX"):
            raise ValueError("US exchange must be NASD/NYSE/AMEX")
        if side not in ("buy","sell"):
            raise ValueError("side must be buy/sell")

        if side == "buy":
            tr_id = "TTTT1002U"
        else:
            tr_id = "TTTT1006U"

        if self.env == "demo":
            # 공식 샘플은 모의투자 시 T→V 변환.
            # 미국 매도는 문서상 VTTT1001U가 사용됨.
            tr_id = "VTTT1002U" if side == "buy" else "VTTT1001U"

        body = {
            "CANO": self.account_no,
            "ACNT_PRDT_CD": self.product_code,
            "OVRS_EXCG_CD": exchange,
            "PDNO": symbol,
            "ORD_QTY": str(int(qty)),
            "OVRS_ORD_UNPR": f"{float(limit_price):.2f}",
            "CTAC_TLNO":"",
            "MGCO_APTM_ODNO":"",
            "SLL_TYPE":"00" if side == "sell" else "",
            "ORD_SVR_DVSN_CD":"0",
            "ORD_DVSN":"00",
        }
        return self.post("/uapi/overseas-stock/v1/trading/order", tr_id, body)

def discover_domestic_candidates(client: KISClient, top_n: int = 20) -> pd.DataFrame:
    raw = client.domestic_volume_rank()
    rows = raw.get("output", []) or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)

    def pick(*names):
        for n in names:
            if n in df.columns:
                return n
        return None

    code_col = pick("mksc_shrn_iscd","stck_shrn_iscd")
    name_col = pick("hts_kor_isnm","prdt_name")
    price_col = pick("stck_prpr")
    change_col = pick("prdy_ctrt","prdy_vrss_sign")
    volume_col = pick("acml_vol")
    amount_col = pick("acml_tr_pbmn","acml_tr_amt")

    out = pd.DataFrame()
    if code_col: out["종목코드"] = df[code_col].astype(str).str.zfill(6)
    if name_col: out["종목명"] = df[name_col]
    if price_col: out["현재가"] = pd.to_numeric(df[price_col], errors="coerce")
    if change_col:
        out["등락률"] = pd.to_numeric(df[change_col], errors="coerce")
    if volume_col:
        out["누적거래량"] = pd.to_numeric(df[volume_col], errors="coerce")
    if amount_col:
        out["거래대금"] = pd.to_numeric(df[amount_col], errors="coerce")

    # 위험/정리매매 등을 후보에서 자동 제외하기 위한 추가 필터는
    # KIS rank API 파라미터/후속 마스터 정보 기반으로 확장 가능.
    return out.head(top_n)

def _download_yf(symbol: str, market: str):
    import yfinance as yf
    if market == "국내":
        # 먼저 KOSPI, 실패하면 KOSDAQ
        for suffix in (".KS",".KQ"):
            df = yf.download(symbol + suffix, period="5d", interval="5m",
                             auto_adjust=False, progress=False)
            if df is not None and not df.empty:
                return df
        return None
    else:
        return yf.download(symbol, period="5d", interval="5m",
                           auto_adjust=False, progress=False)

def _flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

def indicators(df, bb_period=20, bb_std=2.0, rsi_period=14, vol_period=20):
    d = _flatten(df.copy())
    close = d["Close"].astype(float)
    high = d["High"].astype(float)
    low = d["Low"].astype(float)
    open_ = d["Open"].astype(float)
    volume = d["Volume"].astype(float)

    d["MA20"] = close.rolling(bb_period).mean()
    d["STD20"] = close.rolling(bb_period).std()
    d["BB_UPPER"] = d["MA20"] + bb_std*d["STD20"]
    d["BB_LOWER"] = d["MA20"] - bb_std*d["STD20"]

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(rsi_period).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    rs = gain / loss.replace(0, np.nan)
    d["RSI"] = 100 - 100/(1+rs)

    d["VOL_MA"] = volume.rolling(vol_period).mean()
    d["VOL_RATIO"] = volume / d["VOL_MA"]
    d["MA5"] = close.rolling(5).mean()

    d["BODY"] = (close-open_).abs()
    d["RANGE"] = (high-low).replace(0,np.nan)
    d["BULL"] = close > open_
    d["BEAR"] = close < open_
    return d.dropna()

def score_latest(d):
    if d is None or len(d) < 2:
        return None
    x, prev = d.iloc[-1], d.iloc[-2]
    buy = sell = 0

    if x["Close"] <= x["BB_LOWER"]*1.01: buy += 2
    if prev["Close"] < prev["BB_LOWER"] and x["Close"] > x["BB_LOWER"]: buy += 2
    if x["Close"] >= x["BB_UPPER"]*0.99: sell += 2
    if prev["Close"] > prev["BB_UPPER"] and x["Close"] < x["BB_UPPER"]: sell += 2

    if 30 <= x["RSI"] <= 40 and x["RSI"] > prev["RSI"]: buy += 2
    elif x["RSI"] < 30: buy += 1
    if 60 <= x["RSI"] <= 70 and x["RSI"] < prev["RSI"]: sell += 2
    elif x["RSI"] > 70: sell += 1

    if x["VOL_RATIO"] >= 1.5 and x["BULL"]: buy += 2
    if x["VOL_RATIO"] >= 1.5 and x["BEAR"]: sell += 2

    if x["BULL"] and x["BODY"]/x["RANGE"] >= 0.6: buy += 1
    if x["BEAR"] and x["BODY"]/x["RANGE"] >= 0.6: sell += 1

    if x["MA5"] > x["MA20"]: buy += 1
    if x["MA5"] < x["MA20"]: sell += 1

    net = buy - sell
    signal = "🟢 매수 후보" if net >= 4 else "🔴 매도 후보" if net <= -4 else "🟡 관망"
    return buy, sell, net, signal, x

def score_ticker(symbol: str, market: str):
    df = _download_yf(symbol, market)
    if df is None or len(df) < 30:
        return None
    d = indicators(df)
    s = score_latest(d)
    if not s:
        return None
    buy, sell, net, signal, x = s
    return {
        "종목": symbol,
        "현재가": round(float(x["Close"]), 2),
        "RSI": round(float(x["RSI"]), 1),
        "거래량배수": round(float(x["VOL_RATIO"]), 2),
        "매수점수": int(buy),
        "매도점수": int(sell),
        "순점수": int(net),
        "종합신호": signal,
    }

def split_budget(total: int, parts: List[int]) -> List[int]:
    if sum(parts) <= 0:
        return [0 for _ in parts]
    return [int(total * p / sum(parts)) for p in parts]

def is_market_open(market: str) -> bool:
    now = datetime.now(ZoneInfo("Asia/Seoul" if market == "KR" else "America/New_York"))
    if now.weekday() >= 5:
        return False
    t = now.time()
    if market == "KR":
        return dtime(9,0) <= t < dtime(15,30)
    return dtime(9,30) <= t < dtime(16,0)

def market_force_exit_time(market: str) -> str:
    # 기본 안전값. 휴장/단축장/시간외 거래는 별도 캘린더 모듈로 확장 가능.
    return "15:15 KST" if market == "KR" else "15:50 ET"

def append_trade_log(row: Dict[str,Any]):
    df = pd.DataFrame([row])
    if TRADE_LOG.exists():
        df.to_csv(TRADE_LOG, mode="a", header=False, index=False)
    else:
        df.to_csv(TRADE_LOG, index=False)

def load_trade_log() -> pd.DataFrame:
    if not TRADE_LOG.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(TRADE_LOG)
    except Exception:
        return pd.DataFrame()

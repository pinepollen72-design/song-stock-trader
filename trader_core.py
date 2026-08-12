from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import requests
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")
ET = ZoneInfo("America/New_York")

STATE_DIR = Path(os.getenv("SONG_TRADER_STATE_DIR", "/tmp/song_trader"))
STATE_DIR.mkdir(parents=True, exist_ok=True)
TRADE_LOG = STATE_DIR / "trade_log.csv"

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

@dataclass
class Settings:
    paper_app_key: str = ""
    paper_app_secret: str = ""
    paper_account_no: str = ""
    paper_account_product_code: str = "01"

    live_app_key: str = ""
    live_app_secret: str = ""
    live_account_no: str = ""
    live_account_product_code: str = "01"

    allow_live: bool = False
    live_unlock_phrase: str = "I-UNDERSTAND-LIVE-ORDERS"

    @classmethod
    def from_env(cls) -> "Settings":
        g = os.getenv
        return cls(
            paper_app_key=g("KIS_PAPER_APP_KEY", g("KIS_APP_KEY", "")),
            paper_app_secret=g("KIS_PAPER_APP_SECRET", g("KIS_APP_SECRET", "")),
            paper_account_no=g("KIS_PAPER_ACCOUNT_NO", g("KIS_ACCOUNT_NO", "")),
            paper_account_product_code=g(
                "KIS_PAPER_ACCOUNT_PRODUCT_CODE",
                g("KIS_ACCOUNT_PRODUCT_CODE", "01"),
            ),
            live_app_key=g("KIS_LIVE_APP_KEY", ""),
            live_app_secret=g("KIS_LIVE_APP_SECRET", ""),
            live_account_no=g("KIS_LIVE_ACCOUNT_NO", ""),
            live_account_product_code=g("KIS_LIVE_ACCOUNT_PRODUCT_CODE", "01"),
            allow_live=g("ALLOW_LIVE", "false").lower() in ("1", "true", "yes", "on"),
            live_unlock_phrase=g(
                "LIVE_UNLOCK_PHRASE",
                "I-UNDERSTAND-LIVE-ORDERS",
            ),
        )


# ---------------------------------------------------------------------
# KIS client
# ---------------------------------------------------------------------

class KISClient:
    def __init__(self, settings: Settings, env: str = "demo"):
        self.settings = settings
        self.env = "demo" if str(env).lower() != "real" else "real"

        if self.env == "demo":
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

        self._token: Optional[str] = None
        self._token_time: float = 0.0

    def get_token(self, force: bool = False) -> str:
        if self._token and not force and time.time() - self._token_time < 60 * 50:
            return self._token

        if not self.app_key or not self.app_secret:
            raise RuntimeError("KIS APP KEY/SECRET 환경변수가 비어 있습니다.")

        url = f"{self.base_url}/oauth2/tokenP"
        body = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
        }

        r = requests.post(
            url,
            headers={"content-type": "application/json"},
            json=body,
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()

        token = data.get("access_token")
        if not token:
            raise RuntimeError(f"KIS 토큰 발급 실패: {data}")

        self._token = token
        self._token_time = time.time()
        return token

    def _headers(self, tr_id: str) -> Dict[str, str]:
        return {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self.get_token()}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id,
            "custtype": "P",
        }

    def get(self, path: str, tr_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.get(
            f"{self.base_url}{path}",
            headers=self._headers(tr_id),
            params=params,
            timeout=20,
        )
        r.raise_for_status()
        return r.json()

    def post(self, path: str, tr_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(
            f"{self.base_url}{path}",
            headers=self._headers(tr_id),
            json=body,
            timeout=20,
        )
        r.raise_for_status()
        return r.json()

    def domestic_price(self, code: str) -> Dict[str, Any]:
        return self.get(
            "/uapi/domestic-stock/v1/quotations/inquire-price",
            "FHKST01010100",
            {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": str(code).zfill(6)},
        )

    def domestic_balance(self) -> Dict[str, Any]:
        tr_id = "VTTC8434R" if self.env == "demo" else "TTTC8434R"
        return self.get(
            "/uapi/domestic-stock/v1/trading/inquire-balance",
            tr_id,
            {
                "CANO": self.account_no,
                "ACNT_PRDT_CD": self.product_code,
                "AFHR_FLPR_YN": "N",
                "OFL_YN": "",
                "INQR_DVSN": "02",
                "UNPR_DVSN": "01",
                "FUND_STTL_ICLD_YN": "N",
                "FNCG_AMT_AUTO_RDPT_YN": "N",
                "PRCS_DVSN": "00",
                "CTX_AREA_FK100": "",
                "CTX_AREA_NK100": "",
            },
        )

    def domestic_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        market_order: bool = True,
        price: int = 0,
    ) -> Dict[str, Any]:
        side = str(side).lower()
        if side not in ("buy", "sell"):
            raise ValueError("side는 buy 또는 sell 이어야 합니다.")

        if self.env == "demo":
            tr_id = "VTTC0012U" if side == "buy" else "VTTC0011U"
        else:
            tr_id = "TTTC0012U" if side == "buy" else "TTTC0011U"

        ord_dvsn = "01" if market_order else "00"
        ord_unpr = "0" if market_order else str(int(price))

        return self.post(
            "/uapi/domestic-stock/v1/trading/order-cash",
            tr_id,
            {
                "CANO": self.account_no,
                "ACNT_PRDT_CD": self.product_code,
                "PDNO": str(symbol).zfill(6),
                "ORD_DVSN": ord_dvsn,
                "ORD_QTY": str(int(qty)),
                "ORD_UNPR": ord_unpr,
            },
        )

    def overseas_order_us(
        self,
        symbol: str,
        qty: int,
        side: str,
        limit_price: float,
        exchange: str = "NASD",
    ) -> Dict[str, Any]:
        side = str(side).lower()
        if self.env == "demo":
            tr_id = "VTTT1002U" if side == "buy" else "VTTT1006U"
        else:
            tr_id = "TTTT1002U" if side == "buy" else "TTTT1006U"

        return self.post(
            "/uapi/overseas-stock/v1/trading/order",
            tr_id,
            {
                "CANO": self.account_no,
                "ACNT_PRDT_CD": self.product_code,
                "OVRS_EXCG_CD": exchange,
                "PDNO": str(symbol).upper(),
                "ORD_QTY": str(int(qty)),
                "OVRS_ORD_UNPR": f"{float(limit_price):.4f}",
                "ORD_SVR_DVSN_CD": "0",
                "ORD_DVSN": "00",
            },
        )


# ---------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------

def _empty_price_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])


def _download_yf(symbol: str, market: str = "국내") -> pd.DataFrame:
    """
    Yahoo 데이터 오류가 나도 예외를 밖으로 던지지 않습니다.
    국내는 .KS -> .KQ 순서로 시도하고 둘 다 실패하면 빈 DataFrame 반환.
    이 함수 때문에 worker 전체가 멈추지 않도록 설계했습니다.
    """
    try:
        import yfinance as yf
    except Exception:
        return _empty_price_df()

    raw_symbol = str(symbol).strip()

    if market == "국내":
        code = raw_symbol.zfill(6)
        tickers = [f"{code}.KS", f"{code}.KQ"]
        period = "5d"
        interval = "5m"
    else:
        tickers = [raw_symbol.upper()]
        period = "1mo"
        interval = "30m"

    for ticker in tickers:
        try:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception:
            continue

        if df is None or df.empty:
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        needed = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        if "Close" not in needed or "Volume" not in needed:
            continue

        out = df[needed].dropna(subset=["Close"]).copy()
        if not out.empty:
            out.attrs["source_ticker"] = ticker
            return out

    return _empty_price_df()


def score_ticker(symbol: str, market: str = "국내") -> Optional[Dict[str, Any]]:
    """
    데이터가 없으면 None을 반환합니다.
    yfinance 오류가 전체 스캔을 중단시키지 않습니다.
    """
    df = _download_yf(symbol, market)
    if df is None or df.empty or len(df) < 20:
        return None

    close = pd.to_numeric(df["Close"], errors="coerce")
    vol = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    rsi_last = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else 50.0

    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20

    vol_avg = vol.rolling(20).mean()
    vol_ratio = float(vol.iloc[-1] / vol_avg.iloc[-1]) if vol_avg.iloc[-1] else 0.0

    buy_score = 0
    sell_score = 0

    if rsi_last <= 35:
        buy_score += 2
    elif rsi_last <= 45:
        buy_score += 1
    elif rsi_last >= 70:
        sell_score += 2
    elif rsi_last >= 60:
        sell_score += 1

    if pd.notna(lower.iloc[-1]) and close.iloc[-1] <= lower.iloc[-1]:
        buy_score += 2
    if pd.notna(upper.iloc[-1]) and close.iloc[-1] >= upper.iloc[-1]:
        sell_score += 2

    if vol_ratio >= 1.5:
        buy_score += 1

    net = int(buy_score - sell_score)

    if net >= 2:
        signal = "🟢 매수 후보"
    elif net <= -2:
        signal = "🔴 매도 주의"
    else:
        signal = "⚪ 중립"

    return {
        "종목": str(symbol),
        "RSI": round(rsi_last, 1),
        "거래량배수": round(vol_ratio, 2),
        "매수점수": int(buy_score),
        "매도점수": int(sell_score),
        "순점수": net,
        "종합신호": signal,
    }


def discover_domestic_candidates(client: KISClient, top_n: int = 20) -> pd.DataFrame:
    """
    KIS 거래대금 상위 후보를 가져옵니다.
    응답형식이 달라지거나 일부 행이 이상해도 가능한 행만 살립니다.
    """
    raw = client.get(
        "/uapi/domestic-stock/v1/quotations/volume-rank",
        "FHPST01710000",
        {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_COND_SCR_DIV_CODE": "20171",
            "FID_INPUT_ISCD": "0000",
            "FID_DIV_CLS_CODE": "0",
            "FID_BLNG_CLS_CODE": "0",
            "FID_TRGT_CLS_CODE": "111111111",
            "FID_TRGT_EXLS_CLS_CODE": "000000",
            "FID_INPUT_PRICE_1": "1000",
            "FID_INPUT_PRICE_2": "",
            "FID_VOL_CNT": "100000",
            "FID_INPUT_DATE_1": "",
        },
    )

    rows = (raw or {}).get("output", []) or []
    out = []

    for r in rows:
        code = str(
            r.get("mksc_shrn_iscd")
            or r.get("stck_shrn_iscd")
            or r.get("pdno")
            or ""
        ).strip()

        if not (len(code) == 6 and code.isdigit()):
            continue

        name = str(
            r.get("hts_kor_isnm")
            or r.get("prdt_name")
            or r.get("stck_name")
            or ""
        ).strip()

        try:
            price = abs(float(r.get("stck_prpr", 0) or 0))
        except Exception:
            price = 0.0

        try:
            change = float(r.get("prdy_ctrt", 0) or 0)
        except Exception:
            change = 0.0

        try:
            volume = float(r.get("acml_vol", 0) or 0)
        except Exception:
            volume = 0.0

        try:
            amount = float(r.get("acml_tr_pbmn", 0) or 0)
        except Exception:
            amount = 0.0

        if price < 1000 or volume < 100000:
            continue
        if change < -10 or change > 20:
            continue

        # 아주 단순한 리더점수: 거래대금/등락률/거래량을 정규화 없이 압축
        lead = min(100.0, max(0.0, 40 + change * 2 + min(volume / 1_000_000, 20)))

        out.append({
            "종목코드": code,
            "종목명": name,
            "현재가": int(price),
            "등락률": round(change, 2),
            "누적거래량": int(volume),
            "거래대금": int(amount),
            "주도주점수": round(lead, 1),
        })

        if len(out) >= top_n:
            break

    return pd.DataFrame(out)


# ---------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------

def split_budget(total: int, pcts) -> list[int]:
    pcts = [int(x) for x in pcts]
    s = max(1, sum(pcts))
    parts = [int(total * p / s) for p in pcts]
    if parts:
        parts[-1] += int(total) - sum(parts)
    return parts


def is_market_open(market: str) -> bool:
    if str(market).upper() == "KR":
        now = datetime.now(KST)
        return now.weekday() < 5 and dtime(9, 0) <= now.time() < dtime(15, 30)

    now = datetime.now(ET)
    return now.weekday() < 5 and dtime(9, 30) <= now.time() < dtime(16, 0)


def market_force_exit_time(market: str) -> str:
    return "15:15 KST" if str(market).upper() == "KR" else "15:50 ET"


def append_trade_log(row: Dict[str, Any]) -> None:
    try:
        df = pd.DataFrame([row])
        header = not TRADE_LOG.exists()
        df.to_csv(
            TRADE_LOG,
            mode="a",
            index=False,
            header=header,
            encoding="utf-8-sig",
        )
    except Exception:
        pass


def load_trade_log() -> pd.DataFrame:
    if not TRADE_LOG.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(TRADE_LOG)
    except Exception:
        return pd.DataFrame()

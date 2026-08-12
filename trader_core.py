from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import requests


KST = ZoneInfo("Asia/Seoul")
ET = ZoneInfo("America/New_York")

STATE_DIR = Path(os.getenv("SONG_TRADER_STATE_DIR", "/tmp/song_trader"))
STATE_DIR.mkdir(parents=True, exist_ok=True)
TRADE_LOG_FILE = STATE_DIR / "trade_log.jsonl"


# =========================================================
# 환경 설정
# =========================================================
def _env(name: str, default: str = "") -> str:
    return str(os.getenv(name, default) or "").strip()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env(name, "true" if default else "false").lower()
    return raw in ("1", "true", "yes", "on")


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
        return cls(
            paper_app_key=_env("KIS_PAPER_APP_KEY", _env("KIS_APP_KEY")),
            paper_app_secret=_env("KIS_PAPER_APP_SECRET", _env("KIS_APP_SECRET")),
            paper_account_no=_env("KIS_PAPER_ACCOUNT_NO", _env("KIS_ACCOUNT_NO")),
            paper_account_product_code=_env(
                "KIS_PAPER_ACCOUNT_PRODUCT_CODE",
                _env("KIS_ACCOUNT_PRODUCT_CODE", "01"),
            ),
            live_app_key=_env("KIS_LIVE_APP_KEY"),
            live_app_secret=_env("KIS_LIVE_APP_SECRET"),
            live_account_no=_env("KIS_LIVE_ACCOUNT_NO"),
            live_account_product_code=_env(
                "KIS_LIVE_ACCOUNT_PRODUCT_CODE",
                "01",
            ),
            allow_live=_env_bool("ALLOW_LIVE_TRADING", False),
            live_unlock_phrase=_env(
                "LIVE_UNLOCK_PHRASE",
                "I-UNDERSTAND-LIVE-ORDERS",
            ),
        )

    @classmethod
    def from_streamlit(cls) -> "Settings":
        try:
            import streamlit as st

            def sget(name: str, default: str = "") -> str:
                try:
                    value = st.secrets.get(name, default)
                except Exception:
                    value = default
                if value is None:
                    value = default
                return str(value).strip()

            def sbool(name: str, default: bool = False) -> bool:
                return sget(name, "true" if default else "false").lower() in (
                    "1",
                    "true",
                    "yes",
                    "on",
                )

            return cls(
                paper_app_key=sget("KIS_PAPER_APP_KEY", sget("KIS_APP_KEY")),
                paper_app_secret=sget(
                    "KIS_PAPER_APP_SECRET",
                    sget("KIS_APP_SECRET"),
                ),
                paper_account_no=sget(
                    "KIS_PAPER_ACCOUNT_NO",
                    sget("KIS_ACCOUNT_NO"),
                ),
                paper_account_product_code=sget(
                    "KIS_PAPER_ACCOUNT_PRODUCT_CODE",
                    sget("KIS_ACCOUNT_PRODUCT_CODE", "01"),
                ),
                live_app_key=sget("KIS_LIVE_APP_KEY"),
                live_app_secret=sget("KIS_LIVE_APP_SECRET"),
                live_account_no=sget("KIS_LIVE_ACCOUNT_NO"),
                live_account_product_code=sget(
                    "KIS_LIVE_ACCOUNT_PRODUCT_CODE",
                    "01",
                ),
                allow_live=sbool("ALLOW_LIVE_TRADING", False),
                live_unlock_phrase=sget(
                    "LIVE_UNLOCK_PHRASE",
                    "I-UNDERSTAND-LIVE-ORDERS",
                ),
            )
        except Exception:
            return cls.from_env()


# =========================================================
# KIS API 클라이언트
# =========================================================
class KISClient:
    """
    한국투자증권 REST API 공통 클라이언트.

    이번 교체본의 핵심:
    - 토큰 요청 read timeout 기본 45초
    - 일반 API 요청 timeout 기본 30초
    - 일반 조회 API는 일시적 네트워크 오류 시 최대 3회 재시도
    - 토큰은 과도한 재발급을 피하기 위해 보수적으로 재시도
    - 발급된 토큰은 메모리에 캐시
    """

    def __init__(self, settings: Settings, env: str = "demo"):
        self.settings = settings
        self.env = str(env or "demo").strip().lower()
        if self.env not in ("demo", "real"):
            self.env = "demo"

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
        self._token_expires_at: float = 0.0

        token_cache_name = (
            "kis_token_demo.json"
            if self.env == "demo"
            else "kis_token_real.json"
        )
        self._token_cache_file = STATE_DIR / token_cache_name

        self.token_timeout = float(
            os.getenv("KIS_TOKEN_TIMEOUT_SECONDS", "45")
        )
        self.api_timeout = float(
            os.getenv("KIS_API_TIMEOUT_SECONDS", "30")
        )
        self.api_retries = max(
            1,
            int(os.getenv("KIS_API_RETRIES", "3")),
        )

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "song-stock-trader/1.0",
                "Accept": "application/json",
            }
        )

    def _validate_credentials(self) -> None:
        missing = []
        if not self.app_key:
            missing.append("APP_KEY")
        if not self.app_secret:
            missing.append("APP_SECRET")
        if missing:
            raise RuntimeError(
                f"KIS 인증정보가 비어 있습니다: {', '.join(missing)}"
            )

    def _validate_account(self) -> None:
        if not self.account_no:
            raise RuntimeError("KIS 계좌번호가 비어 있습니다.")
        if not self.product_code:
            raise RuntimeError("KIS 계좌상품코드가 비어 있습니다.")

    def _load_cached_token(self) -> bool:
        try:
            if not self._token_cache_file.exists():
                return False

            data = json.loads(
                self._token_cache_file.read_text(encoding="utf-8")
            )
            token = str(data.get("access_token", "") or "").strip()
            expires_at = float(data.get("expires_at", 0) or 0)

            if token and time.time() < expires_at - 120:
                self._token = token
                self._token_expires_at = expires_at
                return True
        except Exception:
            return False

        return False

    def _save_cached_token(self) -> None:
        if not self._token:
            return
        try:
            self._token_cache_file.write_text(
                json.dumps(
                    {
                        "access_token": self._token,
                        "expires_at": self._token_expires_at,
                        "env": self.env,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

    def get_token(self, force: bool = False) -> str:
        self._validate_credentials()

        now = time.time()

        # 같은 프로세스에서는 기존 토큰 재사용
        if (
            not force
            and self._token
            and now < self._token_expires_at - 120
        ):
            return self._token

        # Render worker 재시작 시 같은 인스턴스의 /tmp 캐시 재사용
        if not force and self._load_cached_token():
            return self._token

        url = f"{self.base_url}/oauth2/tokenP"
        payload = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
        }

        rate_limit_attempts = 0
        network_attempts = 0

        while True:
            try:
                res = self.session.post(
                    url,
                    headers={
                        "content-type": "application/json; charset=UTF-8",
                    },
                    json=payload,
                    timeout=(10, self.token_timeout),
                )
            except requests.exceptions.Timeout as e:
                network_attempts += 1
                if network_attempts < self.api_retries:
                    time.sleep(min(2 ** network_attempts, 10))
                    continue
                raise RuntimeError(
                    "KIS 토큰 발급 시간이 초과되었습니다. "
                    f"(timeout={self.token_timeout:.0f}초)"
                ) from e
            except requests.exceptions.RequestException as e:
                network_attempts += 1
                if network_attempts < self.api_retries:
                    time.sleep(min(2 ** network_attempts, 10))
                    continue
                raise RuntimeError(
                    f"KIS 토큰 연결 오류: {type(e).__name__}: {e}"
                ) from e

            if res.status_code == 403:
                body = res.text[:1200]

                # 실제 확인된 KIS 접근토큰 1분당 1회 제한
                if "EGW00133" in body:
                    rate_limit_attempts += 1
                    if rate_limit_attempts <= 3:
                        time.sleep(65)
                        continue

                raise RuntimeError(
                    f"KIS 토큰 403 Forbidden: {body}"
                )

            if res.status_code >= 500:
                network_attempts += 1
                if network_attempts < self.api_retries:
                    time.sleep(min(3 * network_attempts, 15))
                    continue

            res.raise_for_status()

            data = res.json()
            token = str(data.get("access_token", "") or "").strip()

            if not token:
                raise RuntimeError(
                    f"KIS 토큰 응답에 access_token이 없습니다: {data}"
                )

            # 공식 문서상 24시간 유효. 응답 expires_in이 있으면 우선 사용.
            expires_in = data.get("expires_in", 24 * 3600)
            try:
                expires_in = int(expires_in)
            except Exception:
                expires_in = 24 * 3600

            # 너무 끝까지 사용하지 않고 약간 일찍 갱신
            self._token = token
            self._token_expires_at = time.time() + max(3600, expires_in)
            self._save_cached_token()
            return token

    def _headers(
        self,
        tr_id: str,
        custtype: str = "P",
        extra: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        token = self.get_token()

        headers = {
            "content-type": "application/json; charset=UTF-8",
            "authorization": f"Bearer {token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id,
            "custtype": custtype,
        }
        if extra:
            headers.update(extra)
        return headers

    def _request(
        self,
        method: str,
        path: str,
        tr_id: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        retries: Optional[int] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        timeout = float(timeout or self.api_timeout)
        retries = max(1, int(retries or self.api_retries))

        last_error: Optional[Exception] = None

        for attempt in range(1, retries + 1):
            try:
                headers = self._headers(tr_id, extra=extra_headers)

                if method.upper() == "GET":
                    res = self.session.get(
                        url,
                        headers=headers,
                        params=params or {},
                        timeout=(10, timeout),
                    )
                else:
                    res = self.session.post(
                        url,
                        headers=headers,
                        json=json_body or {},
                        timeout=(10, timeout),
                    )

                # 인증 만료 시에만 토큰을 비우고 다음 요청에서 갱신
                if res.status_code == 401:
                    self._token = None
                    self._token_expires_at = 0.0
                    try:
                        if self._token_cache_file.exists():
                            self._token_cache_file.unlink()
                    except Exception:
                        pass
                    if attempt < retries:
                        time.sleep(min(5 * attempt, 15))
                        continue

                res.raise_for_status()

                try:
                    return res.json()
                except Exception as e:
                    raise RuntimeError(
                        f"KIS 응답 JSON 해석 실패: {res.text[:1000]}"
                    ) from e

            except (
                requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectTimeout,
                requests.exceptions.ConnectionError,
            ) as e:
                last_error = e
                if attempt >= retries:
                    break
                time.sleep(min(2 ** attempt, 10))

            except requests.HTTPError as e:
                last_error = e
                # 429/5xx는 일시 오류로 보고 재시도
                status = getattr(e.response, "status_code", 0) or 0
                if status == 429 or status >= 500:
                    if attempt < retries:
                        time.sleep(min(3 * attempt, 15))
                        continue
                raise

            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt >= retries:
                    break
                time.sleep(min(2 ** attempt, 10))

        raise RuntimeError(
            f"KIS API 요청 실패({retries}회 시도): "
            f"{type(last_error).__name__ if last_error else 'UnknownError'}: "
            f"{last_error}"
        )

    def get(
        self,
        path: str,
        tr_id: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._request(
            "GET",
            path,
            tr_id,
            params=params,
        )

    def post(
        self,
        path: str,
        tr_id: str,
        body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._request(
            "POST",
            path,
            tr_id,
            json_body=body,
        )

    # -----------------------------------------------------
    # 국내 현재가
    # -----------------------------------------------------
    def domestic_price(self, code: str) -> Dict[str, Any]:
        return self.get(
            "/uapi/domestic-stock/v1/quotations/inquire-price",
            "FHKST01010100",
            {
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": str(code).zfill(6),
            },
        )

    # -----------------------------------------------------
    # 국내 잔고
    # -----------------------------------------------------
    def domestic_balance(self) -> Dict[str, Any]:
        self._validate_account()

        tr_id = (
            "VTTC8434R"
            if self.env == "demo"
            else "TTTC8434R"
        )

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


    # -----------------------------------------------------
    # 국내 매수가능금액 / 수량 조회
    # -----------------------------------------------------
    def domestic_buying_power(
        self,
        symbol: str,
        reference_price: int,
    ) -> Dict[str, Any]:
        self._validate_account()

        tr_id = (
            "VTTC8908R"
            if self.env == "demo"
            else "TTTC8908R"
        )

        return self.get(
            "/uapi/domestic-stock/v1/trading/inquire-psbl-order",
            tr_id,
            {
                "CANO": self.account_no,
                "ACNT_PRDT_CD": self.product_code,
                "PDNO": str(symbol).zfill(6),
                "ORD_UNPR": str(max(1, int(reference_price))),
                "ORD_DVSN": "01",
                "CMA_EVLU_AMT_ICLD_YN": "N",
                "OVRS_ICLD_YN": "N",
            },
        )

    # -----------------------------------------------------
    # 국내 현금주문
    # -----------------------------------------------------
    def domestic_order(
        self,
        code: str,
        qty: int,
        side: str,
        market_order: bool = True,
        price: int = 0,
    ) -> Dict[str, Any]:
        self._validate_account()

        side = str(side).lower().strip()
        if side not in ("buy", "sell"):
            raise ValueError("side는 buy 또는 sell 이어야 합니다.")

        if self.env == "demo":
            tr_id = "VTTC0012U" if side == "buy" else "VTTC0011U"
        else:
            tr_id = "TTTC0012U" if side == "buy" else "TTTC0011U"

        ord_dvsn = "01" if market_order else "00"
        ord_unpr = "0" if market_order else str(int(price))

        body = {
            "CANO": self.account_no,
            "ACNT_PRDT_CD": self.product_code,
            "PDNO": str(code).zfill(6),
            "ORD_DVSN": ord_dvsn,
            "ORD_QTY": str(int(qty)),
            "ORD_UNPR": ord_unpr,
        }

        return self.post(
            "/uapi/domestic-stock/v1/trading/order-cash",
            tr_id,
            body,
        )


    # -----------------------------------------------------
    # 미국 현재가
    # -----------------------------------------------------
    def overseas_price_us(
        self,
        symbol: str,
        exchange: str = "NASD",
    ) -> Dict[str, Any]:
        exchange = str(exchange or "NASD").upper()
        quote_exchange = {
            "NASD": "NAS",
            "NYSE": "NYS",
            "AMEX": "AMS",
        }.get(exchange, "NAS")

        return self.get(
            "/uapi/overseas-price/v1/quotations/price",
            "HHDFS00000300",
            {
                "AUTH": "",
                "EXCD": quote_exchange,
                "SYMB": str(symbol).upper(),
            },
        )

    # -----------------------------------------------------
    # 미국 잔고
    # -----------------------------------------------------
    def overseas_balance_us(
        self,
        exchange: str = "NASD",
    ) -> Dict[str, Any]:
        self._validate_account()

        tr_id = (
            "VTTS3012R"
            if self.env == "demo"
            else "TTTS3012R"
        )

        return self.get(
            "/uapi/overseas-stock/v1/trading/inquire-balance",
            tr_id,
            {
                "CANO": self.account_no,
                "ACNT_PRDT_CD": self.product_code,
                "OVRS_EXCG_CD": str(exchange or "NASD").upper(),
                "TR_CRCY_CD": "USD",
                "CTX_AREA_FK200": "",
                "CTX_AREA_NK200": "",
            },
        )

    # -----------------------------------------------------
    # 미국 지정가 주문
    # -----------------------------------------------------
    def overseas_order_us(
        self,
        symbol: str,
        qty: int,
        side: str,
        limit_price: float,
        exchange: str = "NASD",
    ) -> Dict[str, Any]:
        self._validate_account()

        side = str(side).lower().strip()
        if side not in ("buy", "sell"):
            raise ValueError("side는 buy 또는 sell 이어야 합니다.")

        exchange = str(exchange or "NASD").upper()

        # 프로젝트 기존 인터페이스와 호환되는 TR ID
        if self.env == "demo":
            tr_id = "VTTT1002U" if side == "buy" else "VTTT1006U"
        else:
            tr_id = "TTTT1002U" if side == "buy" else "TTTT1006U"

        body = {
            "CANO": self.account_no,
            "ACNT_PRDT_CD": self.product_code,
            "OVRS_EXCG_CD": exchange,
            "PDNO": str(symbol).upper(),
            "ORD_QTY": str(int(qty)),
            "OVRS_ORD_UNPR": f"{float(limit_price):.4f}".rstrip("0").rstrip("."),
            "ORD_SVR_DVSN_CD": "0",
            "ORD_DVSN": "00",
        }

        return self.post(
            "/uapi/overseas-stock/v1/trading/order",
            tr_id,
            body,
        )

    # -----------------------------------------------------
    # 국내 거래대금/등락률 후보용 랭킹
    # -----------------------------------------------------
    def domestic_volume_rank(self) -> Dict[str, Any]:
        return self.get(
            "/uapi/domestic-stock/v1/quotations/volume-rank",
            "FHPST01710000",
            {
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_COND_SCR_DIV_CODE": "20171",
                "FID_INPUT_ISCD": "0000",
                "FID_DIV_CLS_CODE": "0",
                "FID_BLNG_CLS_CODE": "0",
                "FID_TRGT_CLS_CODE": "111111111",
                "FID_TRGT_EXLS_CLS_CODE": "0000000000",
                "FID_INPUT_PRICE_1": "1000",
                "FID_INPUT_PRICE_2": "",
                "FID_VOL_CNT": "100000",
                "FID_INPUT_DATE_1": "",
            },
        )


# =========================================================
# 거래 로그
# =========================================================
def append_trade_log(row: Dict[str, Any]) -> None:
    payload = dict(row or {})
    payload.setdefault(
        "time",
        datetime.now(KST).isoformat(timespec="seconds"),
    )

    try:
        with TRADE_LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )
    except Exception:
        pass


def load_trade_log() -> pd.DataFrame:
    if not TRADE_LOG_FILE.exists():
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    try:
        for line in TRADE_LOG_FILE.read_text(
            encoding="utf-8"
        ).splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    except Exception:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# =========================================================
# 자금 분할
# =========================================================
def split_budget(
    total: int,
    weights: Iterable[int] = (40, 30, 30),
) -> List[int]:
    weights = [max(0, int(x)) for x in weights]
    if not weights:
        return []

    s = sum(weights)
    if s <= 0:
        return [0 for _ in weights]

    parts = [
        int(int(total) * w / s)
        for w in weights
    ]

    # 반올림 손실은 마지막 구간에 보정
    if parts:
        parts[-1] += int(total) - sum(parts)

    return parts


# =========================================================
# 시장 시간
# =========================================================
def is_market_open(market: str) -> bool:
    market = str(market).upper()

    if market in ("KR", "국내"):
        now = datetime.now(KST)
        if now.weekday() >= 5:
            return False
        return dtime(9, 0) <= now.time() < dtime(15, 30)

    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    return dtime(9, 30) <= now.time() < dtime(16, 0)


def market_force_exit_time(market: str) -> str:
    market = str(market).upper()
    if market in ("KR", "국내"):
        return "15:15 KST"
    return "15:50 ET"


# =========================================================
# yfinance 기반 기술데이터
# =========================================================
def _yf_symbol(symbol: str, market: str) -> str:
    market = str(market).lower()

    if market in ("국내", "kr", "korea"):
        code = str(symbol).zfill(6)

        # 기본은 KOSPI(.KS), 실패 시 score_ticker에서 .KQ도 재시도
        return f"{code}.KS"

    return str(symbol).upper()


def _download_yf(
    symbol: str,
    market: str,
    period: str = "5d",
    interval: str = "5m",
) -> pd.DataFrame:
    try:
        import yfinance as yf
    except Exception as e:
        raise RuntimeError(
            "yfinance가 설치되어 있지 않습니다."
        ) from e

    market_norm = str(market).lower()

    candidates = [_yf_symbol(symbol, market)]
    if market_norm in ("국내", "kr", "korea"):
        code = str(symbol).zfill(6)
        candidates = [f"{code}.KS", f"{code}.KQ"]

    last_df = pd.DataFrame()

    for ticker in candidates:
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

        keep = [
            c
            for c in ["Open", "High", "Low", "Close", "Volume"]
            if c in df.columns
        ]

        if not keep:
            continue

        last_df = df[keep].copy().dropna()
        if not last_df.empty:
            return last_df

    return last_df


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    avg_up = up.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_down = down.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = avg_up / avg_down.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def score_ticker(
    symbol: str,
    market: str = "국내",
) -> Optional[Dict[str, Any]]:
    df = _download_yf(symbol, market)

    if df is None or len(df) < 25:
        return None

    close = pd.to_numeric(df["Close"], errors="coerce")
    volume = pd.to_numeric(df["Volume"], errors="coerce")

    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    rsi = _rsi(close, 14)

    vol_avg20 = volume.rolling(20).mean()
    vol_ratio = volume.iloc[-1] / vol_avg20.iloc[-1] if vol_avg20.iloc[-1] else 0

    last = float(close.iloc[-1])
    prev = float(close.iloc[-2])
    rsi_last = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else 50.0

    buy_score = 0
    sell_score = 0
    reasons_buy: List[str] = []
    reasons_sell: List[str] = []

    if last > float(ma5.iloc[-1]) > float(ma20.iloc[-1]):
        buy_score += 2
        reasons_buy.append("단기 상승정렬")
    elif last < float(ma5.iloc[-1]) < float(ma20.iloc[-1]):
        sell_score += 2
        reasons_sell.append("단기 하락정렬")

    if rsi_last < 35:
        buy_score += 1
        reasons_buy.append("RSI 과매도권")
    elif rsi_last > 70:
        sell_score += 1
        reasons_sell.append("RSI 과열권")

    if pd.notna(lower.iloc[-1]) and last <= float(lower.iloc[-1]) * 1.01:
        buy_score += 1
        reasons_buy.append("볼린저 하단 근접")

    if pd.notna(upper.iloc[-1]) and last >= float(upper.iloc[-1]) * 0.995:
        sell_score += 1
        reasons_sell.append("볼린저 상단 근접")

    if vol_ratio >= 1.5 and last > prev:
        buy_score += 2
        reasons_buy.append("거래량 급증 상승")
    elif vol_ratio >= 1.5 and last < prev:
        sell_score += 2
        reasons_sell.append("거래량 급증 하락")

    net = int(buy_score - sell_score)

    if net >= 2:
        signal = "🟢 매수 후보"
    elif net <= -2:
        signal = "🔴 매도 주의"
    else:
        signal = "🟡 관망"

    return {
        "종목": str(symbol),
        "현재가": round(last, 4),
        "RSI": round(rsi_last, 1),
        "거래량배수": round(float(vol_ratio or 0), 2),
        "매수점수": int(buy_score),
        "매도점수": int(sell_score),
        "순점수": int(net),
        "종합신호": signal,
        "매수근거": ", ".join(reasons_buy),
        "매도근거": ", ".join(reasons_sell),
    }


# =========================================================
# 국내 후보 탐색
# =========================================================
def _safe_num(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(str(value).replace(",", ""))
    except Exception:
        return default


def _excluded_name(name: str) -> bool:
    name = str(name or "").strip()
    upper = name.upper()

    if not name:
        return True

    keywords = (
        "ETF",
        "ETN",
        "스팩",
        "SPAC",
        "리츠",
        "REIT",
    )
    if any(k in upper for k in keywords):
        return True

    # 우선주 간단 필터
    if name.endswith("우") or name.endswith("우B") or name.endswith("우C"):
        return True

    return False


def discover_domestic_candidates(
    client: KISClient,
    top_n: int = 20,
) -> pd.DataFrame:
    raw = client.domestic_volume_rank()
    rows = (raw or {}).get("output", []) or []

    if isinstance(rows, dict):
        rows = [rows]

    out: List[Dict[str, Any]] = []

    for r in rows:
        code = str(
            r.get("mksc_shrn_iscd")
            or r.get("stck_shrn_iscd")
            or r.get("pdno")
            or ""
        ).strip().zfill(6)

        name = str(
            r.get("hts_kor_isnm")
            or r.get("prdt_name")
            or ""
        ).strip()

        if not code or code == "000000":
            continue
        if _excluded_name(name):
            continue

        price = _safe_num(
            r.get("stck_prpr")
            or r.get("prpr")
        )

        change = _safe_num(
            r.get("prdy_ctrt")
            or r.get("prdy_vrss_sign")
        )

        volume = _safe_num(
            r.get("acml_vol")
            or r.get("acml_voln")
        )

        amount = _safe_num(
            r.get("acml_tr_pbmn")
            or r.get("acml_tr_amt")
        )

        if price < 1000:
            continue
        if volume < 100000:
            continue
        if change < -10 or change > 20:
            continue

        # 주도주점수: 거래대금 + 거래량 + 상승률을 0~100 범위로 단순 정규화
        amount_score = min(50.0, max(0.0, amount / 1_000_000_000 * 2.0))
        volume_score = min(25.0, max(0.0, volume / 1_000_000 * 5.0))
        change_score = min(25.0, max(0.0, change * 2.5))
        leader_score = min(100.0, amount_score + volume_score + change_score)

        out.append(
            {
                "종목코드": code,
                "종목명": name,
                "현재가": int(price),
                "등락률": round(change, 2),
                "누적거래량": int(volume),
                "거래대금": int(amount),
                "주도주점수": round(leader_score, 1),
            }
        )

    if not out:
        return pd.DataFrame(
            columns=[
                "종목코드",
                "종목명",
                "현재가",
                "등락률",
                "누적거래량",
                "거래대금",
                "주도주점수",
            ]
        )

    df = pd.DataFrame(out).sort_values(
        ["주도주점수", "거래대금", "누적거래량"],
        ascending=[False, False, False],
    )

    return df.head(int(top_n)).reset_index(drop=True)

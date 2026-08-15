from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

def _resolve_state_dir() -> Path:
    """Worker와 같은 규칙으로 영구 상태 저장소를 찾습니다.

    Railway Volume이 연결돼 있으면 RAILWAY_VOLUME_MOUNT_PATH 아래를 사용해
    토큰 캐시가 재배포/재시작 뒤에도 유지되도록 합니다.
    """
    explicit = os.getenv("SONG_TRADER_STATE_DIR", "").strip()
    if explicit:
        return Path(explicit)

    railway_volume = os.getenv("RAILWAY_VOLUME_MOUNT_PATH", "").strip()
    if railway_volume:
        return Path(railway_volume) / "song_trader_v2"

    return Path("/tmp/song_trader_v2")


TOKEN_DIR = _resolve_state_dir()
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
            return str(v).lower() in ("1", "true", "yes", "on")

        return cls(
            paper_app_key=g("KIS_PAPER_APP_KEY", g("KIS_APP_KEY")),
            paper_app_secret=g("KIS_PAPER_APP_SECRET", g("KIS_APP_SECRET")),
            paper_account_no=g("KIS_PAPER_ACCOUNT_NO", g("KIS_ACCOUNT_NO")),
            paper_account_product_code=g("KIS_PAPER_ACCOUNT_PRODUCT_CODE", g("KIS_ACCOUNT_PRODUCT_CODE", "01")),
            live_app_key=g("KIS_LIVE_APP_KEY"),
            live_app_secret=g("KIS_LIVE_APP_SECRET"),
            live_account_no=g("KIS_LIVE_ACCOUNT_NO"),
            live_account_product_code=g("KIS_LIVE_ACCOUNT_PRODUCT_CODE", "01"),
            allow_live=gb("ALLOW_LIVE_TRADING", False),
            live_unlock_phrase=g("LIVE_UNLOCK_PHRASE", "LIVE-TRADING-UNLOCK"),
        )

    @classmethod
    def from_env(cls):
        def g(name, default=""):
            return os.getenv(name, default)

        def gb(name, default=False):
            return g(name, str(default)).lower() in ("1", "true", "yes", "on")

        return cls(
            paper_app_key=g("KIS_PAPER_APP_KEY", g("KIS_APP_KEY")),
            paper_app_secret=g("KIS_PAPER_APP_SECRET", g("KIS_APP_SECRET")),
            paper_account_no=g("KIS_PAPER_ACCOUNT_NO", g("KIS_ACCOUNT_NO")),
            paper_account_product_code=g("KIS_PAPER_ACCOUNT_PRODUCT_CODE", g("KIS_ACCOUNT_PRODUCT_CODE", "01")),
            live_app_key=g("KIS_LIVE_APP_KEY"),
            live_app_secret=g("KIS_LIVE_APP_SECRET"),
            live_account_no=g("KIS_LIVE_ACCOUNT_NO"),
            live_account_product_code=g("KIS_LIVE_ACCOUNT_PRODUCT_CODE", "01"),
            allow_live=gb("ALLOW_LIVE_TRADING", False),
            live_unlock_phrase=g("LIVE_UNLOCK_PHRASE", "LIVE-TRADING-UNLOCK"),
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
            raw_account_no = settings.paper_account_no
            raw_product_code = settings.paper_account_product_code
        else:
            self.base_url = "https://openapi.koreainvestment.com:9443"
            self.app_key = settings.live_app_key
            self.app_secret = settings.live_app_secret
            raw_account_no = settings.live_account_no
            raw_product_code = settings.live_account_product_code

        raw_account_text = str(raw_account_no or "").strip().replace(" ", "")
        raw_product_text = str(raw_product_code or "").strip()
        derived_product = ""

        if "-" in raw_account_text:
            left, right = raw_account_text.split("-", 1)
            raw_account_text = left
            derived_product = "".join(ch for ch in right if ch.isdigit())[:2]

        digits = "".join(ch for ch in raw_account_text if ch.isdigit())
        if len(digits) >= 10:
            derived_product = derived_product or digits[8:10]
            digits = digits[:8]
        elif len(digits) >= 8:
            digits = digits[:8]

        self.account_no = digits
        self.product_code = (
            "".join(ch for ch in raw_product_text if ch.isdigit())[:2]
            or derived_product or "01"
        ).zfill(2)

        if not self.app_key or not self.app_secret:
            raise ValueError("해당 운용모드의 App Key/App Secret이 없습니다.")
        if not self.account_no:
            raise ValueError("해당 운용모드의 계좌번호가 없습니다.")

        default_interval = "1.20" if self.env == "demo" else "0.15"
        self._rest_min_interval = max(0.0, float(os.getenv("KIS_REST_MIN_INTERVAL", default_interval)))
        self._last_rest_request_at = 0.0

    @property
    def token_file(self) -> Path:
        return TOKEN_DIR / f"token_{self.env}.json"

    @property
    def token_error_file(self) -> Path:
        return TOKEN_DIR / f"token_{self.env}_last_error.json"

    def _read_token_cache(self) -> Dict[str, Any]:
        try:
            if not self.token_file.exists():
                return {}
            data = json.loads(self.token_file.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _write_token_cache(self, token: str, expires_at: float, raw_exp: str = "") -> None:
        payload = {
            "token": token,
            "expires_at": float(expires_at),
            "access_token_token_expired": str(raw_exp or ""),
            "saved_at": time.time(),
            "env": self.env,
        }
        tmp = self.token_file.with_suffix(self.token_file.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.token_file)

    def _write_token_error(self, *, status_code: int | None, message: str) -> None:
        # App Key/App Secret 등 비밀값은 절대로 기록하지 않습니다.
        payload = {
            "time": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
            "env": self.env,
            "http_status": status_code,
            "message": str(message)[:1000],
        }
        try:
            tmp = self.token_error_file.with_suffix(self.token_error_file.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.token_error_file)
        except Exception:
            pass

    @staticmethod
    def _token_error_message(r: requests.Response) -> str:
        try:
            data = r.json()
            if isinstance(data, dict):
                parts = [
                    str(data.get("error_description") or ""),
                    str(data.get("msg1") or ""),
                    str(data.get("message") or ""),
                    str(data.get("msg_cd") or ""),
                ]
                msg = " | ".join(x for x in parts if x)
                if msg:
                    return msg[:1000]
        except Exception:
            pass
        return (r.text or f"HTTP {r.status_code}")[:1000]

    def get_token(self) -> str:
        """KIS 접근토큰을 영구 캐시 우선으로 반환합니다.

        핵심 원칙
        - Railway Volume에 토큰을 저장해 재배포 때 불필요한 재발급을 막습니다.
        - 만료 60초 전까지 기존 토큰을 그대로 재사용합니다.
        - 갱신 요청이 실패해도 아직 유효한 기존 토큰이 있으면 즉시 폴백합니다.
        - 일시적 403/429/5xx/네트워크 오류는 짧게 재시도하되 무한 호출하지 않습니다.
        """
        now = time.time()
        cached = self._read_token_cache()
        cached_token = str(cached.get("token") or cached.get("access_token") or "").strip()
        try:
            cached_expires_at = float(cached.get("expires_at", 0) or 0)
        except Exception:
            cached_expires_at = 0.0

        # 구버전 캐시는 만료시각을 timezone 없는 datetime.timestamp()로 저장했을 수 있습니다.
        # 파일 저장시각 기준 23시간을 상한으로 두어 이미 만료된 토큰을 오래 쓰지 않습니다.
        if cached_token:
            try:
                saved_at = float(cached.get("saved_at", 0) or 0)
            except Exception:
                saved_at = 0.0
            if saved_at <= 0:
                try:
                    saved_at = self.token_file.stat().st_mtime
                except Exception:
                    saved_at = 0.0
            if saved_at > 0:
                conservative_expiry = saved_at + 23 * 3600
                cached_expires_at = (
                    min(cached_expires_at, conservative_expiry)
                    if cached_expires_at > 0
                    else conservative_expiry
                )

        # 만료 직전까지 최대한 기존 토큰을 재사용해 tokenP 호출 자체를 줄입니다.
        if cached_token and cached_expires_at > now + 60:
            return cached_token

        url = f"{self.base_url}/oauth2/tokenP"
        body = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
        }

        # 토큰 발급 API를 빠르게 연속 호출하지 않도록 제한적으로만 재시도합니다.
        delays = (0.0, 20.0, 40.0)
        last_exc: Exception | None = None

        for attempt, delay in enumerate(delays, start=1):
            if delay > 0:
                time.sleep(delay)

            try:
                r = requests.post(
                    url,
                    headers={"content-type": "application/json"},
                    data=json.dumps(body),
                    timeout=15,
                )

                if r.ok:
                    data = r.json()
                    token = str(data.get("access_token") or "").strip()
                    if not token:
                        raise RuntimeError("KIS 토큰 응답에 access_token이 없습니다.")

                    # 응답의 만료시각은 한국시간 기준 문자열로 처리합니다.
                    expires_at = time.time() + 23 * 3600
                    raw_exp = str(data.get("access_token_token_expired") or "").strip()
                    if raw_exp:
                        try:
                            dt = datetime.strptime(raw_exp, "%Y-%m-%d %H:%M:%S")
                            dt = dt.replace(tzinfo=ZoneInfo("Asia/Seoul"))
                            expires_at = dt.timestamp()
                        except Exception:
                            pass

                    self._write_token_cache(token, expires_at, raw_exp)
                    try:
                        self.token_error_file.unlink(missing_ok=True)
                    except Exception:
                        pass
                    return token

                msg = self._token_error_message(r)
                self._write_token_error(status_code=r.status_code, message=msg)

                # 갱신은 실패했지만 기존 토큰이 실제 만료 전이면 그 토큰을 계속 씁니다.
                if cached_token and cached_expires_at > time.time() + 5:
                    return cached_token

                last_exc = requests.HTTPError(
                    f"KIS token HTTP {r.status_code}: {msg}",
                    response=r,
                )

                # 400/401 계열은 자격정보/요청 자체 문제일 가능성이 높아 반복 호출하지 않습니다.
                # 403은 일시적인 토큰 발급 제어일 수 있어 위 delays 범위에서만 재시도합니다.
                if r.status_code in (400, 401) or attempt >= len(delays):
                    break

            except requests.RequestException as exc:
                last_exc = exc
                self._write_token_error(status_code=None, message=f"{type(exc).__name__}: {exc}")
                if cached_token and cached_expires_at > time.time() + 5:
                    return cached_token
            except Exception as exc:
                last_exc = exc
                self._write_token_error(status_code=None, message=f"{type(exc).__name__}: {exc}")
                break

        if last_exc is not None:
            raise RuntimeError(
                "KIS 접근토큰을 가져오지 못했습니다. "
                f"마지막 오류는 {self.token_error_file.name}에 비밀값 없이 기록했습니다: {last_exc}"
            ) from last_exc
        raise RuntimeError("KIS 접근토큰을 가져오지 못했습니다.")

    def _headers(self, tr_id: str) -> Dict[str, str]:
        return {
            "content-type": "application/json; charset=utf-8",
            "accept": "text/plain",
            "authorization": f"Bearer {self.get_token()}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id,
            "custtype": "P",
            "tr_cont": "",
        }

    @staticmethod
    def _response_dict(r: requests.Response) -> Dict[str, Any]:
        try:
            data = r.json()
            if not isinstance(data, dict):
                data = {"response": data}
        except Exception:
            data = {"rt_cd": "HTTP_ERROR", "msg_cd": str(r.status_code), "msg1": r.text[:1500]}

        data.setdefault("http_status", r.status_code)
        data.setdefault("rt_cd", "0" if r.ok else "HTTP_ERROR")
        if not r.ok:
            data.setdefault("msg_cd", str(r.status_code))
            data.setdefault("msg1", r.text[:1500])
        return data

    def _rest_wait(self) -> None:
        elapsed = time.monotonic() - self._last_rest_request_at
        wait_for = self._rest_min_interval - elapsed
        if wait_for > 0:
            time.sleep(wait_for)

    @staticmethod
    def _is_rate_limit_response(data: Dict[str, Any]) -> bool:
        if not isinstance(data, dict):
            return False
        msg_cd = str(data.get("msg_cd", "")).upper()
        msg1 = str(data.get("msg1", ""))
        return msg_cd == "EGW00201" or "초당 거래건수" in msg1 or "거래건수를 초과" in msg1

    def get(self, path: str, tr_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        max_attempts = 3 if self.env == "demo" else 2
        last_data: Dict[str, Any] = {}
        for attempt in range(max_attempts):
            self._rest_wait()
            r = requests.get(
                f"{self.base_url}{path}",
                headers=self._headers(tr_id),
                params=params,
                timeout=15,
            )
            self._last_rest_request_at = time.monotonic()
            data = self._response_dict(r) if not r.ok else r.json()
            if not isinstance(data, dict):
                data = {"response": data}
            last_data = data
            if not self._is_rate_limit_response(data):
                return data
            if attempt < max_attempts - 1:
                time.sleep(max(1.5, self._rest_min_interval))
        return last_data

    def post(self, path: str, tr_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        self._rest_wait()
        r = requests.post(
            f"{self.base_url}{path}",
            headers=self._headers(tr_id),
            data=json.dumps(body),
            timeout=15,
        )
        self._last_rest_request_at = time.monotonic()
        return self._response_dict(r)

    def domestic_price(self, code: str) -> Dict[str, Any]:
        return self.get(
            "/uapi/domestic-stock/v1/quotations/inquire-price",
            "FHKST01010100",
            {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code},
        )

    def domestic_volume_rank(self) -> Dict[str, Any]:
        return self.get(
            "/uapi/domestic-stock/v1/quotations/volume-rank",
            "FHPST01710000",
            {
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_COND_SCR_DIV_CODE": "20171",
                "FID_INPUT_ISCD": "0000",
                "FID_DIV_CLS_CODE": "0",
                "FID_BLNG_CLS_CODE": "3",
                "FID_TRGT_CLS_CODE": "111111111",
                "FID_TRGT_EXLS_CLS_CODE": "0000000000",
                "FID_INPUT_PRICE_1": "0",
                "FID_INPUT_PRICE_2": "1000000",
                "FID_VOL_CNT": "100000",
                "FID_INPUT_DATE_1": "",
            },
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

    def domestic_buying_power(self, symbol: str, reference_price: int = 0) -> Dict[str, Any]:
        tr_id = "VTTC8908R" if self.env == "demo" else "TTTC8908R"
        return self.get(
            "/uapi/domestic-stock/v1/trading/inquire-psbl-order",
            tr_id,
            {
                "CANO": self.account_no,
                "ACNT_PRDT_CD": self.product_code,
                "PDNO": str(symbol).zfill(6),
                "ORD_UNPR": str(max(0, int(reference_price or 0))),
                "ORD_DVSN": "01",
                "CMA_EVLU_AMT_ICLD_YN": "Y",
                "OVRS_ICLD_YN": "Y",
            },
        )

    def domestic_order(
        self, code: str, qty: int, side: str, price: int = 0, market_order: bool = True
    ) -> Dict[str, Any]:
        if side not in ("buy", "sell"):
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
            "EXCG_ID_DVSN_CD": "KRX",
            "SLL_TYPE": "01" if side == "sell" else "",
            "CNDT_PRIC": "",
        }
        return self.post("/uapi/domestic-stock/v1/trading/order-cash", tr_id, body)

    def overseas_balance_us(self, exchange: str = "NASD", currency: str = "USD") -> Dict[str, Any]:
        tr_id = "VTTS3012R" if self.env == "demo" else "TTTS3012R"
        return self.get(
            "/uapi/overseas-stock/v1/trading/inquire-balance",
            tr_id,
            {
                "CANO": self.account_no,
                "ACNT_PRDT_CD": self.product_code,
                "OVRS_EXCG_CD": str(exchange).upper(),
                "TR_CRCY_CD": str(currency).upper(),
                "CTX_AREA_FK200": "",
                "CTX_AREA_NK200": "",
            },
        )

    def overseas_present_balance_us(self, foreign_currency: bool = True) -> Dict[str, Any]:
        tr_id = "VTRP6504R" if self.env == "demo" else "CTRP6504R"
        return self.get(
            "/uapi/overseas-stock/v1/trading/inquire-present-balance",
            tr_id,
            {
                "CANO": self.account_no,
                "ACNT_PRDT_CD": self.product_code,
                "WCRC_FRCR_DVSN_CD": "02" if foreign_currency else "01",
                "NATN_CD": "840",
                "TR_MKET_CD": "00",
                "INQR_DVSN_CD": "00",
            },
        )

    def overseas_all_us_balances(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for exchange in ("NASD", "NYSE", "AMEX"):
            data = self.overseas_balance_us(exchange=exchange, currency="USD")
            if isinstance(data, dict):
                data = dict(data)
                data["_exchange"] = exchange
            results.append(data)
        return results

    def overseas_buying_power_us(
        self, symbol: str, limit_price: float, exchange: str = "NASD"
    ) -> Dict[str, Any]:
        exchange = str(exchange).upper().strip()
        symbol = str(symbol).upper().strip()
        if exchange not in ("NASD", "NYSE", "AMEX"):
            raise ValueError("US exchange must be NASD/NYSE/AMEX")
        if not symbol:
            raise ValueError("symbol is required")
        if float(limit_price) <= 0:
            raise ValueError("limit_price must be positive")

        tr_id = "VTTS3007R" if self.env == "demo" else "TTTS3007R"
        return self.get(
            "/uapi/overseas-stock/v1/trading/inquire-psamount",
            tr_id,
            {
                "CANO": self.account_no,
                "ACNT_PRDT_CD": self.product_code,
                "OVRS_EXCG_CD": exchange,
                "OVRS_ORD_UNPR": f"{float(limit_price):.2f}",
                "ITEM_CD": symbol,
            },
        )

    def overseas_order_us(
        self, symbol: str, qty: int, side: str, limit_price: float, exchange: str = "NASD"
    ) -> Dict[str, Any]:
        exchange = str(exchange).upper().strip()
        symbol = str(symbol).upper().strip()
        if exchange not in ("NASD", "NYSE", "AMEX"):
            raise ValueError("US exchange must be NASD/NYSE/AMEX")
        if side not in ("buy", "sell"):
            raise ValueError("side must be buy/sell")
        if not symbol:
            raise ValueError("symbol is required")
        if int(qty) <= 0:
            raise ValueError("qty must be positive")
        if float(limit_price) <= 0:
            raise ValueError("limit_price must be positive")

        if self.env == "demo":
            tr_id = "VTTT1002U" if side == "buy" else "VTTT1001U"
        else:
            tr_id = "TTTT1002U" if side == "buy" else "TTTT1006U"

        body = {
            "CANO": self.account_no,
            "ACNT_PRDT_CD": self.product_code,
            "OVRS_EXCG_CD": exchange,
            "PDNO": symbol,
            "ORD_QTY": str(int(qty)),
            "OVRS_ORD_UNPR": f"{float(limit_price):.2f}",
            "CTAC_TLNO": "",
            "MGCO_APTM_ODNO": "",
            "SLL_TYPE": "00" if side == "sell" else "",
            "ORD_SVR_DVSN_CD": "0",
            "ORD_DVSN": "00",
        }

        res = self.post("/uapi/overseas-stock/v1/trading/order", tr_id, body)
        res.setdefault("_request", {})
        res["_request"].update({
            "env": self.env,
            "tr_id": tr_id,
            "exchange": exchange,
            "symbol": symbol,
            "qty": int(qty),
            "limit_price": f"{float(limit_price):.2f}",
            "ord_dvsn": "00",
        })
        return res


def _looks_like_non_common_stock(name: str) -> bool:
    if not name:
        return False
    n = str(name).upper().replace(" ", "")
    blocked_keywords = [
        "KODEX", "TIGER", "ACE", "SOL", "RISE", "KOSEF", "HANARO",
        "KBSTAR", "ARIRANG", "TIMEFOLIO", "PLUS", "FOCUS", "WOORI",
        "ETN", "인버스", "레버리지", "선물", "S&P", "NASDAQ", "나스닥",
        "채권", "국고채", "회사채", "금리", "달러", "엔선물", "원유",
        "골드", "금선물", "리츠", "REIT", "스팩", "SPAC",
    ]
    if any(k.upper().replace(" ", "") in n for k in blocked_keywords):
        return True
    if n.endswith("우") or "우B" in n or "우C" in n:
        return True
    return False


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

    code_col = pick("mksc_shrn_iscd", "stck_shrn_iscd")
    name_col = pick("hts_kor_isnm", "prdt_name")
    price_col = pick("stck_prpr")
    change_col = pick("prdy_ctrt")
    volume_col = pick("acml_vol")
    amount_col = pick("acml_tr_pbmn", "acml_tr_amt")

    out = pd.DataFrame()
    if code_col:
        out["종목코드"] = df[code_col].astype(str).str.zfill(6)
    if name_col:
        out["종목명"] = df[name_col].astype(str)
    if price_col:
        out["현재가"] = pd.to_numeric(df[price_col], errors="coerce")
    if change_col:
        out["등락률"] = pd.to_numeric(df[change_col], errors="coerce")
    if volume_col:
        out["누적거래량"] = pd.to_numeric(df[volume_col], errors="coerce")
    if amount_col:
        out["거래대금"] = pd.to_numeric(df[amount_col], errors="coerce")

    if "종목명" in out.columns:
        out = out[~out["종목명"].apply(_looks_like_non_common_stock)]
    if "현재가" in out.columns:
        out = out[out["현재가"].fillna(0) >= 1000]
    if "누적거래량" in out.columns:
        out = out[out["누적거래량"].fillna(0) >= 100000]
    if "등락률" in out.columns:
        out = out[out["등락률"].between(-10, 20, inclusive="both")]

    amount_rank = (
        out["거래대금"].rank(pct=True, method="average").fillna(0)
        if "거래대금" in out.columns else pd.Series(0, index=out.index)
    )
    volume_rank = (
        out["누적거래량"].rank(pct=True, method="average").fillna(0)
        if "누적거래량" in out.columns else pd.Series(0, index=out.index)
    )
    change_norm = (
        out["등락률"].clip(lower=0, upper=20) / 20.0
        if "등락률" in out.columns else pd.Series(0, index=out.index)
    )

    out["주도주점수"] = (amount_rank * 50 + volume_rank * 30 + change_norm * 20).round(1)
    sort_cols = [c for c in ["주도주점수", "거래대금", "누적거래량"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    return out.head(top_n).reset_index(drop=True)


def _download_yf(symbol: str, market: str):
    import yfinance as yf
    if market == "국내":
        for suffix in (".KS", ".KQ"):
            df = yf.download(
                symbol + suffix, period="5d", interval="5m",
                auto_adjust=False, progress=False,
            )
            if df is not None and not df.empty:
                return df
        return None
    return yf.download(symbol, period="5d", interval="5m", auto_adjust=False, progress=False)


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
    d["BB_UPPER"] = d["MA20"] + bb_std * d["STD20"]
    d["BB_LOWER"] = d["MA20"] - bb_std * d["STD20"]

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(rsi_period).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    rs = gain / loss.replace(0, np.nan)
    d["RSI"] = 100 - 100 / (1 + rs)

    d["VOL_MA"] = volume.rolling(vol_period).mean()
    d["VOL_RATIO"] = volume / d["VOL_MA"]
    d["MA5"] = close.rolling(5).mean()

    d["BODY"] = (close - open_).abs()
    d["RANGE"] = (high - low).replace(0, np.nan)
    d["BULL"] = close > open_
    d["BEAR"] = close < open_
    return d.dropna()


def score_latest(d):
    if d is None or len(d) < 2:
        return None
    x, prev = d.iloc[-1], d.iloc[-2]
    buy = sell = 0

    if x["Close"] <= x["BB_LOWER"] * 1.01:
        buy += 2
    if prev["Close"] < prev["BB_LOWER"] and x["Close"] > x["BB_LOWER"]:
        buy += 2
    if x["Close"] >= x["BB_UPPER"] * 0.99:
        sell += 2
    if prev["Close"] > prev["BB_UPPER"] and x["Close"] < x["BB_UPPER"]:
        sell += 2

    if 30 <= x["RSI"] <= 40 and x["RSI"] > prev["RSI"]:
        buy += 2
    elif x["RSI"] < 30:
        buy += 1

    if 60 <= x["RSI"] <= 70 and x["RSI"] < prev["RSI"]:
        sell += 2
    elif x["RSI"] > 70:
        sell += 1

    if x["VOL_RATIO"] >= 1.5 and x["BULL"]:
        buy += 2
    if x["VOL_RATIO"] >= 1.5 and x["BEAR"]:
        sell += 2

    if x["BULL"] and x["BODY"] / x["RANGE"] >= 0.6:
        buy += 1
    if x["BEAR"] and x["BODY"] / x["RANGE"] >= 0.6:
        sell += 1

    if x["MA5"] > x["MA20"]:
        buy += 1
    if x["MA5"] < x["MA20"]:
        sell += 1

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


def _num_from_df(df: pd.DataFrame, col_name: str | None, default=0.0) -> pd.Series:
    if not col_name:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(
        df[col_name].astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    ).fillna(default)


def parse_domestic_holdings(balance_json: Dict[str, Any]) -> pd.DataFrame:
    rows = (balance_json or {}).get("output1", []) or []
    if not rows:
        return pd.DataFrame(
            columns=["종목코드", "종목명", "보유수량", "매도가능수량", "평균매입가", "현재가"]
        )

    df = pd.DataFrame(rows)

    def first_existing(*names):
        for n in names:
            if n in df.columns:
                return n
        return None

    code = first_existing("pdno", "mksc_shrn_iscd")
    name = first_existing("prdt_name", "hts_kor_isnm")
    qty = first_existing("hldg_qty", "hold_qty")
    sellable = first_existing("ord_psbl_qty", "sell_psbl_qty", "sll_psbl_qty")
    avg = first_existing("pchs_avg_pric", "avg_pric")
    cur = first_existing("prpr", "stck_prpr")

    out = pd.DataFrame(index=df.index)
    out["종목코드"] = df[code].astype(str).str.zfill(6) if code else ""
    out["종목명"] = df[name].astype(str) if name else ""
    out["보유수량"] = _num_from_df(df, qty, 0).astype(int)
    out["매도가능수량"] = (
        _num_from_df(df, sellable, 0).astype(int) if sellable else out["보유수량"]
    )
    out["평균매입가"] = _num_from_df(df, avg, 0.0)
    out["현재가"] = _num_from_df(df, cur, 0.0)
    return out[out["보유수량"] > 0].reset_index(drop=True)


def parse_overseas_holdings(
    balance_json: Dict[str, Any],
    default_exchange: str = "",
) -> pd.DataFrame:
    empty = pd.DataFrame(columns=[
        "종목코드", "종목명", "거래소", "보유수량",
        "매도가능수량", "평균매입가", "현재가",
        "평가금액", "평가손익", "수익률",
    ])
    if not isinstance(balance_json, dict):
        return empty

    rows = balance_json.get("output1", []) or []
    if not rows and isinstance(balance_json.get("output"), list):
        rows = balance_json.get("output") or []
    if not rows:
        return empty

    df = pd.DataFrame(rows)
    if df.empty:
        return empty

    def first_existing(*names):
        for n in names:
            if n in df.columns:
                return n
        return None

    code = first_existing("ovrs_pdno", "pdno", "symb", "item_cd", "ovrs_item_cd")
    name = first_existing("ovrs_item_name", "prdt_name", "item_name", "ovrs_prdt_name")
    exch = first_existing("ovrs_excg_cd", "ovrs_excg_name", "tr_mket_name", "excg_cd")
    qty = first_existing("ovrs_cblc_qty", "cblc_qty13", "hldg_qty", "hold_qty", "ord_psbl_qty", "ovrs_stck_qty")
    sellable = first_existing("ord_psbl_qty", "sell_psbl_qty", "sll_psbl_qty", "ovrs_ord_psbl_qty")
    avg = first_existing("pchs_avg_pric", "avg_unpr3", "pchs_avg_pric2", "frcr_pchs_avg_pric", "avg_pric")
    cur = first_existing("now_pric2", "ovrs_now_pric1", "ovrs_now_pric", "last", "prpr")
    eval_amt = first_existing("ovrs_stck_evlu_amt", "frcr_evlu_amt2", "evlu_amt", "ovrs_evlu_amt")
    pnl = first_existing("frcr_evlu_pfls_amt", "evlu_pfls_amt2", "evlu_pfls_amt", "ovrs_evlu_pfls_amt")
    pnl_rate = first_existing("evlu_pfls_rt", "evlu_pfls_rt1", "pfls_rt", "profit_loss_rate")

    out = pd.DataFrame(index=df.index)
    out["종목코드"] = df[code].astype(str).str.strip().str.upper() if code else ""
    out["종목명"] = df[name].astype(str).str.strip() if name else ""

    fallback_exchange = str(balance_json.get("_exchange") or default_exchange or "").strip().upper()
    out["거래소"] = df[exch].astype(str).str.strip().str.upper() if exch else fallback_exchange

    out["보유수량"] = _num_from_df(df, qty, 0.0)
    out["매도가능수량"] = _num_from_df(df, sellable, 0.0)
    out["평균매입가"] = _num_from_df(df, avg, 0.0)
    out["현재가"] = _num_from_df(df, cur, 0.0)
    out["평가금액"] = _num_from_df(df, eval_amt, 0.0)
    out["평가손익"] = _num_from_df(df, pnl, 0.0)
    out["수익률"] = _num_from_df(df, pnl_rate, 0.0)

    missing_eval = out["평가금액"].eq(0) & out["현재가"].gt(0) & out["보유수량"].gt(0)
    out.loc[missing_eval, "평가금액"] = out.loc[missing_eval, "현재가"] * out.loc[missing_eval, "보유수량"]

    missing_pnl = (
        out["평가손익"].eq(0) & out["현재가"].gt(0)
        & out["평균매입가"].gt(0) & out["보유수량"].gt(0)
    )
    out.loc[missing_pnl, "평가손익"] = (
        (out.loc[missing_pnl, "현재가"] - out.loc[missing_pnl, "평균매입가"])
        * out.loc[missing_pnl, "보유수량"]
    )

    missing_rate = out["수익률"].eq(0) & out["현재가"].gt(0) & out["평균매입가"].gt(0)
    out.loc[missing_rate, "수익률"] = (
        (out.loc[missing_rate, "현재가"] / out.loc[missing_rate, "평균매입가"] - 1.0) * 100.0
    )

    out = out[out["보유수량"] > 0].copy()
    out = out[out["종목코드"].astype(str).str.len() > 0]
    return out.reset_index(drop=True)


def merge_overseas_holdings(balance_responses: List[Dict[str, Any]]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for item in balance_responses or []:
        if not isinstance(item, dict):
            continue
        exchange = str(item.get("_exchange", "")).upper().strip()
        parsed = parse_overseas_holdings(item, default_exchange=exchange)
        if not parsed.empty:
            frames.append(parsed)

    if not frames:
        return parse_overseas_holdings({})

    out = pd.concat(frames, ignore_index=True)
    dedup_cols = [c for c in ["종목코드", "거래소"] if c in out.columns]
    if dedup_cols:
        out = out.drop_duplicates(subset=dedup_cols, keep="first")
    return out.reset_index(drop=True)


def split_budget(total: int, parts: List[int]) -> List[int]:
    if sum(parts) <= 0:
        return [0 for _ in parts]
    raw = [int(total * p / sum(parts)) for p in parts]
    if raw:
        raw[-1] += total - sum(raw)
    return raw


def is_market_open(market: str) -> bool:
    now = datetime.now(ZoneInfo("Asia/Seoul" if market == "KR" else "America/New_York"))
    if now.weekday() >= 5:
        return False
    t = now.time()
    if market == "KR":
        return dtime(9, 0) <= t < dtime(15, 30)
    return dtime(9, 30) <= t < dtime(16, 0)


def market_force_exit_time(market: str) -> str:
    return "15:15 KST" if market == "KR" else "15:50 ET"


def append_trade_log(row: Dict[str, Any]):
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

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from strategy_kr import _score_candidate

KST = ZoneInfo("Asia/Seoul")


def _resolve_state_dir() -> Path:
    explicit = os.getenv("SONG_TRADER_STATE_DIR", "").strip()
    if explicit:
        return Path(explicit)
    railway_volume = os.getenv("RAILWAY_VOLUME_MOUNT_PATH", "").strip()
    if railway_volume:
        return Path(railway_volume) / "song_trader_v2"
    return Path("/tmp/song_trader_v2")


STATE_DIR = _resolve_state_dir()
REPLAY_DIR = STATE_DIR / "replays"
KR_SNAPSHOT_DIR = REPLAY_DIR / "kr_live_snapshots"
REPLAY_DIR.mkdir(parents=True, exist_ok=True)
KR_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

KR_REPLAY_VERSION = "kr-trade-replay-v3-buy2-abc"
KR_CACHE_DIR = REPLAY_DIR / "kr_cache"
KR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 같은 날짜를 A/B/C로 반복할 때 yfinance를 다시 받지 않도록 메모리 캐시
_INTRADAY_CACHE: dict[tuple, tuple[dict, dict]] = {}


# 과거 KIS 거래량랭킹 자체는 조회할 수 없으므로,
# 최근 국내장 리플레이용으로 거래가 활발한 KOSPI/KOSDAQ 종목군을 사용한다.
# 실제 장중 후보군은 향후 Worker snapshot 기록으로 정확도를 높인다.
DEFAULT_KR_UNIVERSE = [
    ("005930", "삼성전자", "KS"),
    ("000660", "SK하이닉스", "KS"),
    ("373220", "LG에너지솔루션", "KS"),
    ("207940", "삼성바이오로직스", "KS"),
    ("005380", "현대차", "KS"),
    ("000270", "기아", "KS"),
    ("068270", "셀트리온", "KS"),
    ("105560", "KB금융", "KS"),
    ("055550", "신한지주", "KS"),
    ("035420", "NAVER", "KS"),
    ("035720", "카카오", "KS"),
    ("012450", "한화에어로스페이스", "KS"),
    ("009540", "HD한국조선해양", "KS"),
    ("329180", "HD현대중공업", "KS"),
    ("010140", "삼성중공업", "KS"),
    ("042660", "한화오션", "KS"),
    ("028260", "삼성물산", "KS"),
    ("006400", "삼성SDI", "KS"),
    ("051910", "LG화학", "KS"),
    ("066570", "LG전자", "KS"),
    ("003670", "포스코퓨처엠", "KS"),
    ("005490", "POSCO홀딩스", "KS"),
    ("034020", "두산에너빌리티", "KS"),
    ("086790", "하나금융지주", "KS"),
    ("316140", "우리금융지주", "KS"),
    ("032830", "삼성생명", "KS"),
    ("017670", "SK텔레콤", "KS"),
    ("030200", "KT", "KS"),
    ("096770", "SK이노베이션", "KS"),
    ("267260", "HD현대일렉트릭", "KS"),
    ("047050", "포스코인터내셔널", "KS"),
    ("009150", "삼성전기", "KS"),
    ("018260", "삼성에스디에스", "KS"),
    ("011200", "HMM", "KS"),
    ("010130", "고려아연", "KS"),
    ("004020", "현대제철", "KS"),
    ("000810", "삼성화재", "KS"),
    ("090430", "아모레퍼시픽", "KS"),
    ("352820", "하이브", "KS"),
    ("259960", "크래프톤", "KS"),
    ("251270", "넷마블", "KS"),
    ("000100", "유한양행", "KS"),
    ("128940", "한미약품", "KS"),
    ("326030", "SK바이오팜", "KS"),
    ("247540", "에코프로비엠", "KQ"),
    ("086520", "에코프로", "KQ"),
    ("196170", "알테오젠", "KQ"),
    ("028300", "HLB", "KQ"),
    ("214150", "클래시스", "KQ"),
    ("035900", "JYP Ent.", "KQ"),
    ("041510", "에스엠", "KQ"),
    ("122870", "와이지엔터테인먼트", "KQ"),
    ("293490", "카카오게임즈", "KQ"),
    ("263750", "펄어비스", "KQ"),
    ("145020", "휴젤", "KQ"),
    ("058470", "리노공업", "KQ"),
    ("039030", "이오테크닉스", "KQ"),
    ("403870", "HPSP", "KQ"),
    ("095340", "ISC", "KQ"),
    ("240810", "원익IPS", "KQ"),
    ("036930", "주성엔지니어링", "KQ"),
    ("005290", "동진쎄미켐", "KQ"),
    ("067310", "하나마이크론", "KQ"),
    ("357780", "솔브레인", "KQ"),
    ("183300", "코미코", "KQ"),
]


@dataclass
class KRReplayConfig:
    daily_budget_krw: int = 10_000_000
    per_stock_budget_krw: int = 3_000_000
    max_positions: int = 3
    max_daily_orders: int = 12

    buy1_pct: int = 50
    buy2_pct: int = 50

    min_score: float = 50.0
    stop_loss_pct: float = 3.0
    take1_pct: float = 3.0
    take2_pct: float = 5.0

    add2_trigger_pct: float = 0.40

    # BUY2 비교용
    # CURRENT: 현재 국내 로직
    # STRICT: 국내장용 강화 확인
    # NONE: 2차매수 없음
    buy2_mode: str = "CURRENT"
    buy2_min_hold_minutes: float = 5.0
    buy2_strict_trigger_pct: float = 0.80
    buy2_max_rank: int = 3
    buy2_min_score: float = 70.0
    buy2_require_ret3_nonnegative: bool = True
    buy2_require_ret5_positive: bool = True
    buy2_require_ret10_nonnegative: bool = True

    profit_guard_trigger_pct: float = 1.20
    profit_guard_drawdown_pct: float = 0.80

    last_entry_time: str = "14:50"
    force_exit_time: str = "15:15"

    scan_seconds: int = 90
    manage_seconds: int = 45
    scan_count: int = 8  # Railway demo 기준 현재 strategy_kr 기본값

    # 국내는 시장가 주문. 과거 호가가 없으므로 보수적 슬리피지 가정.
    buy_slippage_pct: float = 0.10
    sell_slippage_pct: float = 0.10


def append_kr_top5_snapshot(df: pd.DataFrame, at: datetime | None = None) -> None:
    """실제 Worker가 본 국내 TOP5를 매 스캔 저장. 향후 정확 리플레이용."""
    if df is None or df.empty:
        return
    ts = at or datetime.now(KST)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=KST)
    ts = ts.astimezone(KST)
    path = KR_SNAPSHOT_DIR / f"{ts.strftime('%Y-%m-%d')}.jsonl"
    row = {
        "at": ts.isoformat(timespec="seconds"),
        "top5": df.to_dict("records"),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _clock_seconds(hhmm: str) -> int:
    h, m = [int(x) for x in str(hhmm).split(":")]
    return h * 3600 + m * 60


def _seconds_of_day(ts: pd.Timestamp) -> int:
    return ts.hour * 3600 + ts.minute * 60 + ts.second


def _normalize_universe(codes: Iterable[str] | None = None):
    if not codes:
        return list(DEFAULT_KR_UNIVERSE)
    wanted = {str(x).strip().zfill(6) for x in codes if str(x).strip()}
    known = {code: (code, name, exch) for code, name, exch in DEFAULT_KR_UNIVERSE}
    out = []
    for code in wanted:
        if code in known:
            out.append(known[code])
        else:
            # 거래소를 모르면 KS/KQ 자동탐색을 위해 AUTO로 둔다.
            out.append((code, code, "AUTO"))
    return out


def _extract_yf_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            # group_by='ticker'이면 첫 레벨이 ticker
            if ticker in raw.columns.get_level_values(0):
                d = raw[ticker].copy()
            elif ticker in raw.columns.get_level_values(-1):
                d = raw.xs(ticker, axis=1, level=-1).copy()
            else:
                return pd.DataFrame()
        else:
            d = raw.copy()
    except Exception:
        return pd.DataFrame()

    needed = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in d.columns]
    if "Close" not in needed:
        return pd.DataFrame()
    d = d[needed].copy()
    for c in needed:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["Close"])
    d = d[d["Close"] > 0]
    if d.empty:
        return d

    idx = pd.DatetimeIndex(d.index)
    if idx.tz is None:
        idx = idx.tz_localize(KST)
    else:
        idx = idx.tz_convert(KST)
    d.index = idx
    return d.sort_index()


def _download_intraday(date_text: str, universe) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    """
    FAST V3:
    - 목표 거래일의 1분봉만 다운로드한다.
    - 전일종가는 가벼운 일봉 데이터에서 따로 구한다.
    - 예전처럼 5거래일치 1분봉 전체를 받지 않는다.
    """
    cache_key = (
        str(date_text),
        tuple((str(a), str(b), str(c)) for a, b, c in universe),
    )
    cached = _INTRADAY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        import yfinance as yf
    except Exception as e:
        raise RuntimeError(f"yfinance import 실패: {e}") from e

    target = pd.Timestamp(date_text)
    next_day = (target + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    daily_start = (target - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    target_text = target.strftime("%Y-%m-%d")

    ticker_candidates: dict[str, list[str]] = {}
    ticker_meta: dict[str, dict] = {}
    all_tickers: list[str] = []

    for code, name, exch in universe:
        suffixes = [exch] if exch in ("KS", "KQ") else ["KS", "KQ"]
        ticker_candidates[code] = []
        for suffix in suffixes:
            ticker = f"{code}.{suffix}"
            ticker_candidates[code].append(ticker)
            ticker_meta[ticker] = {
                "code": code,
                "name": name,
                "exchange": suffix,
            }
            all_tickers.append(ticker)

    # 일봉은 매우 작으므로 한 번에 받는다.
    daily_raw = yf.download(
        tickers=" ".join(all_tickers),
        start=daily_start,
        end=next_day,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
        timeout=15,
    )

    prev_close_by_ticker: dict[str, float] = {}
    for ticker in all_tickers:
        d = _extract_yf_frame(daily_raw, ticker)
        if d.empty:
            continue
        before = d[d.index.strftime("%Y-%m-%d") < target_text]
        if before.empty:
            continue
        try:
            prev_close_by_ticker[ticker] = float(
                before["Close"].dropna().iloc[-1]
            )
        except Exception:
            pass

    # 1분봉은 목표일 하루치만 받는다.
    # 너무 큰 단일 요청이 멈추는 것을 줄이기 위해 작은 묶음으로 처리한다.
    batch_size = max(10, int(os.getenv("KR_REPLAY_YF_BATCH_SIZE", "22")))
    raw_parts: list[pd.DataFrame] = []

    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i + batch_size]
        try:
            part = yf.download(
                tickers=" ".join(batch),
                start=target_text,
                end=next_day,
                interval="1m",
                group_by="ticker",
                auto_adjust=False,
                prepost=False,
                threads=True,
                progress=False,
                timeout=12,
            )
            if part is not None and not part.empty:
                raw_parts.append(part)
        except Exception:
            # 일부 묶음 실패가 전체 리플레이를 죽이지 않게 한다.
            continue

    if not raw_parts:
        return {}, {}

    # 각 batch를 개별로 검사한다.
    frames: dict[str, pd.DataFrame] = {}
    meta: dict[str, dict] = {}

    for code, candidates in ticker_candidates.items():
        chosen = pd.DataFrame()
        chosen_ticker = ""

        for ticker in candidates:
            for raw in raw_parts:
                d = _extract_yf_frame(raw, ticker)
                if d.empty:
                    continue
                target_day = d[d.index.strftime("%Y-%m-%d") == date_text].copy()
                if target_day.empty:
                    continue
                chosen = target_day
                chosen_ticker = ticker
                break
            if not chosen.empty:
                break

        if chosen.empty:
            continue

        prev = float(prev_close_by_ticker.get(chosen_ticker, 0.0) or 0.0)
        if prev <= 0:
            # 전일종가가 없으면 일중 등락률을 정확히 계산할 수 없으므로 제외
            continue

        chosen = chosen[["Open", "High", "Low", "Close", "Volume"]].copy()
        chosen["Volume"] = pd.to_numeric(
            chosen["Volume"], errors="coerce"
        ).fillna(0.0)
        chosen["Close"] = pd.to_numeric(
            chosen["Close"], errors="coerce"
        )
        chosen = chosen.dropna(subset=["Close"])
        if chosen.empty:
            continue

        # 반복 계산을 줄이기 위한 누적값
        chosen["_cum_volume"] = chosen["Volume"].cumsum()
        chosen["_cum_amount"] = (
            chosen["Close"] * chosen["Volume"]
        ).cumsum()
        chosen.attrs["prev_close"] = prev

        frames[code] = chosen
        meta[code] = dict(ticker_meta[chosen_ticker])
        meta[code]["ticker"] = chosen_ticker
        meta[code]["prev_close"] = prev

    _INTRADAY_CACHE[cache_key] = (frames, meta)
    return frames, meta


def _previous_close(frame: pd.DataFrame, date_text: str) -> float:
    try:
        cached = float(frame.attrs.get("prev_close", 0.0) or 0.0)
        if cached > 0:
            return cached
    except Exception:
        pass

    before = frame[frame.index.strftime("%Y-%m-%d") < date_text]
    if before.empty:
        return 0.0
    try:
        return float(before["Close"].dropna().iloc[-1])
    except Exception:
        return 0.0


def _bars_until(frame: pd.DataFrame, cutoff: pd.Timestamp, date_text: str) -> pd.DataFrame:
    # FAST V2 frame은 이미 목표일 하루치만 들어 있다.
    # 1분봉 close는 해당 분이 끝난 뒤 알 수 있으므로 60초 전까지만 사용.
    safe_cutoff = cutoff - pd.Timedelta(seconds=60)
    if frame is None or frame.empty:
        return pd.DataFrame()
    return frame.loc[:safe_cutoff].copy()


def _candidate_base_at(frames, meta, date_text: str, cutoff: pd.Timestamp) -> pd.DataFrame:
    rows = []
    safe_cutoff = cutoff - pd.Timedelta(seconds=60)

    for code, frame in frames.items():
        if frame is None or frame.empty:
            continue

        idx = frame.index.searchsorted(safe_cutoff, side="right") - 1
        if idx < 0:
            continue

        prev = _previous_close(frame, date_text)
        if prev <= 0:
            continue

        row = frame.iloc[int(idx)]
        last = float(row.get("Close", 0) or 0)
        if last <= 0:
            continue

        volume = float(row.get("_cum_volume", 0) or 0)
        amount = float(row.get("_cum_amount", 0) or 0)
        day_ret = (last / prev - 1.0) * 100.0
        m = meta.get(code, {})

        rows.append({
            "종목코드": code,
            "종목명": m.get("name", code),
            "현재가": last,
            "등락률": day_ret,
            "누적거래량": volume,
            "거래대금": amount,
        })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out[out["현재가"] >= 1000]
    out = out[out["누적거래량"] >= 100000]
    out = out[out["등락률"].between(-10, 20, inclusive="both")]
    if out.empty:
        return out

    amount_rank = out["거래대금"].rank(
        pct=True, method="average"
    ).fillna(0)
    volume_rank = out["누적거래량"].rank(
        pct=True, method="average"
    ).fillna(0)
    change_norm = out["등락률"].clip(
        lower=0, upper=20
    ) / 20.0

    out["주도주점수"] = (
        amount_rank * 50
        + volume_rank * 30
        + change_norm * 20
    ).round(1)

    return out.sort_values(
        ["주도주점수", "거래대금", "누적거래량"],
        ascending=False,
    ).reset_index(drop=True)


def _build_top5_at(frames, meta, date_text: str, cutoff: pd.Timestamp, scan_count: int = 8) -> pd.DataFrame:
    base = _candidate_base_at(frames, meta, date_text, cutoff)
    if base.empty:
        return pd.DataFrame()
    base = base.head(max(5, int(scan_count)))

    rows = []
    for _, base_row in base.iterrows():
        code = str(base_row.get("종목코드", "")).zfill(6)
        frame = frames.get(code)
        if frame is None or frame.empty:
            continue
        intraday = _bars_until(frame, cutoff, date_text)
        if intraday.empty:
            continue
        intraday = intraday[["Open", "High", "Low", "Close", "Volume"]].copy()
        scored = _score_candidate(base_row, intraday)
        if scored:
            scored["스캔시각"] = cutoff.isoformat(timespec="seconds")
            rows.append(scored)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(
        ["종합점수", "최근3분수익률", "거래량배수", "주도주점수"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    green = df[df["판정"].astype(str).str.contains("매수 후보", na=False)]
    others = df[~df.index.isin(green.index)]
    df = pd.concat([green, others], ignore_index=True)
    df["순위"] = [f"{i+1}위" for i in range(len(df))]
    return df.head(5).reset_index(drop=True)


def _price_at(frame: pd.DataFrame, date_text: str, now: pd.Timestamp) -> float:
    if frame is None or frame.empty:
        return 0.0

    safe_cutoff = now - pd.Timedelta(seconds=60)
    idx = frame.index.searchsorted(safe_cutoff, side="right") - 1
    if idx < 0:
        return 0.0

    try:
        return float(frame.iloc[int(idx)]["Close"])
    except Exception:
        return 0.0


def _fill_price(cfg: KRReplayConfig, side: str, ref_price: float) -> float:
    if side.upper() == "BUY":
        return ref_price * (1.0 + cfg.buy_slippage_pct / 100.0)
    return ref_price * (1.0 - cfg.sell_slippage_pct / 100.0)



def _rank_number(row) -> int:
    raw = str(row.get("순위", "") or "").strip()
    m = re.search(r"(\d+)", raw)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    return 999


def _buy2_allowed(cfg: KRReplayConfig, pos: dict, row, pnl: float, now: pd.Timestamp) -> tuple[bool, str]:
    mode = str(getattr(cfg, "buy2_mode", "CURRENT") or "CURRENT").upper()

    if mode == "NONE":
        return False, "BUY2 비활성"

    if row is None:
        return False, "TOP5 이탈"

    signal_ok = "매수 후보" in str(row.get("판정", ""))
    weak = bool(row.get("모멘텀약화", False))
    score = float(row.get("종합점수", 0) or 0)

    if not signal_ok or weak or score < cfg.min_score:
        return False, "최신 모멘텀 미충족"

    if mode == "CURRENT":
        if pnl < float(cfg.add2_trigger_pct):
            return False, "현재 BUY2 수익률 트리거 미충족"
        return True, "현재 BUY2"

    if mode != "STRICT":
        return False, f"알 수 없는 BUY2 mode={mode}"

    created_at = pos.get("created_at")
    try:
        created = pd.Timestamp(created_at)
        if created.tzinfo is None:
            created = created.tz_localize(KST)
        hold_minutes = max(0.0, (now - created).total_seconds() / 60.0)
    except Exception:
        hold_minutes = 0.0

    rank = _rank_number(row)
    ret3 = float(row.get("최근3분수익률", 0) or 0)
    ret5 = float(row.get("최근5분수익률", 0) or 0)
    ret10 = float(row.get("최근10분수익률", 0) or 0)

    checks = [
        hold_minutes >= float(cfg.buy2_min_hold_minutes),
        pnl >= float(cfg.buy2_strict_trigger_pct),
        rank <= int(cfg.buy2_max_rank),
        score >= float(cfg.buy2_min_score),
    ]

    if bool(cfg.buy2_require_ret3_nonnegative):
        checks.append(ret3 >= 0.0)
    if bool(cfg.buy2_require_ret5_positive):
        checks.append(ret5 > 0.0)
    if bool(cfg.buy2_require_ret10_nonnegative):
        checks.append(ret10 >= 0.0)

    if not all(checks):
        return False, (
            f"STRICT 미충족 · 보유 {hold_minutes:.1f}분 · pnl {pnl:.2f}% · "
            f"TOP{rank} · 점수 {score:.1f} · 3/5/10분 "
            f"{ret3:+.2f}/{ret5:+.2f}/{ret10:+.2f}%"
        )

    return True, (
        f"STRICT 통과 · 보유 {hold_minutes:.1f}분 · pnl {pnl:.2f}% · "
        f"TOP{rank} · 점수 {score:.1f} · 3/5/10분 "
        f"{ret3:+.2f}/{ret5:+.2f}/{ret10:+.2f}%"
    )


def run_kr_trade_replay(
    date_text: str = "2026-08-14",
    codes: Iterable[str] | None = None,
    config: KRReplayConfig | None = None,
    use_cache: bool = True,
) -> dict:
    cfg = config or KRReplayConfig()

    # 기본 CURRENT 전략만 날짜별 결과 캐시를 사용한다.
    cache_path = KR_CACHE_DIR / f"kr_trade_replay_{date_text}.json"
    cache_eligible = (
        use_cache
        and not codes
        and str(getattr(cfg, "buy2_mode", "CURRENT") or "CURRENT").upper() == "CURRENT"
    )
    if cache_eligible and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                isinstance(cached, dict)
                and cached.get("ok") is True
                and cached.get("version") == KR_REPLAY_VERSION
            ):
                cached["cached"] = True
                return cached
        except Exception:
            pass
    universe = _normalize_universe(codes)
    frames, meta = _download_intraday(date_text, universe)
    if not frames:
        raise RuntimeError("해당 날짜의 국내 1분봉 데이터를 받지 못했습니다.")

    # 실제 데이터가 존재하는 종목만 사용
    target_frames = {
        code: frame
        for code, frame in frames.items()
        if not frame[frame.index.strftime("%Y-%m-%d") == date_text].empty
    }
    if not target_frames:
        raise RuntimeError(f"{date_text} 국내 장중 1분봉 데이터가 없습니다.")

    date0 = pd.Timestamp(date_text, tz=KST)
    start = date0 + pd.Timedelta(hours=9, minutes=9)
    end = date0 + pd.Timedelta(hours=15, minutes=16)
    last_entry_sec = _clock_seconds(cfg.last_entry_time)
    force_exit_sec = _clock_seconds(cfg.force_exit_time)

    # 분할 예산
    total_pct = max(1, cfg.buy1_pct + cfg.buy2_pct)
    buy1_amount = int(cfg.per_stock_budget_krw * cfg.buy1_pct / total_pct)
    buy2_amount = cfg.per_stock_budget_krw - buy1_amount

    positions: dict[str, dict] = {}
    events: list[dict] = []
    latest_top5 = pd.DataFrame()
    last_scan = None
    daily_buy_amount = 0.0
    daily_orders = 0

    top5_appearance: dict[str, int] = {}
    green_appearance: dict[str, int] = {}
    best_score: dict[str, float] = {}

    def add_event(ts, symbol, action, side, qty, ref_price, fill_price, reason, pnl="", realized=0.0, score="", rank=""):
        nonlocal daily_orders
        events.append({
            "시간KST": ts.isoformat(),
            "종목코드": symbol,
            "종목명": meta.get(symbol, {}).get("name", symbol),
            "액션": action,
            "구분": side,
            "수량": int(qty),
            "기준가": round(float(ref_price), 2),
            "가정체결가": round(float(fill_price), 2),
            "주문금액KRW": int(round(float(fill_price) * int(qty))),
            "손익률": "" if pnl == "" else round(float(pnl), 3),
            "실현손익KRW": int(round(float(realized))),
            "종합점수": score,
            "TOP5순위": rank,
            "이유": reason,
        })
        daily_orders += 1

    now = start
    while now <= end:
        if last_scan is None or (now - last_scan).total_seconds() >= int(cfg.scan_seconds):
            latest_top5 = _build_top5_at(target_frames, meta, date_text, now, cfg.scan_count)
            last_scan = now
            if latest_top5 is not None and not latest_top5.empty:
                for _, r in latest_top5.iterrows():
                    code = str(r.get("종목코드", "")).zfill(6)
                    top5_appearance[code] = top5_appearance.get(code, 0) + 1
                    score = float(r.get("종합점수", 0) or 0)
                    best_score[code] = max(best_score.get(code, 0.0), score)
                    if "매수 후보" in str(r.get("판정", "")):
                        green_appearance[code] = green_appearance.get(code, 0) + 1

        top5_map = {}
        if latest_top5 is not None and not latest_top5.empty:
            for _, r in latest_top5.iterrows():
                top5_map[str(r.get("종목코드", "")).zfill(6)] = r

        # 1) 보유종목 관리
        for symbol, pos in list(positions.items()):
            frame = target_frames.get(symbol)
            if frame is None:
                continue
            ref_price = _price_at(frame, date_text, now)
            if ref_price <= 0:
                continue
            qty = int(pos.get("qty", 0))
            avg = float(pos.get("avg_price", 0) or 0)
            if qty <= 0 or avg <= 0:
                continue
            pnl = (ref_price / avg - 1.0) * 100.0
            peak = max(float(pos.get("peak_pnl", pnl)), pnl)
            pos["peak_pnl"] = peak
            dd = max(0.0, peak - pnl)

            # 강제청산
            if _seconds_of_day(now) >= force_exit_sec:
                fill = _fill_price(cfg, "SELL", ref_price)
                realized = (fill - avg) * qty
                add_event(now, symbol, "FORCE_SELL", "SELL", qty, ref_price, fill,
                          f"당일 강제청산 {cfg.force_exit_time} KST", pnl, realized)
                positions.pop(symbol, None)
                continue

            if pnl <= -abs(cfg.stop_loss_pct):
                fill = _fill_price(cfg, "SELL", ref_price)
                realized = (fill - avg) * qty
                add_event(now, symbol, "STOP_LOSS", "SELL", qty, ref_price, fill,
                          f"손절 {pnl:.2f}%", pnl, realized)
                positions.pop(symbol, None)
                continue

            if pnl >= cfg.take1_pct and not bool(pos.get("take1_sent")):
                sell_qty = max(1, qty // 2)
                fill = _fill_price(cfg, "SELL", ref_price)
                realized = (fill - avg) * sell_qty
                add_event(now, symbol, "TAKE1", "SELL", sell_qty, ref_price, fill,
                          f"1차 익절 {pnl:.2f}% · 약 50%", pnl, realized)
                pos["qty"] = qty - sell_qty
                pos["take1_sent"] = True
                pos["realized"] = float(pos.get("realized", 0.0)) + realized
                if pos["qty"] <= 0:
                    positions.pop(symbol, None)
                continue

            if pnl >= cfg.take2_pct and bool(pos.get("take1_sent")):
                fill = _fill_price(cfg, "SELL", ref_price)
                realized = (fill - avg) * qty
                add_event(now, symbol, "TAKE2", "SELL", qty, ref_price, fill,
                          f"2차 익절 {pnl:.2f}% · 전량", pnl, realized)
                positions.pop(symbol, None)
                continue

            if peak >= cfg.profit_guard_trigger_pct and dd >= cfg.profit_guard_drawdown_pct:
                if not bool(pos.get("take1_sent")):
                    sell_qty = max(1, qty // 2)
                    fill = _fill_price(cfg, "SELL", ref_price)
                    realized = (fill - avg) * sell_qty
                    add_event(now, symbol, "PROFIT_GUARD1", "SELL", sell_qty, ref_price, fill,
                              f"수익보호 1차 · 최고 +{peak:.2f}% → 현재 {pnl:.2f}% ({dd:.2f}%p 되밀림)", pnl, realized)
                    pos["qty"] = qty - sell_qty
                    pos["take1_sent"] = True
                    pos["realized"] = float(pos.get("realized", 0.0)) + realized
                    if pos["qty"] <= 0:
                        positions.pop(symbol, None)
                    continue
                else:
                    fill = _fill_price(cfg, "SELL", ref_price)
                    realized = (fill - avg) * qty
                    add_event(now, symbol, "PROFIT_GUARD2", "SELL", qty, ref_price, fill,
                              f"수익보호 2차 · 최고 +{peak:.2f}% → 현재 {pnl:.2f}% ({dd:.2f}%p 되밀림)", pnl, realized)
                    positions.pop(symbol, None)
                    continue

            # 2차 분할매수: CURRENT / STRICT / NONE 비교 가능
            if (
                _seconds_of_day(now) < last_entry_sec
                and int(pos.get("stage", 1)) == 1
                and daily_orders < cfg.max_daily_orders
            ):
                row = top5_map.get(symbol)
                buy2_ok, buy2_reason = _buy2_allowed(
                    cfg, pos, row, pnl, now
                )
                if buy2_ok:
                    fill = _fill_price(cfg, "BUY", ref_price)
                    qty2 = int(buy2_amount // fill)
                    cost = fill * qty2
                    if qty2 > 0 and daily_buy_amount + cost <= cfg.daily_budget_krw:
                        old_qty = int(pos["qty"])
                        old_avg = float(pos["avg_price"])
                        new_qty = old_qty + qty2
                        new_avg = (old_avg * old_qty + fill * qty2) / new_qty
                        rank = str(row.get("순위", ""))
                        score = float(row.get("종합점수", 0) or 0)
                        add_event(
                            now, symbol, "BUY2", "BUY", qty2,
                            ref_price, fill,
                            f"2차 분할매수 · {buy2_reason}",
                            pnl, 0.0, score, rank,
                        )
                        pos["qty"] = new_qty
                        pos["avg_price"] = new_avg
                        pos["stage"] = 2
                        daily_buy_amount += cost

        # 2) 신규진입
        if (
            _seconds_of_day(now) < last_entry_sec
            and len(positions) < cfg.max_positions
            and daily_orders < cfg.max_daily_orders
            and latest_top5 is not None
            and not latest_top5.empty
        ):
            for _, row in latest_top5.iterrows():
                if len(positions) >= cfg.max_positions or daily_orders >= cfg.max_daily_orders:
                    break
                symbol = str(row.get("종목코드", "")).zfill(6)
                if symbol in positions:
                    continue
                signal = str(row.get("판정", ""))
                score = float(row.get("종합점수", 0) or 0)
                weak = bool(row.get("모멘텀약화", False))
                if "매수 후보" not in signal or weak or score < cfg.min_score:
                    continue
                frame = target_frames.get(symbol)
                if frame is None:
                    continue
                ref_price = _price_at(frame, date_text, now)
                if ref_price <= 0:
                    continue
                fill = _fill_price(cfg, "BUY", ref_price)
                qty1 = int(buy1_amount // fill)
                if qty1 <= 0:
                    continue
                cost = fill * qty1
                if daily_buy_amount + cost > cfg.daily_budget_krw:
                    continue
                rank = str(row.get("순위", ""))
                r3 = float(row.get("최근3분수익률", 0) or 0)
                r5 = float(row.get("최근5분수익률", 0) or 0)
                vr = float(row.get("거래량배수", 0) or 0)
                add_event(now, symbol, "BUY1", "BUY", qty1, ref_price, fill,
                          f"빠른모멘텀 1차매수 · 점수 {score:.1f} · 3분 {r3:+.2f}% · 5분 {r5:+.2f}% · 거래량 {vr:.2f}배",
                          "", 0.0, score, rank)
                positions[symbol] = {
                    "qty": qty1,
                    "avg_price": fill,
                    "stage": 1,
                    "created_at": now.isoformat(),
                    "take1_sent": False,
                    "peak_pnl": 0.0,
                    "opened_at": now.isoformat(),
                }
                daily_buy_amount += cost

        now += pd.Timedelta(seconds=int(cfg.manage_seconds))

    # 안전: 리플레이 종료 시 남은 포지션이 있으면 마지막 가격으로 청산 표시
    if positions:
        for symbol, pos in list(positions.items()):
            frame = target_frames.get(symbol)
            ref_price = _price_at(frame, date_text, end) if frame is not None else 0.0
            if ref_price <= 0:
                continue
            qty = int(pos.get("qty", 0))
            avg = float(pos.get("avg_price", 0) or 0)
            fill = _fill_price(cfg, "SELL", ref_price)
            pnl = (ref_price / avg - 1.0) * 100.0 if avg > 0 else 0.0
            realized = (fill - avg) * qty
            add_event(end, symbol, "FORCE_SELL_END", "SELL", qty, ref_price, fill,
                      "리플레이 종료 안전청산", pnl, realized)
            positions.pop(symbol, None)

    events_df = pd.DataFrame(events)
    buy_amount = float(events_df.loc[events_df["구분"] == "BUY", "주문금액KRW"].sum()) if not events_df.empty else 0.0
    sell_amount = float(events_df.loc[events_df["구분"] == "SELL", "주문금액KRW"].sum()) if not events_df.empty else 0.0
    realized = float(events_df["실현손익KRW"].sum()) if not events_df.empty else 0.0

    # 종목별 집계
    symbol_rows = []
    if not events_df.empty:
        for symbol, g in events_df.groupby("종목코드", sort=False):
            buys = g[g["구분"] == "BUY"]
            sells = g[g["구분"] == "SELL"]
            pnl = float(g["실현손익KRW"].sum())
            symbol_rows.append({
                "종목코드": symbol,
                "종목명": str(g.iloc[0].get("종목명", symbol)),
                "매수횟수": int(len(buys)),
                "매도횟수": int(len(sells)),
                "총매수금액KRW": int(buys["주문금액KRW"].sum()) if not buys.empty else 0,
                "총매도금액KRW": int(sells["주문금액KRW"].sum()) if not sells.empty else 0,
                "실현손익KRW": int(round(pnl)),
                "매수금액대비수익률": round((pnl / float(buys["주문금액KRW"].sum()) * 100.0), 3) if not buys.empty and float(buys["주문금액KRW"].sum()) > 0 else 0.0,
                "종료사유": str(sells.iloc[-1].get("액션", "")) if not sells.empty else "",
            })

    leader_rows = []
    for code in sorted(top5_appearance, key=lambda x: (green_appearance.get(x, 0), best_score.get(x, 0)), reverse=True):
        leader_rows.append({
            "종목코드": code,
            "종목명": meta.get(code, {}).get("name", code),
            "TOP5등장횟수": int(top5_appearance.get(code, 0)),
            "매수후보횟수": int(green_appearance.get(code, 0)),
            "최고점수": round(float(best_score.get(code, 0.0)), 1),
        })

    summary = {
        "총주문횟수": int(len(events_df)),
        "매수주문횟수": int((events_df["구분"] == "BUY").sum()) if not events_df.empty else 0,
        "매도주문횟수": int((events_df["구분"] == "SELL").sum()) if not events_df.empty else 0,
        "거래종목수": int(len(symbol_rows)),
        "수익종목수": int(sum(1 for x in symbol_rows if x["실현손익KRW"] > 0)),
        "손실종목수": int(sum(1 for x in symbol_rows if x["실현손익KRW"] < 0)),
        "누적매수금액KRW": int(round(buy_amount)),
        "누적매도금액KRW": int(round(sell_amount)),
        "실현손익KRW": int(round(realized)),
        "누적매수금액대비수익률": round(realized / buy_amount * 100.0, 3) if buy_amount > 0 else 0.0,
        "일일예산1000만원대비수익률": round(realized / cfg.daily_budget_krw * 100.0, 3),
    }

    stamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    detail_path = REPLAY_DIR / f"kr_trade_replay_{date_text}_{stamp}.csv"
    summary_path = REPLAY_DIR / f"kr_trade_replay_{date_text}_{stamp}.json"
    if not events_df.empty:
        events_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    payload = {
        "ok": True,
        "version": KR_REPLAY_VERSION,
        "date": date_text,
        "strategy": {
            "buy2_mode": str(getattr(cfg, "buy2_mode", "CURRENT")).upper(),
            "buy2_current_trigger_pct": float(cfg.add2_trigger_pct),
            "buy2_strict_trigger_pct": float(cfg.buy2_strict_trigger_pct),
            "buy2_min_hold_minutes": float(cfg.buy2_min_hold_minutes),
            "buy2_max_rank": int(cfg.buy2_max_rank),
            "buy2_min_score": float(cfg.buy2_min_score),
        },
        "universe_count": len(universe),
        "data_available_count": len(target_frames),
        "universe_mode": "fixed-liquid-universe-approximation",
        "summary": summary,
        "symbols": symbol_rows,
        "leader_summary": leader_rows[:20],
        "events": events,
        "assumptions": {
            "real_orders": False,
            "data": "yfinance 1-minute historical bars",
            "candidate_reconstruction": "현재 strategy_kr 점수식을 고정 유동성 종목군 안에서 재구성",
            "scan_cadence_seconds": cfg.scan_seconds,
            "management_cadence_seconds": cfg.manage_seconds,
            "buy_slippage_pct": cfg.buy_slippage_pct,
            "sell_slippage_pct": cfg.sell_slippage_pct,
            "fees_taxes": "별도 미포함",
            "important_limit": "과거 KIS 실시간 거래량랭킹 원본은 없으므로 당시 전체시장 TOP5를 100% 복원하는 리플레이는 아님",
            "future_exactness": "이 버전부터 실제 Worker TOP5 snapshot을 /data에 저장해 향후 정확도를 높임",
        },
    }
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    if not codes and str(getattr(cfg, "buy2_mode", "CURRENT")).upper() == "CURRENT":
        try:
            cache_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:
            pass

    payload["cached"] = False
    return payload


def compare_kr_buy2_strategies(
    date_text: str = "2026-08-10",
    codes: Iterable[str] | None = None,
) -> dict:
    """
    국내 BUY2 A/B/C
    A_CURRENT: 현재 +0.40% + 최신 매수후보 유지
    B_STRICT: 최소 5분 +0.80%, TOP3, 70점, 3/5/10분 모멘텀 유지
    C_NO_BUY2: 2차매수 없음
    """
    variants = [
        (
            "A_CURRENT",
            KRReplayConfig(
                buy2_mode="CURRENT",
            ),
            "현재 국내 BUY2",
        ),
        (
            "B_STRICT",
            KRReplayConfig(
                buy2_mode="STRICT",
                buy2_min_hold_minutes=5.0,
                buy2_strict_trigger_pct=0.80,
                buy2_max_rank=3,
                buy2_min_score=70.0,
                buy2_require_ret3_nonnegative=True,
                buy2_require_ret5_positive=True,
                buy2_require_ret10_nonnegative=True,
            ),
            "최소 5분 +0.8% · TOP3 · 70점 · 3/5/10분 모멘텀 유지",
        ),
        (
            "C_NO_BUY2",
            KRReplayConfig(
                buy2_mode="NONE",
            ),
            "2차매수 없음",
        ),
    ]

    results = []
    for key, cfg, description in variants:
        out = run_kr_trade_replay(
            date_text=date_text,
            codes=codes,
            config=cfg,
            use_cache=False,
        )
        summary = dict(out.get("summary", {}) or {})
        events = list(out.get("events", []) or [])
        buy2_events = [
            e for e in events
            if str(e.get("액션", "")).upper() == "BUY2"
        ]
        stop_events = [
            e for e in events
            if str(e.get("액션", "")).upper() == "STOP_LOSS"
        ]

        results.append({
            "strategy": key,
            "description": description,
            "실현손익KRW": int(summary.get("실현손익KRW", 0) or 0),
            "일일예산1000만원대비수익률": float(
                summary.get("일일예산1000만원대비수익률", 0) or 0
            ),
            "총주문횟수": int(summary.get("총주문횟수", 0) or 0),
            "매수주문횟수": int(summary.get("매수주문횟수", 0) or 0),
            "매도주문횟수": int(summary.get("매도주문횟수", 0) or 0),
            "BUY2횟수": len(buy2_events),
            "STOP_LOSS횟수": len(stop_events),
            "거래종목수": int(summary.get("거래종목수", 0) or 0),
            "수익종목수": int(summary.get("수익종목수", 0) or 0),
            "손실종목수": int(summary.get("손실종목수", 0) or 0),
            "symbols": out.get("symbols", []),
            "events": events,
        })

    ranked = sorted(
        results,
        key=lambda x: (
            int(x.get("실현손익KRW", 0)),
            -int(x.get("STOP_LOSS횟수", 0)),
        ),
        reverse=True,
    )

    return {
        "ok": True,
        "version": "kr-buy2-abc-v1",
        "date": date_text,
        "comparison": results,
        "recommended_by_replay": ranked[0]["strategy"] if ranked else "",
        "warning": (
            "과거 KIS 전체시장 실시간 거래량 랭킹은 완전 복원되지 않으며, "
            "수수료·세금은 별도 미포함입니다. 이 결과만으로 실전 수익을 보장하지 않습니다."
        ),
    }


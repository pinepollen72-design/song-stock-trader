from __future__ import annotations

"""
KR SWING V2 — Dual Entry: Pullback Recovery + 20D Breakout

Research-only multi-day backtest.

Core principles
- Separate from intraday D-v2/CLEAN logic.
- Daily bars only.
- Signal is decided at day T close; entry is day T+1 open (no look-ahead fill).
- Maximum 3 positions.
- Position size is capped by both 3M KRW and ~1% equity initial risk.
- Initial stop uses ATR + recent swing structure.
- No fixed +3%/+5% full exit. Winners are allowed to run.
- Trend / time / trailing-stop exits.
- KIS read-only daily price API; never sends an order.
- Same fixed liquidity universe limitation as existing historical KR replay.
"""

import gzip
import json
import math
import os
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from zoneinfo import ZoneInfo

from trader_core import Settings, KISClient
import replay_kr

KST = ZoneInfo("Asia/Seoul")
SWING_VERSION = "kr-swing-v2-dual-entry-fast-v1-diag1"

_THREAD: threading.Thread | None = None
_LOCK = threading.RLock()
_STOP = threading.Event()


def _resolve_state_dir() -> Path:
    explicit = os.getenv("SONG_TRADER_STATE_DIR", "").strip()
    if explicit:
        return Path(explicit)
    railway_volume = os.getenv("RAILWAY_VOLUME_MOUNT_PATH", "").strip()
    if railway_volume:
        return Path(railway_volume) / "song_trader_v2"
    return Path("/tmp/song_trader_v2")


ROOT = _resolve_state_dir() / "replays" / "kr_swing_v2"
# Reuse the already-downloaded V1 daily-bar cache when available.
RAW_DAILY_DIR = _resolve_state_dir() / "replays" / "kr_swing_v1" / "kis_daily_ohlcv"
STATE_FILE = ROOT / "state.json"
RESULT_FILE = ROOT / "result.json"

for _p in (ROOT, RAW_DAILY_DIR):
    _p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class SwingConfig:
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"
    initial_capital_krw: int = 10_000_000
    max_positions: int = 3
    max_new_entries_per_day: int = 2
    per_stock_cap_krw: int = 3_000_000

    # Risk
    risk_per_trade_pct: float = 1.0
    min_stop_pct: float = 3.5
    max_stop_pct: float = 8.0
    atr_stop_multiple: float = 2.0
    trailing_atr_multiple: float = 2.5
    breakeven_at_r: float = 1.0
    trailing_start_r: float = 2.0

    # Entry trend / pullback / re-acceleration
    min_sma_days: int = 60
    min_score: float = 42.0
    min_rs_percentile: float = 0.50
    neutral_min_rs_percentile: float = 0.60
    min_volume_ratio: float = 0.65
    neutral_min_volume_ratio: float = 0.75
    min_pullback_pct: float = 1.0
    max_pullback_pct: float = 12.0
    max_extension_from_sma20_pct: float = 12.0
    max_breakout_extension_pct: float = 4.0
    max_entry_gap_up_pct: float = 4.0
    max_entry_gap_down_pct: float = -2.5
    cooldown_trading_days: int = 3

    # Market regime based on breadth of the fixed liquidity universe
    risk_on_breadth: float = 0.50
    neutral_breadth: float = 0.30
    risk_on_min_median_ret20: float = -0.03
    neutral_max_positions: int = 2
    risk_off_max_positions: int = 1
    risk_off_min_rs_percentile: float = 0.75
    risk_off_min_breakout_volume_ratio: float = 1.15
    breakout_min_volume_ratio: float = 0.95
    neutral_breakout_min_volume_ratio: float = 1.00
    pullback_recovery_min_close_gain_pct: float = 0.50

    # Exit
    below_sma20_exit_days: int = 2
    time_stop_days: int = 5
    time_stop_min_return_pct: float = 2.0
    max_hold_days: int = 20

    # Execution assumptions
    buy_slippage_pct: float = 0.10
    sell_slippage_pct: float = 0.10
    fees_taxes: str = "excluded"
    kis_min_interval_seconds: float = 0.22
    max_api_attempts: int = 5


def _atomic_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    tmp.replace(path)


def _read_json(path: Path, default=None):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


def _save_gzip(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, default=str)
    tmp.replace(path)


def _load_gzip(path: Path, default=None):
    try:
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default


def _state(**updates) -> dict:
    with _LOCK:
        cur = dict(_read_json(STATE_FILE, {}) or {})
        cur.update(updates)
        cur["version"] = SWING_VERSION
        cur["updated_at"] = datetime.now(KST).isoformat(timespec="seconds")
        _atomic_json(STATE_FILE, cur)
        return cur


def _public_state() -> dict:
    out = dict(_read_json(STATE_FILE, {}) or {})
    if not out:
        out = {
            "ok": True,
            "version": SWING_VERSION,
            "status": "not_started",
            "result_ready": False,
        }
    out["ok"] = True
    out["version"] = SWING_VERSION
    out["thread_alive"] = bool(_THREAD and _THREAD.is_alive())
    return out


def _universe() -> list[tuple[str, str, str]]:
    return replay_kr._normalize_universe(None)


def _wait_if_protected(protected_window_fn) -> None:
    while not _STOP.is_set():
        try:
            live, label = protected_window_fn()
        except Exception:
            live, label = False, ""
        if not live:
            return
        _state(
            status="paused_live_window",
            phase="PAUSED",
            pause_reason=str(label or ""),
            message="실시간 자동매매 보호를 위해 SWING 데이터 수집을 잠시 멈춥니다.",
        )
        _STOP.wait(30.0)


def _api_get(
    client: KISClient,
    path: str,
    tr_id: str,
    params: dict,
    cfg: SwingConfig,
    protected_window_fn,
) -> dict:
    last: dict = {}
    for attempt in range(int(cfg.max_api_attempts)):
        _wait_if_protected(protected_window_fn)
        if _STOP.is_set():
            raise RuntimeError("swing backtest stopped")
        try:
            data = client.get(path, tr_id, params)
            if not isinstance(data, dict):
                data = {"response": data}
            last = data
            rt_cd = str(data.get("rt_cd", "0"))
            msg_cd = str(data.get("msg_cd", ""))
            msg1 = str(data.get("msg1", ""))
            if rt_cd == "0" or data.get("output2") is not None:
                return data
            rate_limited = (
                msg_cd.upper() == "EGW00201"
                or "초당 거래건수" in msg1
                or "거래건수를 초과" in msg1
            )
            if not rate_limited and attempt >= 1:
                return data
        except Exception as exc:
            last = {
                "rt_cd": "EXCEPTION",
                "msg1": f"{type(exc).__name__}: {exc}",
            }
        if attempt < int(cfg.max_api_attempts) - 1:
            time.sleep(min(8.0, 0.8 * (2 ** attempt)))
    return last


def _cache_path(code: str) -> Path:
    return RAW_DAILY_DIR / f"{str(code).zfill(6)}.json.gz"


def _parse_num(value, default=0.0) -> float:
    try:
        return float(str(value).replace(",", "").strip() or default)
    except Exception:
        return float(default)


def _fetch_daily_ohlcv(
    client: KISClient,
    code: str,
    cfg: SwingConfig,
    protected_window_fn,
) -> pd.DataFrame:
    code = str(code).zfill(6)
    path = _cache_path(code)
    cached = _load_gzip(path, {}) or {}

    lookback_start = (
        pd.Timestamp(cfg.start_date) - pd.Timedelta(days=180)
    ).strftime("%Y-%m-%d")

    if isinstance(cached, dict) and cached:
        keys = sorted(k for k in cached if len(str(k)) == 10)
        if (
            keys
            and keys[0] <= (
                pd.Timestamp(cfg.start_date) - pd.Timedelta(days=100)
            ).strftime("%Y-%m-%d")
            and keys[-1] >= (
                pd.Timestamp(cfg.end_date) - pd.Timedelta(days=5)
            ).strftime("%Y-%m-%d")
        ):
            return _records_dict_to_df(cached)

    floor = pd.Timestamp(lookback_start).strftime("%Y%m%d")
    cursor_end = pd.Timestamp(cfg.end_date).strftime("%Y%m%d")
    out: dict[str, dict] = {}

    for _ in range(10):
        data = _api_get(
            client,
            "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
            "FHKST03010100",
            {
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": code,
                "FID_INPUT_DATE_1": floor,
                "FID_INPUT_DATE_2": cursor_end,
                "FID_PERIOD_DIV_CODE": "D",
                "FID_ORG_ADJ_PRC": "0",
            },
            cfg,
            protected_window_fn,
        )
        rows = data.get("output2") or []
        if isinstance(rows, dict):
            rows = [rows]

        parsed: list[pd.Timestamp] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            raw_date = str(row.get("stck_bsop_date", "")).strip()
            if len(raw_date) != 8:
                continue
            try:
                dt = pd.Timestamp(raw_date)
            except Exception:
                continue

            o = _parse_num(row.get("stck_oprc"))
            h = _parse_num(row.get("stck_hgpr"))
            l = _parse_num(row.get("stck_lwpr"))
            c = _parse_num(row.get("stck_clpr", row.get("stck_prpr")))
            v = _parse_num(row.get("acml_vol"))
            if min(o, h, l, c) <= 0:
                continue

            key = dt.strftime("%Y-%m-%d")
            out[key] = {
                "Open": o,
                "High": h,
                "Low": l,
                "Close": c,
                "Volume": max(0.0, v),
            }
            parsed.append(dt)

        if not parsed:
            break

        oldest = min(parsed)
        if oldest.strftime("%Y%m%d") <= floor:
            break

        cursor_end = (oldest - pd.Timedelta(days=1)).strftime("%Y%m%d")
        if cursor_end < floor:
            break

    if out:
        _save_gzip(path, out)
    return _records_dict_to_df(out)


def _records_dict_to_df(records: dict) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume"]
        )
    df = pd.DataFrame.from_dict(records, orient="index")
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].sort_index()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    return df.dropna(subset=["Open", "High", "Low", "Close"])


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["SMA10"] = x["Close"].rolling(10).mean()
    x["SMA20"] = x["Close"].rolling(20).mean()
    x["SMA50"] = x["Close"].rolling(50).mean()
    x["SMA60"] = x["Close"].rolling(60).mean()
    x["SMA20_5AGO"] = x["SMA20"].shift(5)

    prev_close = x["Close"].shift(1)
    tr = pd.concat(
        [
            x["High"] - x["Low"],
            (x["High"] - prev_close).abs(),
            (x["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    x["ATR14"] = tr.rolling(14).mean()

    x["AVG_VOL20"] = x["Volume"].rolling(20).mean()
    x["VOL_RATIO"] = x["Volume"] / x["AVG_VOL20"].replace(0, pd.NA)
    x["RET20"] = x["Close"] / x["Close"].shift(20) - 1.0
    x["RET5"] = x["Close"] / x["Close"].shift(5) - 1.0

    x["HIGH20_PREV"] = x["High"].shift(1).rolling(20).max()
    x["HIGH3_PREV"] = x["High"].shift(1).rolling(3).max()
    x["LOW5_PREV"] = x["Low"].shift(1).rolling(5).min()
    x["PREV_HIGH"] = x["High"].shift(1)
    x["PREV_CLOSE"] = x["Close"].shift(1)

    x["PULLBACK_DEPTH"] = (
        (x["HIGH20_PREV"] - x["LOW5_PREV"])
        / x["HIGH20_PREV"].replace(0, pd.NA)
    )
    x["EXT_SMA20"] = (
        x["Close"] / x["SMA20"].replace(0, pd.NA) - 1.0
    )
    return x


def _market_snapshot(
    day: pd.Timestamp,
    frames: dict[str, pd.DataFrame],
    cfg: SwingConfig,
) -> dict:
    above20 = []
    ret20 = []
    for df in frames.values():
        if day not in df.index:
            continue
        r = df.loc[day]
        if pd.isna(r.get("SMA20")) or pd.isna(r.get("RET20")):
            continue
        above20.append(float(r["Close"]) > float(r["SMA20"]))
        ret20.append(float(r["RET20"]))

    breadth = (
        sum(1 for x in above20 if x) / len(above20)
        if above20
        else 0.0
    )
    median_ret20 = (
        float(pd.Series(ret20).median())
        if ret20
        else -1.0
    )

    if (
        breadth >= cfg.risk_on_breadth
        and median_ret20 >= cfg.risk_on_min_median_ret20
    ):
        regime = "RISK_ON"
    elif breadth >= cfg.neutral_breadth:
        regime = "NEUTRAL"
    else:
        regime = "RISK_OFF"

    return {
        "regime": regime,
        "breadth_above_sma20": round(breadth, 4),
        "median_ret20": round(median_ret20, 5),
        "sample_size": len(above20),
    }


def _rs_percentiles(
    day: pd.Timestamp,
    frames: dict[str, pd.DataFrame],
) -> dict[str, float]:
    vals = []
    for code, df in frames.items():
        if day not in df.index:
            continue
        v = df.loc[day].get("RET20")
        if pd.isna(v):
            continue
        vals.append((code, float(v)))

    if not vals:
        return {}

    vals.sort(key=lambda x: x[1])
    n = max(1, len(vals) - 1)
    return {
        code: idx / n
        for idx, (code, _v) in enumerate(vals)
    }


def _candidate_rows(
    day: pd.Timestamp,
    frames: dict[str, pd.DataFrame],
    names: dict[str, str],
    market: dict,
    cfg: SwingConfig,
) -> list[dict]:
    """V2 has two independent entry doors.

    A) PULLBACK_RECOVERY
       Existing uptrend + recent 1~12% pullback + recovery above prior-day
       high (or a meaningful close recovery above SMA10).

    B) BREAKOUT_20D
       Existing uptrend + fresh previous-20-day-high breakout.
       Pullback is NOT required.

    RISK_OFF is not a blanket ban in V2. Only strong 20D breakouts may enter,
    with one-position exposure and stricter RS/volume requirements.
    """
    rs = _rs_percentiles(day, frames)
    regime = market["regime"]

    rows: list[dict] = []
    for code, df in frames.items():
        if day not in df.index:
            continue
        r = df.loc[day]

        required = [
            "SMA10", "SMA20", "SMA60", "SMA20_5AGO", "ATR14",
            "VOL_RATIO", "RET20", "HIGH20_PREV", "HIGH3_PREV",
            "LOW5_PREV", "PULLBACK_DEPTH", "EXT_SMA20",
            "PREV_HIGH", "PREV_CLOSE",
        ]
        if any(pd.isna(r.get(k)) for k in required):
            continue

        close = float(r["Close"])
        open_ = float(r["Open"])
        high = float(r["High"])
        low = float(r["Low"])
        sma10 = float(r["SMA10"])
        sma20 = float(r["SMA20"])
        sma60 = float(r["SMA60"])
        sma20_old = float(r["SMA20_5AGO"])
        vol_ratio = float(r["VOL_RATIO"])
        ret20 = float(r["RET20"])
        high20 = float(r["HIGH20_PREV"])
        high3 = float(r["HIGH3_PREV"])
        low5 = float(r["LOW5_PREV"])
        prev_high = float(r["PREV_HIGH"])
        prev_close = float(r["PREV_CLOSE"])
        pullback = float(r["PULLBACK_DEPTH"])
        extension = float(r["EXT_SMA20"])
        rs_pct = float(rs.get(code, 0.0))

        # V2 keeps the long-term trend discipline.
        trend_ok = bool(
            close > sma20 > sma60
            and sma20 > sma20_old
            and ret20 > -0.02
        )
        if not trend_ok:
            continue

        if regime == "RISK_ON":
            pullback_min_rs = cfg.min_rs_percentile
            pullback_min_vol = cfg.min_volume_ratio
            breakout_min_rs = max(cfg.min_rs_percentile, 0.55)
            breakout_min_vol = cfg.breakout_min_volume_ratio
        elif regime == "NEUTRAL":
            pullback_min_rs = cfg.neutral_min_rs_percentile
            pullback_min_vol = cfg.neutral_min_volume_ratio
            breakout_min_rs = max(cfg.neutral_min_rs_percentile, 0.65)
            breakout_min_vol = cfg.neutral_breakout_min_volume_ratio
        else:
            # In weak markets, do not buy ordinary pullbacks.
            pullback_min_rs = 2.0
            pullback_min_vol = 99.0
            breakout_min_rs = cfg.risk_off_min_rs_percentile
            breakout_min_vol = cfg.risk_off_min_breakout_volume_ratio

        # -------- A) Pullback recovery --------
        pullback_ok = bool(
            cfg.min_pullback_pct / 100.0
            <= pullback
            <= cfg.max_pullback_pct / 100.0
        )
        close_gain_pct = (
            (close / prev_close - 1.0) * 100.0
            if prev_close > 0
            else 0.0
        )
        recovery_ok = bool(
            close > open_
            and close > sma10
            and (
                close > prev_high
                or (
                    close > high3 * 0.995
                    and close_gain_pct
                    >= cfg.pullback_recovery_min_close_gain_pct
                )
            )
        )
        pullback_location_ok = bool(
            extension <= 0.07
            and close <= high20 * 1.03
        )
        pullback_signal = bool(
            regime != "RISK_OFF"
            and pullback_ok
            and recovery_ok
            and pullback_location_ok
            and rs_pct >= pullback_min_rs
            and vol_ratio >= pullback_min_vol
        )

        # -------- B) Fresh 20-day breakout --------
        breakout_extension = (
            close / high20 - 1.0
            if high20 > 0
            else 99.0
        )
        breakout_signal = bool(
            close > high20
            and close > open_
            and 0.0 <= breakout_extension
            <= cfg.max_breakout_extension_pct / 100.0
            and extension
            <= cfg.max_extension_from_sma20_pct / 100.0
            and rs_pct >= breakout_min_rs
            and vol_ratio >= breakout_min_vol
        )

        if not (pullback_signal or breakout_signal):
            continue

        # If both trigger on the same day, classify as breakout.
        setup_type = (
            "BREAKOUT_20D"
            if breakout_signal
            else "PULLBACK_RECOVERY"
        )

        # Score is ranking only; it is deliberately much less restrictive than V1.
        if setup_type == "BREAKOUT_20D":
            setup_bonus = 18.0
            location_quality = max(
                0.0,
                12.0 - max(0.0, breakout_extension * 100.0) * 2.0,
            )
        else:
            setup_bonus = 12.0
            location_quality = max(
                0.0,
                14.0 - abs(pullback * 100.0 - 5.0) * 1.5,
            )

        score = (
            rs_pct * 40.0
            + min(16.0, max(0.0, (ret20 + 0.02) * 100.0))
            + min(14.0, max(0.0, (vol_ratio - 0.5) * 10.0))
            + setup_bonus
            + location_quality
        )
        if score < cfg.min_score:
            continue

        rows.append(
            {
                "code": code,
                "name": names.get(code, code),
                "signal_date": day.strftime("%Y-%m-%d"),
                "signal_close": close,
                "setup_type": setup_type,
                "score": round(score, 3),
                "rs_percentile": round(rs_pct, 4),
                "ret20": round(ret20, 5),
                "vol_ratio": round(vol_ratio, 4),
                "pullback_pct": round(pullback * 100.0, 3),
                "extension_sma20_pct": round(extension * 100.0, 3),
                "breakout_extension_pct": (
                    round(breakout_extension * 100.0, 3)
                    if high20 > 0 else None
                ),
                "atr14": float(r["ATR14"]),
                "swing_low5": low5,
                "market_regime": regime,
            }
        )

    rows.sort(
        key=lambda x: (
            x["score"],
            x["setup_type"] == "BREAKOUT_20D",
            x["rs_percentile"],
            x["vol_ratio"],
        ),
        reverse=True,
    )
    return rows

def _sell_fill(base_price: float, cfg: SwingConfig) -> float:
    return float(base_price) * (
        1.0 - cfg.sell_slippage_pct / 100.0
    )


def _buy_fill(base_price: float, cfg: SwingConfig) -> float:
    return float(base_price) * (
        1.0 + cfg.buy_slippage_pct / 100.0
    )


def _initial_stop(
    entry: float,
    atr: float,
    swing_low5: float,
    cfg: SwingConfig,
) -> float:
    atr_stop = entry - cfg.atr_stop_multiple * atr
    struct_stop = swing_low5 * 0.995
    raw = max(atr_stop, struct_stop)

    # Never tighter than min_stop_pct and never looser than max_stop_pct.
    raw = min(
        raw,
        entry * (1.0 - cfg.min_stop_pct / 100.0),
    )
    raw = max(
        raw,
        entry * (1.0 - cfg.max_stop_pct / 100.0),
    )
    return max(1.0, float(raw))


def _max_drawdown(equity_curve: list[dict]) -> tuple[float, str, str]:
    peak = -float("inf")
    peak_date = ""
    worst = 0.0
    worst_peak = ""
    worst_trough = ""
    for row in equity_curve:
        eq = float(row["equity_krw"])
        if eq > peak:
            peak = eq
            peak_date = row["date"]
        dd = peak - eq
        if dd > worst:
            worst = dd
            worst_peak = peak_date
            worst_trough = row["date"]
    return worst, worst_peak, worst_trough


def _run_backtest(
    frames: dict[str, pd.DataFrame],
    names: dict[str, str],
    cfg: SwingConfig,
    reference_result: dict,
) -> dict:
    # Trading calendar: Samsung first, otherwise union.
    anchor = frames.get("005930")
    if anchor is not None and not anchor.empty:
        days = list(anchor.index)
    else:
        s = set()
        for df in frames.values():
            s.update(df.index.tolist())
        days = sorted(s)

    days = [
        pd.Timestamp(d)
        for d in days
        if pd.Timestamp(cfg.start_date)
        <= pd.Timestamp(d)
        <= pd.Timestamp(cfg.end_date)
        and pd.Timestamp(d).weekday() < 5
    ]

    cash = float(cfg.initial_capital_krw)
    positions: dict[str, dict] = {}
    pending_entries: list[dict] = []
    pending_exits: dict[str, str] = {}
    trades: list[dict] = []
    equity_curve: list[dict] = []
    cooldown_until_idx: dict[str, int] = {}

    diagnostics = {
        "signals": 0,
        "entries": 0,
        "pullback_entries": 0,
        "breakout_entries": 0,
        "risk_off_breakout_entries": 0,
        "risk_off_days": 0,
        "risk_off_signal_blocks": 0,
        "neutral_days": 0,
        "risk_on_days": 0,
        "gap_skips": 0,
        "cash_or_risk_size_skips": 0,
        "cooldown_blocks": 0,
        "slot_blocks": 0,
        "stop_exits": 0,
        "trend_exits": 0,
        "time_exits": 0,
        "max_hold_exits": 0,
        "end_mark_exits": 0,
        "breakeven_raises": 0,
        "trailing_raises": 0,
    }

    regime_daily: list[dict] = []

    def close_trade(
        code: str,
        exit_date: pd.Timestamp,
        exit_price: float,
        reason: str,
    ) -> None:
        nonlocal cash
        pos = positions.pop(code)
        proceeds = exit_price * pos["qty"]
        cash += proceeds
        pnl = (
            (exit_price - pos["entry_price"])
            * pos["qty"]
        )
        ret = (
            exit_price / pos["entry_price"] - 1.0
        )
        trades.append(
            {
                "code": code,
                "name": pos["name"],
                "entry_date": pos["entry_date"],
                "exit_date": exit_date.strftime("%Y-%m-%d"),
                "qty": int(pos["qty"]),
                "entry_price": round(pos["entry_price"], 4),
                "exit_price": round(exit_price, 4),
                "initial_stop": round(pos["initial_stop"], 4),
                "final_stop": round(pos["stop"], 4),
                "hold_days": int(pos["hold_days"]),
                "pnl_krw": int(round(pnl)),
                "return_pct": round(ret * 100.0, 3),
                "reason": reason,
                "market_regime_at_entry": pos["market_regime"],
                "setup_type": pos.get("setup_type", ""),
                "entry_score": pos["entry_score"],
                "rs_percentile": pos["rs_percentile"],
                "mfe_pct": round(pos["mfe_pct"], 3),
                "mae_pct": round(pos["mae_pct"], 3),
                "initial_risk_krw": int(
                    round(pos["initial_risk_krw"])
                ),
            }
        )

    for i, day in enumerate(days):
        # 1) Next-open exits decided from previous close.
        for code, reason in list(pending_exits.items()):
            if code not in positions:
                pending_exits.pop(code, None)
                continue
            df = frames.get(code)
            if df is None or day not in df.index:
                continue
            row = df.loc[day]
            exit_price = _sell_fill(float(row["Open"]), cfg)
            close_trade(code, day, exit_price, reason)
            pending_exits.pop(code, None)
            cooldown_until_idx[code] = i + cfg.cooldown_trading_days

            if reason == "TREND_EXIT":
                diagnostics["trend_exits"] += 1
            elif reason == "TIME_STOP":
                diagnostics["time_exits"] += 1
            elif reason == "MAX_HOLD":
                diagnostics["max_hold_exits"] += 1

        # 2) Execute entries generated from previous close.
        todays_pending = list(pending_entries)
        pending_entries = []

        entries_today = 0
        for sig in todays_pending:
            code = sig["code"]
            if code in positions:
                continue
            if i <= int(cooldown_until_idx.get(code, -1)):
                diagnostics["cooldown_blocks"] += 1
                continue

            market_limit = (
                cfg.max_positions
                if sig["market_regime"] == "RISK_ON"
                else (
                    cfg.neutral_max_positions
                    if sig["market_regime"] == "NEUTRAL"
                    else cfg.risk_off_max_positions
                )
            )
            if (
                len(positions) >= market_limit
                or entries_today >= cfg.max_new_entries_per_day
            ):
                diagnostics["slot_blocks"] += 1
                continue

            df = frames.get(code)
            if df is None or day not in df.index:
                continue
            row = df.loc[day]
            raw_open = float(row["Open"])
            gap_pct = (
                raw_open / float(sig["signal_close"]) - 1.0
            ) * 100.0
            if (
                gap_pct > cfg.max_entry_gap_up_pct
                or gap_pct < cfg.max_entry_gap_down_pct
            ):
                diagnostics["gap_skips"] += 1
                continue

            entry = _buy_fill(raw_open, cfg)
            stop = _initial_stop(
                entry,
                float(sig["atr14"]),
                float(sig["swing_low5"]),
                cfg,
            )
            risk_per_share = max(1.0, entry - stop)

            # Equity at open, approximated by cash + held positions at open.
            open_equity = cash
            for c2, p2 in positions.items():
                df2 = frames.get(c2)
                if df2 is not None and day in df2.index:
                    open_equity += (
                        float(df2.loc[day]["Open"]) * p2["qty"]
                    )
                else:
                    open_equity += p2["entry_price"] * p2["qty"]

            risk_budget = (
                open_equity * cfg.risk_per_trade_pct / 100.0
            )
            qty_by_risk = int(risk_budget // risk_per_share)
            qty_by_cap = int(cfg.per_stock_cap_krw // entry)
            qty_by_cash = int(cash // entry)
            qty = min(qty_by_risk, qty_by_cap, qty_by_cash)

            if qty <= 0:
                diagnostics["cash_or_risk_size_skips"] += 1
                continue

            cost = entry * qty
            cash -= cost
            initial_risk = (entry - stop) * qty
            positions[code] = {
                "code": code,
                "name": sig["name"],
                "entry_date": day.strftime("%Y-%m-%d"),
                "entry_price": entry,
                "qty": qty,
                "initial_stop": stop,
                "stop": stop,
                "initial_r_per_share": entry - stop,
                "initial_risk_krw": initial_risk,
                "highest_close": entry,
                "highest_high": entry,
                "hold_days": 0,
                "below_sma20_streak": 0,
                "market_regime": sig["market_regime"],
                "setup_type": sig.get("setup_type", ""),
                "entry_score": sig["score"],
                "rs_percentile": sig["rs_percentile"],
                "mfe_pct": 0.0,
                "mae_pct": 0.0,
            }
            diagnostics["entries"] += 1
            if sig.get("setup_type") == "BREAKOUT_20D":
                diagnostics["breakout_entries"] += 1
                if sig.get("market_regime") == "RISK_OFF":
                    diagnostics["risk_off_breakout_entries"] += 1
            else:
                diagnostics["pullback_entries"] += 1
            entries_today += 1

        # 3) Same-day stop management, including new entries.
        for code in list(positions.keys()):
            pos = positions.get(code)
            if pos is None:
                continue
            df = frames.get(code)
            if df is None or day not in df.index:
                continue
            row = df.loc[day]
            o = float(row["Open"])
            h = float(row["High"])
            l = float(row["Low"])
            c = float(row["Close"])

            pos["hold_days"] += 1
            pos["highest_high"] = max(pos["highest_high"], h)
            pos["highest_close"] = max(pos["highest_close"], c)

            mfe = (
                pos["highest_high"] / pos["entry_price"] - 1.0
            ) * 100.0
            mae = (
                l / pos["entry_price"] - 1.0
            ) * 100.0
            pos["mfe_pct"] = max(pos["mfe_pct"], mfe)
            pos["mae_pct"] = min(pos["mae_pct"], mae)

            stop = float(pos["stop"])
            if o <= stop:
                exit_price = _sell_fill(o, cfg)
                close_trade(code, day, exit_price, "STOP_GAP")
                diagnostics["stop_exits"] += 1
                pending_exits.pop(code, None)
                cooldown_until_idx[code] = (
                    i + cfg.cooldown_trading_days
                )
                continue
            if l <= stop:
                exit_price = _sell_fill(stop, cfg)
                close_trade(code, day, exit_price, "STOP")
                diagnostics["stop_exits"] += 1
                pending_exits.pop(code, None)
                cooldown_until_idx[code] = (
                    i + cfg.cooldown_trading_days
                )
                continue

        # 4) End-of-day management and next-open exit decisions.
        for code, pos in list(positions.items()):
            df = frames.get(code)
            if df is None or day not in df.index:
                continue
            row = df.loc[day]
            close = float(row["Close"])
            atr = float(row.get("ATR14") or 0.0)
            sma20 = float(row.get("SMA20") or 0.0)
            sma50 = float(row.get("SMA50") or 0.0)

            if sma20 > 0 and close < sma20:
                pos["below_sma20_streak"] += 1
            else:
                pos["below_sma20_streak"] = 0

            r = float(pos["initial_r_per_share"])
            if r > 0:
                gain_r = (
                    close - pos["entry_price"]
                ) / r

                if gain_r >= cfg.breakeven_at_r:
                    new_stop = max(
                        pos["stop"],
                        pos["entry_price"] * 1.001,
                    )
                    if new_stop > pos["stop"] + 1e-9:
                        pos["stop"] = new_stop
                        diagnostics["breakeven_raises"] += 1

                if (
                    gain_r >= cfg.trailing_start_r
                    and atr > 0
                ):
                    trail = (
                        pos["highest_close"]
                        - cfg.trailing_atr_multiple * atr
                    )
                    new_stop = max(pos["stop"], trail)
                    if new_stop > pos["stop"] + 1e-9:
                        pos["stop"] = new_stop
                        diagnostics["trailing_raises"] += 1

            if code in pending_exits:
                continue

            if (
                pos["below_sma20_streak"]
                >= cfg.below_sma20_exit_days
            ):
                pending_exits[code] = "TREND_EXIT"
                continue

            if sma50 > 0 and close < sma50:
                pending_exits[code] = "TREND_EXIT"
                continue

            ret_pct = (
                close / pos["entry_price"] - 1.0
            ) * 100.0
            if (
                pos["hold_days"] >= cfg.time_stop_days
                and ret_pct < cfg.time_stop_min_return_pct
            ):
                pending_exits[code] = "TIME_STOP"
                continue

            if pos["hold_days"] >= cfg.max_hold_days:
                pending_exits[code] = "MAX_HOLD"

        # 5) Market regime and next-day signals.
        market = _market_snapshot(day, frames, cfg)
        regime_daily.append(
            {
                "date": day.strftime("%Y-%m-%d"),
                **market,
            }
        )
        if market["regime"] == "RISK_ON":
            diagnostics["risk_on_days"] += 1
        elif market["regime"] == "NEUTRAL":
            diagnostics["neutral_days"] += 1
        else:
            diagnostics["risk_off_days"] += 1

        candidates = _candidate_rows(
            day, frames, names, market, cfg
        )
        diagnostics["signals"] += len(candidates)

        if market["regime"] == "RISK_OFF":
            diagnostics["risk_off_signal_blocks"] += len(candidates)
        else:
            existing_or_pending = (
                set(positions.keys())
                | {x["code"] for x in pending_entries}
            )
            for sig in candidates:
                if sig["code"] in existing_or_pending:
                    continue
                if i <= int(
                    cooldown_until_idx.get(sig["code"], -1)
                ):
                    diagnostics["cooldown_blocks"] += 1
                    continue
                pending_entries.append(sig)
                existing_or_pending.add(sig["code"])
                if (
                    len(pending_entries)
                    >= cfg.max_new_entries_per_day
                ):
                    break

        # 6) Mark-to-market equity at close.
        equity = cash
        for code, pos in positions.items():
            df = frames.get(code)
            if df is not None and day in df.index:
                equity += float(df.loc[day]["Close"]) * pos["qty"]
            else:
                equity += pos["entry_price"] * pos["qty"]

        equity_curve.append(
            {
                "date": day.strftime("%Y-%m-%d"),
                "cash_krw": int(round(cash)),
                "equity_krw": int(round(equity)),
                "positions": len(positions),
                "market_regime": market["regime"],
            }
        )

        _state(
            status="running",
            phase="BACKTEST",
            total_days=len(days),
            completed_days=i + 1,
            progress_pct=round(
                100.0 * (i + 1) / max(1, len(days)), 1
            ),
            current_date=day.strftime("%Y-%m-%d"),
            result_ready=False,
            message=(
                f"SWING V2 {i + 1}/{len(days)} · "
                f"{day.strftime('%Y-%m-%d')}"
            ),
        )

    # Force-close any remaining positions at last close for a finite test.
    if days:
        last_day = days[-1]
        for code in list(positions.keys()):
            df = frames.get(code)
            if df is None or last_day not in df.index:
                continue
            exit_price = _sell_fill(
                float(df.loc[last_day]["Close"]), cfg
            )
            close_trade(
                code,
                last_day,
                exit_price,
                "END_OF_TEST_MARK",
            )
            diagnostics["end_mark_exits"] += 1

        final_equity = cash
        if equity_curve:
            equity_curve[-1]["equity_krw"] = int(round(final_equity))
            equity_curve[-1]["cash_krw"] = int(round(final_equity))
            equity_curve[-1]["positions"] = 0
    else:
        final_equity = cash

    net = final_equity - cfg.initial_capital_krw
    gross_profit = sum(
        max(0, int(t["pnl_krw"]))
        for t in trades
    )
    gross_loss_abs = sum(
        abs(min(0, int(t["pnl_krw"])))
        for t in trades
    )
    wins = [t for t in trades if int(t["pnl_krw"]) > 0]
    losses = [t for t in trades if int(t["pnl_krw"]) < 0]

    avg_win = (
        sum(t["pnl_krw"] for t in wins) / len(wins)
        if wins else 0.0
    )
    avg_loss_abs = (
        abs(sum(t["pnl_krw"] for t in losses) / len(losses))
        if losses else 0.0
    )
    payoff = (
        avg_win / avg_loss_abs
        if avg_loss_abs > 0
        else None
    )
    profit_factor = (
        gross_profit / gross_loss_abs
        if gross_loss_abs > 0
        else None
    )

    mdd, mdd_peak, mdd_trough = _max_drawdown(equity_curve)

    monthly: dict[str, int] = {}
    for t in trades:
        month = str(t["exit_date"])[:7]
        monthly[month] = monthly.get(month, 0) + int(t["pnl_krw"])

    monthly_rows = [
        {"month": m, "realized_pnl_krw": monthly[m]}
        for m in sorted(monthly)
    ]
    positive_months = sum(
        1 for x in monthly_rows
        if x["realized_pnl_krw"] > 0
    )

    # Diagnostic-only breakdown. This does not change entries, exits, sizing,
    # market-regime rules, or any other trading behavior.
    def bucket_stats(rows: list[dict]) -> dict:
        gp = sum(max(0, int(x["pnl_krw"])) for x in rows)
        gl = sum(abs(min(0, int(x["pnl_krw"]))) for x in rows)
        bucket_wins = [x for x in rows if int(x["pnl_krw"]) > 0]
        bucket_losses = [x for x in rows if int(x["pnl_krw"]) < 0]
        avg_bucket_win = (
            sum(int(x["pnl_krw"]) for x in bucket_wins) / len(bucket_wins)
            if bucket_wins else 0.0
        )
        avg_bucket_loss_abs = (
            abs(sum(int(x["pnl_krw"]) for x in bucket_losses) / len(bucket_losses))
            if bucket_losses else 0.0
        )
        return {
            "trades": len(rows),
            "wins": len(bucket_wins),
            "losses": len(bucket_losses),
            "win_rate_pct": round(
                100.0 * len(bucket_wins) / max(1, len(rows)), 2
            ),
            "net_krw": sum(int(x["pnl_krw"]) for x in rows),
            "gross_profit_krw": int(gp),
            "gross_loss_krw": -int(gl),
            "profit_factor": round(gp / gl, 4) if gl else None,
            "avg_win_krw": round(avg_bucket_win, 1),
            "avg_loss_krw": round(-avg_bucket_loss_abs, 1),
            "payoff_ratio": (
                round(avg_bucket_win / avg_bucket_loss_abs, 4)
                if avg_bucket_loss_abs > 0 else None
            ),
        }

    diagnostic_breakdown = {
        "by_setup": {
            setup: bucket_stats([
                t for t in trades
                if t.get("setup_type") == setup
            ])
            for setup in ("PULLBACK_RECOVERY", "BREAKOUT_20D")
        },
        "by_regime": {
            regime: bucket_stats([
                t for t in trades
                if t.get("market_regime_at_entry") == regime
            ])
            for regime in ("RISK_ON", "NEUTRAL", "RISK_OFF")
        },
        "by_setup_regime": {
            f"{setup}_{regime}": bucket_stats([
                t for t in trades
                if t.get("setup_type") == setup
                and t.get("market_regime_at_entry") == regime
            ])
            for setup in ("PULLBACK_RECOVERY", "BREAKOUT_20D")
            for regime in ("RISK_ON", "NEUTRAL", "RISK_OFF")
        },
    }

    ref_d2 = int(
        ((reference_result.get("overall") or {})
         .get("D2_total_KRW", 0))
        or 0
    )

    research_candidate = bool(
        net > 0
        and (profit_factor or 0) >= 1.15
        and (payoff or 0) >= 1.30
        and mdd <= 1_500_000
        and positive_months >= 4
    )

    return {
        "ok": True,
        "version": SWING_VERSION,
        "strategy": "SWING_V2_DUAL_ENTRY",
        "period": {
            "start": cfg.start_date,
            "end": cfg.end_date,
            "trading_days": len(days),
        },
        "capital": {
            "initial_krw": cfg.initial_capital_krw,
            "final_krw": int(round(final_equity)),
            "net_krw": int(round(net)),
            "return_pct": round(
                100.0 * net / cfg.initial_capital_krw, 3
            ),
        },
        "performance": {
            "trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": round(
                100.0 * len(wins) / max(1, len(trades)), 2
            ),
            "gross_profit_krw": int(gross_profit),
            "gross_loss_krw": -int(gross_loss_abs),
            "profit_factor": (
                round(profit_factor, 4)
                if profit_factor is not None
                else None
            ),
            "avg_win_krw": round(avg_win, 1),
            "avg_loss_krw": round(-avg_loss_abs, 1),
            "payoff_ratio": (
                round(payoff, 4)
                if payoff is not None
                else None
            ),
            "avg_hold_days": round(
                sum(t["hold_days"] for t in trades)
                / max(1, len(trades)),
                2,
            ),
            "max_hold_days": max(
                [t["hold_days"] for t in trades] or [0]
            ),
        },
        "risk": {
            "max_drawdown_krw": int(round(mdd)),
            "mdd_peak_date": mdd_peak,
            "mdd_trough_date": mdd_trough,
            "risk_per_trade_pct": cfg.risk_per_trade_pct,
            "max_positions": cfg.max_positions,
        },
        "monthly": monthly_rows,
        "positive_months": positive_months,
        "diagnostics": diagnostics,
        "diagnostic_breakdown": diagnostic_breakdown,
        "research_candidate": research_candidate,
        "reference_only": {
            "D2_total_KRW": ref_d2,
            "swing_minus_D2_KRW": int(round(net - ref_d2)),
            "warning": (
                "Different strategy family / holding period; "
                "comparison is reference only."
            ),
        },
        "rules": {
            "signal_timing": (
                "signal at T close; buy at T+1 open"
            ),
            "market": (
                "RISK_ON/NEUTRAL allow both routes; RISK_OFF allows only "
                "high-RS/high-volume 20D breakout with max 1 position"
            ),
            "trend": "Close>SMA20>SMA60 and rising SMA20",
            "entry_A": (
                "PULLBACK_RECOVERY: recent pullback 1%~12%, close above SMA10, "
                "then prior-high / near-3D-high recovery"
            ),
            "entry_B": (
                "BREAKOUT_20D: fresh previous-20-day-high breakout; "
                "pullback not required"
            ),
            "relative_strength": (
                "RISK_ON relaxed; NEUTRAL stricter; RISK_OFF breakout only"
            ),
            "entry": (
                "max 3 positions; max 2 new/day; target research sample 20~40 trades"
            ),
            "size": (
                "min(3M KRW cap, 1% equity risk sizing, available cash)"
            ),
            "initial_stop": (
                "ATR/5-day swing structure, bounded 3.5%~8%"
            ),
            "profit_exit": (
                "no fixed +3/+5 exit; breakeven at +1R, "
                "ATR trail after +2R"
            ),
            "trend_exit": (
                "2 closes below SMA20 or close below SMA50"
            ),
            "time_stop": (
                "after 5 days if gain <2%; max hold 20 days"
            ),
            "cooldown": "3 trading days after exit",
        },
        "assumptions": {
            "real_orders": False,
            "data": "KIS adjusted daily OHLCV FHKST03010100",
            "future_data_visible": False,
            "buy_slippage_pct": cfg.buy_slippage_pct,
            "sell_slippage_pct": cfg.sell_slippage_pct,
            "fees_taxes": cfg.fees_taxes,
            "candidate_reconstruction": (
                "same fixed liquidity universe used by historical KR replay"
            ),
            "important_limit": (
                "not the exact historical whole-market KIS ranking universe"
            ),
            "purpose": (
                "V2 dual-entry research; target is a usable sample, not production approval"
            ),
        },
        "config": asdict(cfg),
        "trades": trades,
        "equity_curve": equity_curve,
        "market_regime_daily": regime_daily,
    }


def _job(
    reference_result: dict,
    protected_window_fn,
    cfg: SwingConfig,
) -> None:
    try:
        _STOP.clear()
        _state(
            status="running",
            phase="INIT",
            result_ready=False,
            error_days=0,
            last_error="",
            started_at=datetime.now(KST).isoformat(timespec="seconds"),
            total_symbols=len(_universe()),
            message="SWING V2 준비 중",
        )

        settings = Settings.from_env()
        # Historical market data: read-only real quotation API.
        client = KISClient(settings=settings, env="real")
        try:
            client._rest_min_interval = max(
                float(getattr(client, "_rest_min_interval", 0.0)),
                float(cfg.kis_min_interval_seconds),
            )
        except Exception:
            pass
        client.get_token()

        universe = _universe()
        names = {
            str(code).zfill(6): str(name)
            for code, name, _exch in universe
        }
        frames: dict[str, pd.DataFrame] = {}
        errors: list[dict] = []

        total = len(universe)
        for idx, (code, name, _exch) in enumerate(universe, start=1):
            if _STOP.is_set():
                raise RuntimeError("swing backtest stopped")
            try:
                df = _fetch_daily_ohlcv(
                    client,
                    str(code).zfill(6),
                    cfg,
                    protected_window_fn,
                )
                if not df.empty:
                    frames[str(code).zfill(6)] = _add_features(df)
                else:
                    errors.append(
                        {
                            "code": str(code).zfill(6),
                            "name": name,
                            "error": "empty daily history",
                        }
                    )
            except Exception as exc:
                errors.append(
                    {
                        "code": str(code).zfill(6),
                        "name": name,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

            _state(
                status="running",
                phase="DAILY_DATA",
                total_symbols=total,
                completed_symbols=idx,
                symbol_progress_pct=round(
                    100.0 * idx / max(1, total), 1
                ),
                current_symbol=str(code).zfill(6),
                current_symbol_name=name,
                error_symbols=len(errors),
                result_ready=False,
                message=f"SWING 일봉 준비 {idx}/{total}",
            )

        if not frames:
            raise RuntimeError("usable swing daily frames = 0")

        result = _run_backtest(
            frames,
            names,
            cfg,
            reference_result,
        )
        result["data_errors"] = errors
        result["completed_at"] = datetime.now(KST).isoformat(
            timespec="seconds"
        )
        _atomic_json(RESULT_FILE, result)

        _state(
            status="done",
            phase="DONE",
            result_ready=True,
            total_days=result["period"]["trading_days"],
            completed_days=result["period"]["trading_days"],
            progress_pct=100.0,
            error_symbols=len(errors),
            last_error="",
            message="SWING V2 검증 완료",
        )
    except Exception as exc:
        _state(
            status="error",
            phase="ERROR",
            result_ready=False,
            last_error=f"{type(exc).__name__}: {exc}",
            message="SWING V2 오류",
        )


def ensure_swing_v2_started(
    result: dict,
    provider=None,
    codes=None,
    frozen_config=None,
    protected_window_fn=None,
) -> dict:
    global _THREAD

    cfg = SwingConfig()
    existing = _read_json(RESULT_FILE, {}) or {}

    if (
        existing.get("ok") is True
        and existing.get("version") == SWING_VERSION
        and (existing.get("period") or {}).get("start") == cfg.start_date
        and (existing.get("period") or {}).get("end") == cfg.end_date
    ):
        compact = dict(existing)
        compact.pop("trades", None)
        compact.pop("equity_curve", None)
        compact.pop("market_regime_daily", None)
        compact["result_ready"] = True
        compact["status"] = "done"
        return compact

    if protected_window_fn is None:
        protected_window_fn = lambda: (False, "")

    with _LOCK:
        if _THREAD and _THREAD.is_alive():
            return _public_state()

        _THREAD = threading.Thread(
            target=_job,
            args=(dict(result or {}), protected_window_fn, cfg),
            daemon=True,
            name="kr-swing-v2",
        )
        _THREAD.start()

        out = _public_state()
        out["started"] = True
        return out

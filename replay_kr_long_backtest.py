from __future__ import annotations

"""
D-v2 장기 백테스트 러너.

목적
- 현재 고정된 D-v2 규칙을 바꾸지 않고 긴 구간에 그대로 적용한다.
- yfinance의 오래된 1분봉 제한을 피하기 위해 KIS '주식일별분봉조회'를 읽기 전용으로 사용한다.
- 실제 주문은 절대 보내지 않는다.
- Railway Volume에 원천 분봉/일봉/일별 결과를 캐시해 재시작 후 이어서 실행할 수 있다.

주의
- 과거 당시의 KIS 전체시장 실시간 TOP5 원본은 없으므로 기존 replay_kr와 동일한
  고정 유동성 종목군을 사용한다. 따라서 실제 당시 전체시장 후보를 100% 복원하는 백테스트가 아니다.
"""

import gzip
import json
import os
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from trader_core import Settings, KISClient
import replay_kr
import replay_kr_open_defense_v2 as d2_replay

KST = ZoneInfo("Asia/Seoul")
ET = ZoneInfo("America/New_York")
LONG_BACKTEST_VERSION = "kr-d2-long-backtest-v1"


def _resolve_state_dir() -> Path:
    explicit = os.getenv("SONG_TRADER_STATE_DIR", "").strip()
    if explicit:
        return Path(explicit)
    railway_volume = os.getenv("RAILWAY_VOLUME_MOUNT_PATH", "").strip()
    if railway_volume:
        return Path(railway_volume) / "song_trader_v2"
    return Path("/tmp/song_trader_v2")


ROOT = _resolve_state_dir() / "replays" / "kr_d2_long"
RAW_MINUTE_DIR = ROOT / "kis_minute"
DAILY_HISTORY_DIR = ROOT / "kis_daily"
DAY_RESULT_DIR = ROOT / "daily_results"
STATE_FILE = ROOT / "state.json"
RESULT_FILE = ROOT / "result.json"
for _p in (ROOT, RAW_MINUTE_DIR, DAILY_HISTORY_DIR, DAY_RESULT_DIR):
    _p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class LongBacktestConfig:
    start_date: str = "2026-01-01"
    end_date: str = "2026-08-09"
    # 공식 일별분봉 API는 한 호출 최대 120건. 아래 4개 시점으로 정규장 하루를 겹침 없이 덮는다.
    minute_anchor_times: tuple[str, ...] = ("093000", "113000", "133000", "153000")
    min_expected_bars: int = 200
    # 메인 Worker/주문 안정성을 위해 장기 수집은 보수적으로 속도를 제한한다.
    kis_min_interval_seconds: float = 0.22
    max_api_attempts: int = 5
    pause_during_live_windows: bool = True


_LOCK = threading.RLock()
_THREAD: threading.Thread | None = None
_STOP = threading.Event()


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _state(**updates) -> dict:
    with _LOCK:
        cur = _read_json(STATE_FILE, {}) or {}
        cur.update(updates)
        cur.setdefault("version", LONG_BACKTEST_VERSION)
        cur["updated_at"] = datetime.now(KST).isoformat(timespec="seconds")
        _atomic_write_json(STATE_FILE, cur)
        return cur


def _public_state() -> dict:
    state = _read_json(STATE_FILE, {}) or {}
    if not state:
        return {
            "ok": True,
            "version": LONG_BACKTEST_VERSION,
            "status": "not_started",
            "message": "장기 백테스트가 아직 시작되지 않았습니다.",
        }
    state = dict(state)
    state["ok"] = True
    state["thread_alive"] = bool(_THREAD and _THREAD.is_alive())
    return state


def _in_protected_live_window() -> tuple[bool, str]:
    if not LongBacktestConfig().pause_during_live_windows:
        return False, ""
    kr = datetime.now(KST)
    us = datetime.now(ET)
    kr_live = kr.weekday() < 5 and (8, 20) <= (kr.hour, kr.minute) < (15, 40)
    us_live = us.weekday() < 5 and (9, 20) <= (us.hour, us.minute) < (16, 10)
    if kr_live:
        return True, "KR_LIVE"
    if us_live:
        return True, "US_LIVE"
    return False, ""


def _wait_if_live() -> None:
    while not _STOP.is_set():
        live, label = _in_protected_live_window()
        if not live:
            return
        _state(
            status="paused_live_window",
            phase="PAUSED",
            pause_reason=label,
            message="실시간 자동매매 보호를 위해 장기 백테스트를 잠시 멈췄습니다. 장 종료 후 자동 재개합니다.",
        )
        _STOP.wait(30.0)


def _validate_date_text(value: str) -> str:
    ts = pd.Timestamp(str(value).strip())
    if pd.isna(ts):
        raise ValueError("날짜 형식이 올바르지 않습니다.")
    return ts.strftime("%Y-%m-%d")


def _universe() -> list[tuple[str, str, str]]:
    return replay_kr._normalize_universe(None)


def _codes() -> list[str]:
    return [str(x[0]).zfill(6) for x in _universe()]


def _api_get(client: KISClient, path: str, tr_id: str, params: dict, cfg: LongBacktestConfig) -> dict:
    last: dict = {}
    for attempt in range(int(cfg.max_api_attempts)):
        _wait_if_live()
        if _STOP.is_set():
            raise RuntimeError("backtest stopped")
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
            last = {"rt_cd": "EXCEPTION", "msg1": f"{type(exc).__name__}: {exc}"}
        if attempt < int(cfg.max_api_attempts) - 1:
            time.sleep(min(8.0, 0.8 * (2 ** attempt)))
    return last


def _daily_cache_path(code: str) -> Path:
    return DAILY_HISTORY_DIR / f"{code}.json.gz"


def _save_gzip_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, default=str)
    tmp.replace(path)


def _load_gzip_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _daily_history_for_symbol(
    client: KISClient,
    code: str,
    start_date: str,
    end_date: str,
    cfg: LongBacktestConfig,
) -> dict[str, float]:
    path = _daily_cache_path(code)
    cached = _load_gzip_json(path, {}) or {}
    if isinstance(cached, dict):
        # 필요한 구간의 전 거래일이 포함됐는지 느슨하게 확인한다.
        keys = sorted(k for k in cached.keys() if len(str(k)) == 10)
        end_tolerance = (pd.Timestamp(end_date) - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
        if keys and keys[0] <= (pd.Timestamp(start_date) - pd.Timedelta(days=3)).strftime("%Y-%m-%d") and keys[-1] >= end_tolerance:
            return {str(k): float(v) for k, v in cached.items() if float(v or 0) > 0}

    floor = (pd.Timestamp(start_date) - pd.Timedelta(days=20)).strftime("%Y%m%d")
    cursor_end = pd.Timestamp(end_date).strftime("%Y%m%d")
    out: dict[str, float] = {}

    for _ in range(6):
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
                # 수정주가를 사용해 액면분할 등 권리 발생 시 전일비 왜곡을 줄인다.
                "FID_ORG_ADJ_PRC": "0",
            },
            cfg,
        )
        rows = data.get("output2") or []
        if isinstance(rows, dict):
            rows = [rows]
        parsed_dates: list[pd.Timestamp] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            raw_date = str(row.get("stck_bsop_date", "")).strip()
            raw_close = row.get("stck_clpr", row.get("stck_prpr", 0))
            if len(raw_date) != 8:
                continue
            try:
                dt = pd.Timestamp(raw_date)
                close = float(str(raw_close).replace(",", ""))
            except Exception:
                continue
            if close <= 0:
                continue
            key = dt.strftime("%Y-%m-%d")
            out[key] = close
            parsed_dates.append(dt)

        if not parsed_dates:
            break
        oldest = min(parsed_dates)
        if oldest.strftime("%Y%m%d") <= floor:
            break
        cursor_end = (oldest - pd.Timedelta(days=1)).strftime("%Y%m%d")
        if cursor_end < floor:
            break

    if out:
        _save_gzip_json(path, out)
    return out


def _prepare_daily_histories(
    client: KISClient,
    start_date: str,
    end_date: str,
    cfg: LongBacktestConfig,
) -> tuple[dict[str, dict[str, float]], list[str]]:
    universe = _universe()
    histories: dict[str, dict[str, float]] = {}
    total = len(universe)

    # 삼성전자를 먼저 읽어 실제 거래일 달력의 기준으로 쓴다.
    universe = sorted(universe, key=lambda x: 0 if str(x[0]) == "005930" else 1)
    for idx, (code, name, _exch) in enumerate(universe, start=1):
        _wait_if_live()
        hist = _daily_history_for_symbol(client, str(code).zfill(6), start_date, end_date, cfg)
        histories[str(code).zfill(6)] = hist
        _state(
            status="running",
            phase="DAILY_HISTORY",
            phase_current=idx,
            phase_total=total,
            current_symbol=str(code).zfill(6),
            current_symbol_name=name,
            message=f"거래일/전일종가 준비 중 {idx}/{total}",
        )

    anchor = histories.get("005930", {})
    if not anchor:
        # 삼성전자 데이터가 예외적으로 없다면 전체 종목 일봉 날짜의 합집합으로 폴백한다.
        days = set()
        for hist in histories.values():
            days.update(hist.keys())
    else:
        days = set(anchor.keys())

    trading_dates = sorted(
        d for d in days
        if start_date <= d <= end_date and pd.Timestamp(d).weekday() < 5
    )
    return histories, trading_dates


def _minute_cache_path(date_text: str, code: str) -> Path:
    return RAW_MINUTE_DIR / date_text / f"{code}.json.gz"


def _frame_to_records(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for ts, row in frame.iterrows():
        rows.append({
            "ts": pd.Timestamp(ts).isoformat(),
            "Open": float(row.get("Open", 0) or 0),
            "High": float(row.get("High", 0) or 0),
            "Low": float(row.get("Low", 0) or 0),
            "Close": float(row.get("Close", 0) or 0),
            "Volume": float(row.get("Volume", 0) or 0),
        })
    return rows


def _records_to_frame(records: list[dict], prev_close: float) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if "ts" not in df.columns:
        return pd.DataFrame()
    idx = pd.to_datetime(df.pop("ts"), errors="coerce")
    df.index = pd.DatetimeIndex(idx)
    df = df[~df.index.isna()].copy()
    if df.empty:
        return df
    if df.index.tz is None:
        df.index = df.index.tz_localize(KST)
    else:
        df.index = df.index.tz_convert(KST)
    for c in ("Open", "High", "Low", "Close", "Volume"):
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Close"])
    df = df[df["Close"] > 0].sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if df.empty:
        return df
    df["Volume"] = df["Volume"].fillna(0.0).clip(lower=0.0)
    df["_cum_volume"] = df["Volume"].cumsum()
    df["_cum_amount"] = (df["Close"] * df["Volume"]).cumsum()
    df.attrs["prev_close"] = float(prev_close or 0.0)
    return df


def _parse_minute_rows(rows: list[dict], date_text: str) -> pd.DataFrame:
    parsed = []
    ymd = date_text.replace("-", "")
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_date = str(row.get("stck_bsop_date", row.get("bsop_date", ""))).strip()
        raw_time = str(row.get("stck_cntg_hour", row.get("cntg_hour", ""))).strip().zfill(6)
        if raw_date and raw_date != ymd:
            continue
        if len(raw_time) != 6 or not raw_time.isdigit():
            continue
        try:
            ts = pd.Timestamp(
                f"{date_text} {raw_time[0:2]}:{raw_time[2:4]}:{raw_time[4:6]}",
                tz=KST,
            )
            close = float(str(row.get("stck_prpr", 0)).replace(",", ""))
            open_ = float(str(row.get("stck_oprc", close)).replace(",", ""))
            high = float(str(row.get("stck_hgpr", close)).replace(",", ""))
            low = float(str(row.get("stck_lwpr", close)).replace(",", ""))
            volume = float(str(row.get("cntg_vol", 0)).replace(",", ""))
        except Exception:
            continue
        if close <= 0:
            continue
        parsed.append((ts, open_, high, low, close, max(0.0, volume)))

    if not parsed:
        return pd.DataFrame()
    df = pd.DataFrame(parsed, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
    df = df.drop_duplicates(subset=["ts"], keep="last").set_index("ts").sort_index()
    return df


def _download_symbol_minutes(
    client: KISClient,
    date_text: str,
    code: str,
    prev_close: float,
    cfg: LongBacktestConfig,
) -> pd.DataFrame:
    path = _minute_cache_path(date_text, code)
    cached = _load_gzip_json(path, None)
    if isinstance(cached, dict) and cached.get("records"):
        frame = _records_to_frame(cached.get("records") or [], float(cached.get("prev_close", prev_close) or prev_close))
        if not frame.empty:
            return frame

    all_rows: list[dict] = []
    for anchor in cfg.minute_anchor_times:
        data = _api_get(
            client,
            "/uapi/domestic-stock/v1/quotations/inquire-time-dailychartprice",
            "FHKST03010230",
            {
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": code,
                "FID_INPUT_HOUR_1": anchor,
                "FID_INPUT_DATE_1": date_text.replace("-", ""),
                "FID_PW_DATA_INCU_YN": "N",
                "FID_FAKE_TICK_INCU_YN": "",
            },
            cfg,
        )
        rows = data.get("output2") or []
        if isinstance(rows, dict):
            rows = [rows]
        if rows:
            all_rows.extend(rows)

    frame = _parse_minute_rows(all_rows, date_text)
    if frame.empty:
        return frame
    frame["_cum_volume"] = frame["Volume"].cumsum()
    frame["_cum_amount"] = (frame["Close"] * frame["Volume"]).cumsum()
    frame.attrs["prev_close"] = float(prev_close or 0.0)
    _save_gzip_json(
        path,
        {
            "date": date_text,
            "code": code,
            "prev_close": float(prev_close or 0.0),
            "records": _frame_to_records(frame[["Open", "High", "Low", "Close", "Volume"]]),
        },
    )
    return frame


def _provider_for(
    client: KISClient,
    histories: dict[str, dict[str, float]],
    cfg: LongBacktestConfig,
):
    universe_meta = {str(code).zfill(6): (name, exch) for code, name, exch in _universe()}

    def provider(date_text: str, universe: Iterable[tuple[str, str, str]]):
        frames: dict[str, pd.DataFrame] = {}
        meta: dict[str, dict] = {}
        date_ts = pd.Timestamp(date_text)
        for code, name, exch in universe:
            code = str(code).zfill(6)
            hist = histories.get(code, {})
            # 현재 거래일 바로 이전에 존재하는 일봉 종가를 사용한다.
            before = [(d, c) for d, c in hist.items() if d < date_text and float(c or 0) > 0]
            if not before:
                continue
            prev_date, prev_close = max(before, key=lambda x: x[0])
            frame = _download_symbol_minutes(client, date_text, code, float(prev_close), cfg)
            if frame.empty:
                continue
            day_frame = frame[frame.index.strftime("%Y-%m-%d") == date_text].copy()
            if day_frame.empty:
                continue
            # 아주 짧은 데이터는 후보 재구성 품질이 낮아 제외한다.
            if len(day_frame) < int(cfg.min_expected_bars):
                continue
            day_frame.attrs["prev_close"] = float(prev_close)
            frames[code] = day_frame
            base_name, base_exch = universe_meta.get(code, (name, exch))
            meta[code] = {
                "code": code,
                "name": base_name,
                "exchange": base_exch,
                "ticker": f"{code}.{base_exch}" if base_exch in ("KS", "KQ") else code,
                "prev_close": float(prev_close),
                "prev_close_date": prev_date,
                "source": "KIS_FHKST03010230",
            }
        return frames, meta

    return provider


def _day_result_path(date_text: str) -> Path:
    return DAY_RESULT_DIR / f"{date_text}.json.gz"


def _save_day_result(date_text: str, payload: dict) -> None:
    _save_gzip_json(_day_result_path(date_text), payload)


def _load_day_result(date_text: str) -> dict | None:
    data = _load_gzip_json(_day_result_path(date_text), None)
    if isinstance(data, dict) and data.get("ok") is True and data.get("strategy") == "D2_SELECTIVE_OPEN_DEFENSE":
        return data
    return None


def _max_drawdown(values: list[int]) -> tuple[int, int, int]:
    equity = 0
    peak = 0
    max_dd = 0
    peak_index = 0
    trough_index = 0
    current_peak_index = 0
    for i, pnl in enumerate(values):
        equity += int(pnl)
        if equity > peak:
            peak = equity
            current_peak_index = i
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
            peak_index = current_peak_index
            trough_index = i
    return int(max_dd), int(peak_index), int(trough_index)


def _aggregate(start_date: str, end_date: str, daily_payloads: list[dict], errors: list[dict]) -> dict:
    rows = []
    diag_totals: dict[str, int] = {}
    for p in daily_payloads:
        comp = p.get("comparison", {}) or {}
        diag = p.get("diagnostic", {}) or {}
        c = int(comp.get("C_NO_BUY2실현손익KRW", 0) or 0)
        d = int(comp.get("D2_OPEN_DEFENSE실현손익KRW", 0) or 0)
        rows.append({
            "date": str(p.get("date", "")),
            "C_KRW": c,
            "D2_KRW": d,
            "D2_minus_C_KRW": d - c,
            "D2_positive": d > 0,
            "D2_beat_C": d > c,
        })
        for k, v in diag.items():
            try:
                diag_totals[k] = diag_totals.get(k, 0) + int(v or 0)
            except Exception:
                pass

    rows.sort(key=lambda x: x["date"])
    c_values = [r["C_KRW"] for r in rows]
    d_values = [r["D2_KRW"] for r in rows]
    c_total = int(sum(c_values))
    d_total = int(sum(d_values))
    c_mdd, c_peak_i, c_trough_i = _max_drawdown(c_values)
    d_mdd, d_peak_i, d_trough_i = _max_drawdown(d_values)

    monthly: dict[str, dict] = {}
    for r in rows:
        month = r["date"][:7]
        m = monthly.setdefault(month, {"month": month, "days": 0, "C_KRW": 0, "D2_KRW": 0, "D2_minus_C_KRW": 0, "D2_positive_days": 0, "D2_beat_C_days": 0})
        m["days"] += 1
        m["C_KRW"] += r["C_KRW"]
        m["D2_KRW"] += r["D2_KRW"]
        m["D2_minus_C_KRW"] += r["D2_minus_C_KRW"]
        m["D2_positive_days"] += int(r["D2_KRW"] > 0)
        m["D2_beat_C_days"] += int(r["D2_KRW"] > r["C_KRW"])

    def _extreme(key: str, reverse: bool) -> list[dict]:
        return sorted(rows, key=lambda r: r[key], reverse=reverse)[:10]

    def _date_at(idx: int) -> str:
        return rows[idx]["date"] if rows and 0 <= idx < len(rows) else ""

    result = {
        "ok": True,
        "version": LONG_BACKTEST_VERSION,
        "strategy": "D2_SELECTIVE_OPEN_DEFENSE_FROZEN",
        "period": {"start": start_date, "end": end_date},
        "completed_at": datetime.now(KST).isoformat(timespec="seconds"),
        "data_source": {
            "minute": "KIS 주식일별분봉조회 FHKST03010230",
            "daily_prev_close": "KIS 국내주식기간별시세 FHKST03010100",
            "candidate_reconstruction": "기존 replay_kr와 동일한 고정 유동성 종목군",
            "real_orders": False,
            "future_data_visible": False,
            "slippage": "D-v2 기존 가정 유지: 매수 +0.10%, 매도 -0.10%",
            "fees_taxes": "별도 미포함",
        },
        "overall": {
            "completed_trading_days": len(rows),
            "error_days": len(errors),
            "C_total_KRW": c_total,
            "D2_total_KRW": d_total,
            "D2_minus_C_KRW": d_total - c_total,
            "C_positive_days": sum(1 for x in c_values if x > 0),
            "C_negative_days": sum(1 for x in c_values if x < 0),
            "D2_positive_days": sum(1 for x in d_values if x > 0),
            "D2_negative_days": sum(1 for x in d_values if x < 0),
            "D2_beat_C_days": sum(1 for r in rows if r["D2_KRW"] > r["C_KRW"]),
            "C_beat_D2_days": sum(1 for r in rows if r["C_KRW"] > r["D2_KRW"]),
            "ties": sum(1 for r in rows if r["C_KRW"] == r["D2_KRW"]),
            "D2_average_daily_KRW": round(d_total / len(rows), 1) if rows else 0,
            "C_average_daily_KRW": round(c_total / len(rows), 1) if rows else 0,
            "D2_positive_day_rate_pct": round(100 * sum(1 for x in d_values if x > 0) / len(rows), 2) if rows else 0,
            "D2_beat_C_rate_pct": round(100 * sum(1 for r in rows if r["D2_KRW"] > r["C_KRW"]) / len(rows), 2) if rows else 0,
        },
        "risk": {
            "C_max_cumulative_drawdown_KRW": c_mdd,
            "D2_max_cumulative_drawdown_KRW": d_mdd,
            "C_mdd_peak_date": _date_at(c_peak_i),
            "C_mdd_trough_date": _date_at(c_trough_i),
            "D2_mdd_peak_date": _date_at(d_peak_i),
            "D2_mdd_trough_date": _date_at(d_trough_i),
            "C_worst_day": min(rows, key=lambda r: r["C_KRW"]) if rows else {},
            "D2_worst_day": min(rows, key=lambda r: r["D2_KRW"]) if rows else {},
            "C_best_day": max(rows, key=lambda r: r["C_KRW"]) if rows else {},
            "D2_best_day": max(rows, key=lambda r: r["D2_KRW"]) if rows else {},
        },
        "diagnostic_totals": diag_totals,
        "monthly": list(monthly.values()),
        "best_D2_vs_C_days": _extreme("D2_minus_C_KRW", True),
        "worst_D2_vs_C_days": _extreme("D2_minus_C_KRW", False),
        "daily": rows,
        "errors": errors,
        "interpretation_limit": (
            "과거 당시 KIS 전체시장 실시간 TOP5 원본이 없어 후보군은 고정 유동성 종목군으로 재구성합니다. "
            "따라서 실제 당시 전체시장 체결을 100% 재현한 결과가 아니라 C와 D-v2의 상대 비교용입니다."
        ),
    }
    return result


def _run_job(cfg: LongBacktestConfig) -> None:
    start_date = _validate_date_text(cfg.start_date)
    end_date = _validate_date_text(cfg.end_date)
    if end_date < start_date:
        _state(status="error", phase="VALIDATION", last_error="end_date < start_date")
        return

    try:
        settings = Settings.from_env()
        # 과거 일별분봉은 실전 시세 API를 읽기 전용으로 사용한다. 주문 함수는 호출하지 않는다.
        client = KISClient(settings=settings, env="real")
        try:
            client._rest_min_interval = max(float(getattr(client, "_rest_min_interval", 0.0)), float(cfg.kis_min_interval_seconds))
        except Exception:
            pass
        client.get_token()

        _state(
            status="running",
            phase="DAILY_HISTORY",
            start_date=start_date,
            end_date=end_date,
            started_at=datetime.now(KST).isoformat(timespec="seconds"),
            last_error="",
            message="KIS 과거 일봉/거래일을 준비합니다.",
        )
        histories, trading_dates = _prepare_daily_histories(client, start_date, end_date, cfg)
        if not trading_dates:
            raise RuntimeError("조회 기간의 거래일을 찾지 못했습니다.")

        provider = _provider_for(client, histories, cfg)
        # 기존 C와 D-v2 계산 엔진을 그대로 쓰되 데이터 공급자만 KIS 과거분봉으로 바꾼다.
        replay_kr._download_intraday = provider
        d2_replay._download_intraday = provider

        payloads: list[dict] = []
        errors: list[dict] = []
        total = len(trading_dates)
        universe_codes = _codes()

        for idx, date_text in enumerate(trading_dates, start=1):
            if _STOP.is_set():
                _state(status="stopped", phase="STOPPED", message="사용자 요청으로 중단되었습니다.")
                return
            _wait_if_live()
            cached = _load_day_result(date_text)
            if cached is not None:
                payloads.append(cached)
                _state(
                    status="running",
                    phase="REPLAY",
                    current_date=date_text,
                    completed_days=len(payloads),
                    total_days=total,
                    progress_pct=round(100.0 * idx / total, 1),
                    message=f"장기 백테스트 {idx}/{total} · 캐시 사용",
                )
                continue

            _state(
                status="running",
                phase="MINUTE_DOWNLOAD_AND_REPLAY",
                current_date=date_text,
                completed_days=len(payloads),
                total_days=total,
                progress_pct=round(100.0 * (idx - 1) / total, 1),
                message=f"{date_text} KIS 1분봉 수집 + C/D-v2 비교 중 ({idx}/{total})",
            )
            try:
                payload = d2_replay.run_kr_open_defense_replay(
                    date_text=date_text,
                    # codes를 명시하면 기존 yfinance용 D-v2 캐시를 오염시키지 않는다.
                    codes=universe_codes,
                    refresh=True,
                )
                if not isinstance(payload, dict) or payload.get("ok") is not True:
                    raise RuntimeError(str(payload)[:600])
                # 장기 리플레이의 실제 데이터 출처로 표기를 교정한다.
                payload.setdefault("assumptions", {})
                payload["assumptions"].update({
                    "data": "KIS historical 1-minute bars",
                    "long_backtest_version": LONG_BACKTEST_VERSION,
                })
                _save_day_result(date_text, payload)
                payloads.append(payload)
            except Exception as exc:
                err = {
                    "date": date_text,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                errors.append(err)
                _state(last_error=err["error"][:1000])

            _state(
                status="running",
                phase="REPLAY",
                current_date=date_text,
                completed_days=len(payloads),
                error_days=len(errors),
                total_days=total,
                progress_pct=round(100.0 * idx / total, 1),
                message=f"장기 백테스트 {idx}/{total} 완료",
            )

        result = _aggregate(start_date, end_date, payloads, errors)
        _atomic_write_json(RESULT_FILE, result)
        _state(
            status="completed",
            phase="DONE",
            current_date=end_date,
            completed_days=len(payloads),
            error_days=len(errors),
            total_days=total,
            progress_pct=100.0,
            finished_at=datetime.now(KST).isoformat(timespec="seconds"),
            result_ready=True,
            message="D-v2 장기 백테스트 완료",
            last_error="",
        )
    except Exception as exc:
        _state(
            status="error",
            phase="ERROR",
            result_ready=RESULT_FILE.exists(),
            last_error=f"{type(exc).__name__}: {exc}"[:1200],
            message="D-v2 장기 백테스트 오류",
        )


def start_d2_long_backtest(
    start_date: str = "2026-01-01",
    end_date: str = "2026-08-09",
    restart: bool = False,
) -> dict:
    global _THREAD
    start_date = _validate_date_text(start_date)
    end_date = _validate_date_text(end_date)

    with _LOCK:
        if _THREAD and _THREAD.is_alive():
            state = _public_state()
            state["started"] = False
            state["reason"] = "already_running"
            return state

        existing = _read_json(RESULT_FILE, {}) or {}
        if (
            not restart
            and existing.get("ok") is True
            and (existing.get("period") or {}).get("start") == start_date
            and (existing.get("period") or {}).get("end") == end_date
        ):
            return {
                "ok": True,
                "started": False,
                "reason": "already_completed",
                "status": "completed",
                "result_ready": True,
                "period": {"start": start_date, "end": end_date},
            }

        if restart:
            try:
                RESULT_FILE.unlink(missing_ok=True)
            except Exception:
                pass
        _STOP.clear()
        cfg = LongBacktestConfig(start_date=start_date, end_date=end_date)
        _state(
            status="starting",
            phase="STARTING",
            start_date=start_date,
            end_date=end_date,
            config=asdict(cfg),
            result_ready=False,
            progress_pct=0.0,
            completed_days=0,
            error_days=0,
            message="D-v2 장기 백테스트 시작 준비",
            last_error="",
        )
        _THREAD = threading.Thread(
            target=_run_job,
            args=(cfg,),
            daemon=True,
            name="kr-d2-long-backtest",
        )
        _THREAD.start()
        return {
            "ok": True,
            "started": True,
            "status": "starting",
            "version": LONG_BACKTEST_VERSION,
            "period": {"start": start_date, "end": end_date},
            "message": "백그라운드에서 시작했습니다. status 주소로 진행률을 확인할 수 있습니다.",
        }


def d2_long_backtest_status() -> dict:
    return _public_state()


def _compact_day_detail(payload: dict) -> dict:
    """저장된 하루치 D-v2 결과를 분석에 필요한 항목만 읽기 전용으로 정리합니다."""
    return {
        "ok": bool(payload.get("ok", True)),
        "version": payload.get("version"),
        "date": payload.get("date"),
        "strategy": payload.get("strategy"),
        "comparison": payload.get("comparison", {}),
        "diagnostic": payload.get("diagnostic", {}),
        "baseline_summary": payload.get("baseline_summary", {}),
        "d_summary": payload.get("d_summary", {}),
        "d_by_symbol": payload.get("d_by_symbol", []),
        "d_events": payload.get("d_events", []),
        "policy": payload.get("policy", {}),
        "config": payload.get("config", {}),
        "assumptions": payload.get("assumptions", {}),
    }



def _runner_compare_exit_only(result: dict) -> dict:
    """Compare TAKE2 full-exit vs 30% runner exit policies using cached KIS 1-minute bars.

    This is a controlled exit-only A/B test: all original D-v2 trades other than the
    retained runner shares are held fixed. It does not rerun candidate selection or
    position-slot logic, so later same-symbol re-entries are reported as conflicts.
    No KIS/API calls are made; Railway Volume cache is read only.
    """
    trail_pcts = (0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 3.6, 3.8, 4.0, 4.2, 4.5, 5.0, 6.0)
    sell_slippage_pct = 0.10
    force_time = "15:15"
    base_total = int(((result.get("overall") or {}).get("D2_total_KRW", 0)) or 0)
    daily_rows = list(result.get("daily", []) or [])
    base_daily = {
        str(r.get("date", "")): int(r.get("D2_KRW", 0) or 0)
        for r in daily_rows if isinstance(r, dict) and r.get("date")
    }

    variants = {
        "TAKE2_FULL": {"label": "기존 +5% 전량매도", "delta_by_date": {}, "events": []},
        **{
            f"TRAIL_{str(pct).replace('.', '_')}": {
                "label": f"30% Runner / -{pct:.1f}% trailing",
                "trail_pct": pct,
                "delta_by_date": {},
                "events": [],
            }
            for pct in trail_pcts
        },
        "FORCE_1515": {"label": "30% Runner / 15:15 청산", "delta_by_date": {}, "events": []},
    }

    take2_records: list[dict] = []
    missing_day_results: list[str] = []
    missing_minute_cache: list[dict] = []

    # Reconstruct each same-day/same-symbol episode quantity from the recorded events.
    for row in daily_rows:
        if not isinstance(row, dict):
            continue
        date_text = str(row.get("date", "") or "").strip()
        if not date_text:
            continue
        payload = _load_day_result(date_text)
        if payload is None:
            missing_day_results.append(date_text)
            continue
        events = [e for e in list(payload.get("d_events", []) or []) if isinstance(e, dict)]
        events.sort(key=lambda e: str(e.get("시간KST", "")))
        qty_by_symbol: dict[str, int] = {}
        episode_buys_by_symbol: dict[str, int] = {}

        for idx, event in enumerate(events):
            symbol = str(event.get("종목코드", "") or "").zfill(6)
            if not symbol:
                continue
            side = str(event.get("구분", "") or "").upper()
            action = str(event.get("액션", "") or "")
            qty = max(0, int(event.get("수량", 0) or 0))
            held_before = int(qty_by_symbol.get(symbol, 0) or 0)
            episode_bought = int(episode_buys_by_symbol.get(symbol, 0) or 0)

            if side == "BUY":
                if held_before <= 0:
                    episode_bought = 0
                held_after = held_before + qty
                episode_bought += qty
                qty_by_symbol[symbol] = held_after
                episode_buys_by_symbol[symbol] = episode_bought
                continue

            if side != "SELL":
                continue

            if action == "TAKE2":
                try:
                    event_ts = pd.Timestamp(str(event.get("시간KST", "")))
                    if event_ts.tzinfo is None:
                        event_ts = event_ts.tz_localize(KST)
                    else:
                        event_ts = event_ts.tz_convert(KST)
                except Exception:
                    event_ts = pd.Timestamp(f"{date_text} 15:15:00", tz=KST)

                # 30% of the episode's originally accumulated shares, whole shares only.
                original_qty = max(episode_bought, held_before, qty)
                runner_qty = int(original_qty * 0.30)
                runner_qty = max(0, min(runner_qty, held_before if held_before > 0 else qty, qty))
                baseline_fill = float(event.get("가정체결가", 0) or 0)
                baseline_ref = float(event.get("기준가", 0) or 0)

                later_buys = []
                for later in events[idx + 1:]:
                    if str(later.get("종목코드", "") or "").zfill(6) != symbol:
                        continue
                    if str(later.get("구분", "") or "").upper() == "BUY":
                        later_buys.append(str(later.get("시간KST", "") or ""))

                take2_records.append({
                    "date": date_text,
                    "time": event_ts,
                    "symbol": symbol,
                    "name": str(event.get("종목명", "") or symbol),
                    "original_qty": int(original_qty),
                    "held_before_take2": int(held_before),
                    "take2_sell_qty": int(qty),
                    "runner_qty": int(runner_qty),
                    "baseline_ref": float(baseline_ref),
                    "baseline_fill": float(baseline_fill),
                    "later_buy_times": later_buys,
                })

            held_after = max(0, held_before - qty)
            qty_by_symbol[symbol] = held_after
            if held_after <= 0:
                episode_buys_by_symbol[symbol] = 0

    def _load_frame(date_text: str, symbol: str) -> pd.DataFrame:
        raw = _load_gzip_json(_minute_cache_path(date_text, symbol), None)
        if not isinstance(raw, dict) or not raw.get("records"):
            return pd.DataFrame()
        return _records_to_frame(
            list(raw.get("records") or []),
            float(raw.get("prev_close", 0) or 0),
        )

    def _sell_fill(ref_price: float) -> float:
        return float(ref_price) * (1.0 - sell_slippage_pct / 100.0)

    def _force_ref(frame: pd.DataFrame, date_text: str) -> tuple[float, str]:
        if frame is None or frame.empty:
            return 0.0, ""
        force_ts = pd.Timestamp(f"{date_text} {force_time}:00", tz=KST)
        safe_cutoff = force_ts - pd.Timedelta(seconds=60)
        d = frame.loc[:safe_cutoff]
        if d.empty:
            return 0.0, ""
        return float(d["Close"].iloc[-1]), pd.Timestamp(d.index[-1]).isoformat()

    for rec in take2_records:
        date_text = rec["date"]
        symbol = rec["symbol"]
        runner_qty = int(rec["runner_qty"])
        baseline_fill = float(rec["baseline_fill"])
        event_ts = rec["time"]

        # Baseline event row for audit.
        variants["TAKE2_FULL"]["events"].append({
            **{k: v for k, v in rec.items() if k != "time"},
            "time": event_ts.isoformat(),
            "exit_reason": "TAKE2_FULL",
            "exit_ref": round(float(rec["baseline_ref"]), 2),
            "exit_fill": round(baseline_fill, 2),
            "delta_KRW": 0,
            "reentry_conflict": False,
        })

        if runner_qty <= 0:
            for key, v in variants.items():
                if key == "TAKE2_FULL":
                    continue
                v["events"].append({
                    **{k: val for k, val in rec.items() if k != "time"},
                    "time": event_ts.isoformat(),
                    "exit_reason": "NO_WHOLE_SHARE_RUNNER",
                    "exit_ref": round(float(rec["baseline_ref"]), 2),
                    "exit_fill": round(baseline_fill, 2),
                    "delta_KRW": 0,
                    "reentry_conflict": False,
                })
            continue

        frame = _load_frame(date_text, symbol)
        if frame.empty:
            missing_minute_cache.append({"date": date_text, "symbol": symbol})
            continue

        # D-v2 makes decisions from the most recently completed 1-minute bar.
        # The TAKE2 event at time T therefore used bars up to T-60s. Start runner
        # evaluation with the first later completed bar, keeping the same no-lookahead rule.
        source_cutoff = event_ts - pd.Timedelta(seconds=60)
        used = frame.loc[:source_cutoff]
        if not used.empty:
            source_bar_ts = pd.Timestamp(used.index[-1])
        else:
            source_bar_ts = source_cutoff
        force_ts = pd.Timestamp(f"{date_text} {force_time}:00", tz=KST)
        force_cutoff = force_ts - pd.Timedelta(seconds=60)
        future = frame[(frame.index > source_bar_ts) & (frame.index <= force_cutoff)].copy()

        force_ref, force_bar_ts = _force_ref(frame, date_text)
        if force_ref <= 0:
            missing_minute_cache.append({"date": date_text, "symbol": symbol, "reason": "no_force_price"})
            continue

        force_fill = _sell_fill(force_ref)
        force_delta = int(round((force_fill - baseline_fill) * runner_qty))
        later_buy_times = [pd.Timestamp(x) for x in rec.get("later_buy_times", []) if x]
        reentry_conflict_force = any((t.tz_convert(KST) if t.tzinfo else t.tz_localize(KST)) < force_ts for t in later_buy_times)
        variants["FORCE_1515"]["delta_by_date"][date_text] = variants["FORCE_1515"]["delta_by_date"].get(date_text, 0) + force_delta
        variants["FORCE_1515"]["events"].append({
            **{k: v for k, v in rec.items() if k != "time"},
            "time": event_ts.isoformat(),
            "exit_reason": "FORCE_1515",
            "exit_time": force_bar_ts,
            "exit_ref": round(force_ref, 2),
            "exit_fill": round(force_fill, 2),
            "delta_KRW": int(force_delta),
            "reentry_conflict": bool(reentry_conflict_force),
        })

        for pct in trail_pcts:
            key = f"TRAIL_{str(pct).replace('.', '_')}"
            peak_close = float(rec["baseline_ref"] or 0) or float(baseline_fill)
            exit_ref = 0.0
            exit_time = ""
            exit_reason = "FORCE_1515"
            if not future.empty:
                for ts, bar in future.iterrows():
                    close = float(bar.get("Close", 0) or 0)
                    if close <= 0:
                        continue
                    peak_close = max(peak_close, close)
                    stop_level = peak_close * (1.0 - pct / 100.0)
                    if close <= stop_level:
                        exit_ref = close
                        exit_time = pd.Timestamp(ts).isoformat()
                        exit_reason = f"TRAIL_{pct:.1f}"
                        break
            if exit_ref <= 0:
                exit_ref = force_ref
                exit_time = force_bar_ts
            exit_fill = _sell_fill(exit_ref)
            delta = int(round((exit_fill - baseline_fill) * runner_qty))
            exit_ts = pd.Timestamp(exit_time) if exit_time else force_ts
            conflict = False
            for t in later_buy_times:
                try:
                    tt = t.tz_convert(KST) if t.tzinfo else t.tz_localize(KST)
                    if event_ts < tt < exit_ts:
                        conflict = True
                        break
                except Exception:
                    pass
            variants[key]["delta_by_date"][date_text] = variants[key]["delta_by_date"].get(date_text, 0) + delta
            variants[key]["events"].append({
                **{k: v for k, v in rec.items() if k != "time"},
                "time": event_ts.isoformat(),
                "exit_reason": exit_reason,
                "exit_time": exit_time,
                "peak_close": round(peak_close, 2),
                "exit_ref": round(exit_ref, 2),
                "exit_fill": round(exit_fill, 2),
                "delta_KRW": int(delta),
                "reentry_conflict": bool(conflict),
            })

    summary_rows = []
    for key, v in variants.items():
        delta_total = int(sum(int(x or 0) for x in v["delta_by_date"].values()))
        adjusted = dict(base_daily)
        for d, delta in v["delta_by_date"].items():
            adjusted[d] = int(adjusted.get(d, 0)) + int(delta)
        ordered_dates = [str(r.get("date", "")) for r in daily_rows if isinstance(r, dict) and r.get("date")]
        pnl_values = [int(adjusted.get(d, 0)) for d in ordered_dates]
        mdd, _pi, _ti = _max_drawdown(pnl_values) if pnl_values else (0, 0, 0)
        event_rows = list(v.get("events", []) or [])
        summary_rows.append({
            "id": key,
            "label": v["label"],
            "trail_pct": v.get("trail_pct"),
            "take2_events": len(take2_records),
            "effective_runner_events": sum(1 for e in event_rows if int(e.get("runner_qty", 0) or 0) > 0),
            "reentry_conflicts": sum(1 for e in event_rows if bool(e.get("reentry_conflict"))),
            "delta_vs_D2_KRW": delta_total,
            "D2_147_total_KRW": int(base_total + delta_total),
            "average_daily_KRW": round((base_total + delta_total) / len(ordered_dates), 1) if ordered_dates else 0,
            "positive_days": sum(1 for x in pnl_values if x > 0),
            "max_cumulative_drawdown_KRW": int(mdd),
        })

    candidates = [x for x in summary_rows if x["id"].startswith("TRAIL_")]
    best = max(candidates, key=lambda x: x["D2_147_total_KRW"]) if candidates else None

    # Best trailing의 22개 TAKE2별 손익기여도를 따로 정리한다.
    # 한두 종목에 수익이 과도하게 몰려 있는지 확인하기 위한 과최적화 점검용이다.
    best_event_contributions: list[dict] = []
    best_contribution_summary: dict = {}
    if best:
        best_id = str(best.get("id", ""))
        best_events = list((variants.get(best_id) or {}).get("events", []) or [])
        for e in best_events:
            best_event_contributions.append({
                "date": str(e.get("date", "") or ""),
                "time": str(e.get("time", "") or ""),
                "symbol": str(e.get("symbol", "") or ""),
                "name": str(e.get("name", "") or ""),
                "original_qty": int(e.get("original_qty", 0) or 0),
                "runner_qty": int(e.get("runner_qty", 0) or 0),
                "baseline_fill": float(e.get("baseline_fill", 0) or 0),
                "peak_close": float(e.get("peak_close", 0) or 0),
                "exit_reason": str(e.get("exit_reason", "") or ""),
                "exit_time": str(e.get("exit_time", "") or ""),
                "exit_fill": float(e.get("exit_fill", 0) or 0),
                "delta_KRW": int(e.get("delta_KRW", 0) or 0),
                "reentry_conflict": bool(e.get("reentry_conflict")),
            })
        best_event_contributions.sort(key=lambda x: x["delta_KRW"], reverse=True)
        deltas = [int(x.get("delta_KRW", 0) or 0) for x in best_event_contributions]
        positive = [x for x in deltas if x > 0]
        negative = [x for x in deltas if x < 0]
        total = int(sum(deltas))
        top1 = int(sum(sorted(positive, reverse=True)[:1])) if positive else 0
        top3 = int(sum(sorted(positive, reverse=True)[:3])) if positive else 0
        best_contribution_summary = {
            "best_id": best_id,
            "best_label": best.get("label"),
            "event_count": len(best_event_contributions),
            "positive_events": len(positive),
            "negative_events": len(negative),
            "zero_events": sum(1 for x in deltas if x == 0),
            "total_delta_KRW": total,
            "top1_positive_delta_KRW": top1,
            "top3_positive_delta_KRW": top3,
            "top1_share_of_total_pct": round(100.0 * top1 / total, 2) if total > 0 else 0,
            "top3_share_of_total_pct": round(100.0 * top3 / total, 2) if total > 0 else 0,
            "largest_gain_KRW": max(deltas) if deltas else 0,
            "largest_loss_KRW": min(deltas) if deltas else 0,
            "median_delta_KRW": int(pd.Series(deltas).median()) if deltas else 0,
            "reentry_conflicts": sum(1 for x in best_event_contributions if x.get("reentry_conflict")),
        }

    return {
        "ok": not missing_day_results and not missing_minute_cache,
        "mode": "EXIT_ONLY_CONTROLLED_AB_TEST",
        "read_only": True,
        "take2_event_count": len(take2_records),
        "runner_fraction_of_original_position": 0.30,
        "whole_share_rule": "floor(original_episode_qty * 0.30); 0 means no runner",
        "price_rule": "1-minute closed-bar Close only; no future data; trailing high-watermark is peak Close",
        "sell_slippage_pct": sell_slippage_pct,
        "force_exit_time": force_time,
        "baseline_D2_147_total_KRW": base_total,
        "variants": summary_rows,
        "best_trailing": best,
        "best_trailing_contribution_summary": best_contribution_summary,
        "best_trailing_event_contributions": best_event_contributions,
        "missing_day_results": missing_day_results,
        "missing_minute_cache": missing_minute_cache,
        "take2_events": [
            {**{k: v for k, v in r.items() if k != "time"}, "time": r["time"].isoformat()}
            for r in take2_records
        ],
        "variant_event_details": {k: v.get("events", []) for k, v in variants.items()},
        "important_limit": (
            "This holds all original D-v2 trades fixed and changes only the runner shares after the 22 TAKE2 events. "
            "If a hypothetical runner overlaps a later same-symbol BUY, reentry_conflicts is incremented; a full engine rerun is required before production use."
        ),
    }


def d2_long_backtest_result(detail: bool = False) -> dict:
    result = _read_json(RESULT_FILE, {}) or {}
    if not result:
        return {
            "ok": False,
            "version": LONG_BACKTEST_VERSION,
            "error": "result not ready",
            "status": _public_state(),
        }

    if detail:
        # 새 백테스트를 돌리지 않고 Railway Volume에 이미 저장된
        # 147일 daily_results/*.json.gz를 읽기만 한다.
        # 기존 Worker의 /backtest-kr-d2-result?detail=true 주소를 그대로 사용하므로
        # worker.py와 주문 로직을 수정할 필요가 없다.
        detailed = dict(result)
        day_details: dict[str, dict] = {}
        missing_dates: list[str] = []

        for row in list(result.get("daily", []) or []):
            if not isinstance(row, dict):
                continue
            date_text = str(row.get("date", "") or "").strip()
            if not date_text:
                continue
            payload = _load_day_result(date_text)
            if payload is None:
                missing_dates.append(date_text)
                continue
            day_details[date_text] = _compact_day_detail(payload)

        detailed["day_details"] = day_details
        detailed["day_detail_count"] = len(day_details)
        detailed["day_detail_missing_dates"] = missing_dates
        detailed["day_detail_read_only"] = True
        detailed["day_detail_source"] = "Railway Volume / replays/kr_d2_long/daily_results"
        detailed["day_detail_note"] = (
            "기존 장기 백테스트가 Railway Volume에 저장한 일별 캐시를 읽기만 합니다. "
            "KIS 조회, 리플레이 재실행, 주문 실행은 하지 않습니다."
        )
        detailed["runner_comparison"] = _runner_compare_exit_only(result)
        return detailed

    # 휴대폰에서 보기 편하도록 일별 150여 줄은 기본 응답에서 제외한다.
    compact = dict(result)
    compact.pop("daily", None)
    runner = _runner_compare_exit_only(result)
    runner.pop("variant_event_details", None)
    runner.pop("take2_events", None)
    compact["runner_comparison"] = runner
    return compact


def stop_d2_long_backtest() -> dict:
    _STOP.set()
    return {"ok": True, "requested": True, "message": "현재 일자 처리 후 중단합니다."}

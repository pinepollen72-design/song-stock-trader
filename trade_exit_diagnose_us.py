from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from strategy_us import (
    BENCHMARK_SYMBOL,
    _benchmark_metrics,
    _extract_symbol_frame,
    _score_frame,
    _split_session_for_date,
)
from trade_replay_us import (
    ET,
    ReplayTradeConfig,
    _download_intraday,
    _normalize_symbols,
    _rank_snapshot,
    run_trade_replay,
)


def _resolve_state_dir() -> Path:
    explicit = os.getenv("SONG_TRADER_STATE_DIR", "").strip()
    if explicit:
        return Path(explicit)

    railway_volume = os.getenv("RAILWAY_VOLUME_MOUNT_PATH", "").strip()
    if railway_volume:
        return Path(railway_volume) / "song_trader_v2"

    return Path("/tmp/song_trader_v2")


STATE_DIR = _resolve_state_dir()
DIAG_DIR = STATE_DIR / "replays" / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "AMD", "GOOGL",
    "AVGO", "NFLX", "PLTR", "MU", "INTC", "SMCI", "ARM", "COIN",
    "HOOD", "SOFI", "MSTR", "RBLX", "UBER", "CRWD", "PANW", "QCOM",
    "AMAT", "TSM", "MRVL", "LLY", "JPM", "BAC",
]


def _best_current_config() -> ReplayTradeConfig:
    return ReplayTradeConfig(
        entry_initial_pct=50.0,
        entry_confirm_pct=0.0,
        buy1_market_filter_mode="NONE",
        buy1_confirmation_scans=1,
        buy1_confirmation_max_rank=5,
        buy1_require_relative_strength_hold=False,

        # BUY2 = 현재 유력안 B_STRICT
        buy2_enabled=True,
        us_add2_trigger_pct=0.80,
        buy2_min_hold_minutes=5.0,
        buy2_max_rank=3,
        buy2_min_score=70.0,
        buy2_require_recent5_positive=True,
        buy2_require_recent10_positive=True,
        buy2_require_relative_strength_positive=True,
    )


def _ensure_et_index(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index)
    if d.index.tz is None:
        d.index = d.index.tz_localize("UTC").tz_convert(ET)
    else:
        d.index = d.index.tz_convert(ET)
    return d.sort_index()


def _close_at_or_before(frame: pd.DataFrame, cutoff: pd.Timestamp) -> float | None:
    d = _ensure_et_index(frame)
    cols = {str(c).lower(): c for c in d.columns}
    close_col = cols.get("close")
    if close_col is None:
        return None

    before = d[d.index <= cutoff]
    if before.empty:
        return None

    values = pd.to_numeric(before[close_col], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.iloc[-1])


def _build_episodes(events: list[dict]) -> list[dict]:
    open_by_symbol: dict[str, dict] = {}
    episodes: list[dict] = []

    for e in sorted(events, key=lambda x: str(x.get("시간ET", ""))):
        symbol = str(e.get("종목코드", "")).upper()
        side = str(e.get("구분", "")).upper()
        qty = int(e.get("수량", 0) or 0)
        fill = float(e.get("가정체결가", 0) or 0)

        if not symbol or qty <= 0 or fill <= 0:
            continue

        if side == "BUY":
            ep = open_by_symbol.get(symbol)
            if ep is None:
                ep = {
                    "종목코드": symbol,
                    "first_buy_time": str(e.get("시간ET", "")),
                    "first_buy_price": fill,
                    "buy_qty": 0,
                    "buy_cost": 0.0,
                    "remaining_qty": 0,
                    "buy_events": [],
                    "sell_events": [],
                    "realized_pnl": 0.0,
                }
                open_by_symbol[symbol] = ep

            ep["buy_qty"] += qty
            ep["buy_cost"] += qty * fill
            ep["remaining_qty"] += qty
            ep["buy_events"].append(e)

        elif side == "SELL":
            ep = open_by_symbol.get(symbol)
            if ep is None:
                continue

            ep["remaining_qty"] -= qty
            ep["sell_events"].append(e)
            ep["realized_pnl"] += float(e.get("실현손익USD", 0) or 0)

            if ep["remaining_qty"] <= 0:
                ep["exit_time"] = str(e.get("시간ET", ""))
                ep["exit_action"] = str(e.get("액션", ""))
                ep["exit_price"] = fill
                episodes.append(ep)
                open_by_symbol.pop(symbol, None)

    return episodes


def _snapshot_all(
    *,
    cutoff: pd.Timestamp,
    symbol_sessions: dict[str, dict],
    qqq_session: pd.DataFrame,
    qqq_prev_close: float,
    qqq_prev_ok: bool,
) -> pd.DataFrame:
    qqq = _ensure_et_index(qqq_session)
    qqq_slice = qqq[qqq.index <= cutoff].copy()

    benchmark = _benchmark_metrics(
        qqq_slice,
        prev_close=qqq_prev_close,
        prev_close_available=qqq_prev_ok,
    )

    rows = []
    for symbol, meta in symbol_sessions.items():
        d = _ensure_et_index(meta["session"])
        sliced = d[d.index <= cutoff].copy()

        scored = _score_frame(
            symbol,
            sliced,
            benchmark=benchmark,
            prev_close=float(meta["prev_close"]),
            prev_close_available=bool(meta["prev_ok"]),
        )
        if scored:
            rows.append(scored)

    ranked = _rank_snapshot(rows)
    return ranked


def _warning_state(row: dict | None, pnl_pct: float | None) -> dict:
    if row is None or pnl_pct is None:
        return {
            "약화점수": None,
            "조기청산경고": False,
            "강한조기청산경고": False,
            "약화이유": "",
        }

    signal = str(row.get("판정", ""))
    score = float(row.get("종합점수", 0) or 0)
    rank = int(row.get("순위", 999) or 999)
    vwap_gap = float(row.get("VWAP괴리율", 0) or 0)
    ret5 = float(row.get("최근5분수익률", 0) or 0)
    ret10 = float(row.get("최근10분수익률", 0) or 0)
    rel = float(row.get("상대강도", 0) or 0)

    reasons = []
    points = 0

    if pnl_pct < 0:
        points += 1
        reasons.append("손실중")
    if "매수 후보" not in signal:
        points += 1
        reasons.append("매수후보탈락")
    if rank > 5:
        points += 1
        reasons.append("TOP5탈락")
    if score < 62.0:
        points += 1
        reasons.append("점수62미만")
    if vwap_gap < 0:
        points += 1
        reasons.append("VWAP아래")
    if ret5 < 0:
        points += 1
        reasons.append("5분음수")
    if ret10 < 0:
        points += 1
        reasons.append("10분음수")
    if rel < 0:
        points += 1
        reasons.append("상대강도음수")

    warning = bool(pnl_pct <= 0.0 and points >= 4)
    strong = bool(pnl_pct <= -0.50 and points >= 5)

    return {
        "약화점수": int(points),
        "조기청산경고": warning,
        "강한조기청산경고": strong,
        "약화이유": " / ".join(reasons),
    }


def diagnose_exit_state_day(
    date_text: str = "2026-08-14",
    symbols: Iterable[str] | None = None,
) -> dict:
    symbols = _normalize_symbols(symbols or DEFAULT_UNIVERSE)
    config = _best_current_config()

    replay = run_trade_replay(
        date_text=date_text,
        symbols=symbols,
        config=config,
    )
    events = list(replay.get("events", []) or [])
    episodes = _build_episodes(events)

    if not episodes:
        return {
            "ok": True,
            "version": "exit-state-diagnose-v1",
            "date": date_text,
            "summary": {"거래에피소드수": 0},
            "episodes": [],
        }

    batch = _download_intraday(date_text, symbols)
    universe_size = len(symbols) + (0 if BENCHMARK_SYMBOL in symbols else 1)

    qqq_full = _extract_symbol_frame(batch, BENCHMARK_SYMBOL, universe_size)
    if qqq_full is None or qqq_full.empty:
        raise RuntimeError("QQQ 데이터를 찾지 못했습니다.")

    qqq_session, qqq_prev_close, qqq_prev_ok = _split_session_for_date(
        qqq_full,
        date_text,
    )
    if qqq_session is None or qqq_session.empty:
        raise RuntimeError("QQQ 목표일 정규장 데이터가 없습니다.")

    symbol_sessions: dict[str, dict] = {}
    for symbol in symbols:
        full = _extract_symbol_frame(batch, symbol, universe_size)
        if full is None or full.empty:
            continue

        session, prev_close, prev_ok = _split_session_for_date(
            full,
            date_text,
        )
        if session is None or session.empty:
            continue

        symbol_sessions[symbol] = {
            "session": session,
            "prev_close": float(prev_close or 0.0),
            "prev_ok": bool(prev_ok),
        }

    snapshot_cache: dict[str, pd.DataFrame] = {}

    def get_ranked(cutoff: pd.Timestamp) -> pd.DataFrame:
        key = cutoff.isoformat()
        if key not in snapshot_cache:
            snapshot_cache[key] = _snapshot_all(
                cutoff=cutoff,
                symbol_sessions=symbol_sessions,
                qqq_session=qqq_session,
                qqq_prev_close=qqq_prev_close,
                qqq_prev_ok=qqq_prev_ok,
            )
        return snapshot_cache[key]

    rows = []

    for idx, ep in enumerate(episodes, start=1):
        symbol = ep["종목코드"]
        meta = symbol_sessions.get(symbol)
        if not meta:
            continue

        entry_ts = pd.Timestamp(ep["first_buy_time"])
        exit_ts = pd.Timestamp(ep["exit_time"])
        avg_entry = (
            float(ep["buy_cost"]) / int(ep["buy_qty"])
            if int(ep["buy_qty"]) > 0 else 0.0
        )
        actual_return = (
            float(ep["realized_pnl"]) / float(ep["buy_cost"]) * 100.0
            if float(ep["buy_cost"]) > 0 else 0.0
        )

        base = {
            "에피소드": idx,
            "종목코드": symbol,
            "진입시각ET": ep["first_buy_time"],
            "청산시각ET": ep["exit_time"],
            "실제청산사유": ep["exit_action"],
            "실현손익USD": round(float(ep["realized_pnl"]), 2),
            "실현수익률_pct": round(actual_return, 3),
            "매수횟수": len(ep["buy_events"]),
            "평균진입가": round(avg_entry, 4),
        }

        for minutes in (15, 30, 60):
            cutoff = entry_ts + pd.Timedelta(minutes=minutes)
            prefix = f"{minutes}분"

            if cutoff >= exit_ts:
                base[f"{prefix}_상태"] = "이미청산"
                base[f"{prefix}_손익_pct"] = None
                base[f"{prefix}_TOP5순위"] = None
                base[f"{prefix}_판정"] = ""
                base[f"{prefix}_점수"] = None
                base[f"{prefix}_상대강도"] = None
                base[f"{prefix}_VWAP괴리"] = None
                base[f"{prefix}_5분"] = None
                base[f"{prefix}_10분"] = None
                base[f"{prefix}_약화점수"] = None
                base[f"{prefix}_조기청산경고"] = False
                base[f"{prefix}_강한경고"] = False
                base[f"{prefix}_약화이유"] = ""
                continue

            price = _close_at_or_before(meta["session"], cutoff)
            if price is None or avg_entry <= 0:
                base[f"{prefix}_상태"] = "가격없음"
                continue

            pnl = (price / avg_entry - 1.0) * 100.0

            ranked = get_ranked(cutoff)
            target_row = None
            if ranked is not None and not ranked.empty:
                hit = ranked[
                    ranked["종목코드"].astype(str).str.upper() == symbol
                ]
                if not hit.empty:
                    target_row = hit.iloc[0].to_dict()

            warning = _warning_state(target_row, pnl)

            base[f"{prefix}_상태"] = "보유중"
            base[f"{prefix}_손익_pct"] = round(pnl, 3)
            base[f"{prefix}_TOP5순위"] = (
                None if target_row is None
                else int(target_row.get("순위", 999) or 999)
            )
            base[f"{prefix}_판정"] = (
                "" if target_row is None
                else str(target_row.get("판정", ""))
            )
            base[f"{prefix}_점수"] = (
                None if target_row is None
                else round(float(target_row.get("종합점수", 0) or 0), 1)
            )
            base[f"{prefix}_상대강도"] = (
                None if target_row is None
                else round(float(target_row.get("상대강도", 0) or 0), 3)
            )
            base[f"{prefix}_VWAP괴리"] = (
                None if target_row is None
                else round(float(target_row.get("VWAP괴리율", 0) or 0), 3)
            )
            base[f"{prefix}_5분"] = (
                None if target_row is None
                else round(float(target_row.get("최근5분수익률", 0) or 0), 3)
            )
            base[f"{prefix}_10분"] = (
                None if target_row is None
                else round(float(target_row.get("최근10분수익률", 0) or 0), 3)
            )
            base[f"{prefix}_약화점수"] = warning["약화점수"]
            base[f"{prefix}_조기청산경고"] = warning["조기청산경고"]
            base[f"{prefix}_강한경고"] = warning["강한조기청산경고"]
            base[f"{prefix}_약화이유"] = warning["약화이유"]

        rows.append(base)

    df = pd.DataFrame(rows)
    if df.empty:
        return {
            "ok": True,
            "version": "exit-state-diagnose-v1",
            "date": date_text,
            "summary": {"거래에피소드수": 0},
            "episodes": [],
        }

    summary = {
        "거래에피소드수": int(len(df)),
        "수익거래수": int((df["실현손익USD"] > 0).sum()),
        "손실거래수": int((df["실현손익USD"] < 0).sum()),
        "총실현손익USD": round(float(df["실현손익USD"].sum()), 2),
    }

    for minutes in (15, 30, 60):
        prefix = f"{minutes}분"
        active = df[df[f"{prefix}_상태"] == "보유중"].copy()
        losses = active[active["실현손익USD"] < 0]
        wins = active[active["실현손익USD"] > 0]

        summary[f"{prefix}_보유중거래수"] = int(len(active))
        summary[f"{prefix}_손실거래경고수"] = int(
            losses[f"{prefix}_조기청산경고"].fillna(False).astype(bool).sum()
        ) if not losses.empty else 0
        summary[f"{prefix}_수익거래오탐수"] = int(
            wins[f"{prefix}_조기청산경고"].fillna(False).astype(bool).sum()
        ) if not wins.empty else 0
        summary[f"{prefix}_손실거래강한경고수"] = int(
            losses[f"{prefix}_강한경고"].fillna(False).astype(bool).sum()
        ) if not losses.empty else 0
        summary[f"{prefix}_수익거래강한오탐수"] = int(
            wins[f"{prefix}_강한경고"].fillna(False).astype(bool).sum()
        ) if not wins.empty else 0

    stamp = date_text.replace("-", "")
    csv_path = DIAG_DIR / f"us_exit_state_diagnose_{stamp}.csv"
    json_path = DIAG_DIR / f"us_exit_state_diagnose_{stamp}.json"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    payload = {
        "ok": True,
        "version": "exit-state-diagnose-v1",
        "date": date_text,
        "config": {
            "entry": "BUY1 즉시 50%",
            "buy2": "B_STRICT",
            "market_filter": "NONE",
            "checkpoints_minutes": [15, 30, 60],
            "warning_rule": (
                "손실중이면서 매수후보탈락/TOP5탈락/점수62미만/"
                "VWAP아래/5분음수/10분음수/상대강도음수 중 약화점수 합계가 "
                "4점 이상이면 경고, 손익 -0.5% 이하 + 5점 이상이면 강한경고"
            ),
        },
        "summary": summary,
        "episodes": df.to_dict("records"),
        "files": {
            "detail_csv": str(csv_path),
            "json": str(json_path),
        },
        "note": (
            "이 단계는 조기청산 규칙을 실제 적용하는 것이 아니라 "
            "15/30/60분 시점에서 손실거래와 수익거래의 상태 차이를 진단하는 용도입니다."
        ),
    }

    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return payload


if __name__ == "__main__":
    result = diagnose_exit_state_day(
        date_text=os.getenv("DIAG_DATE", "2026-08-14"),
        symbols=[
            x.strip()
            for x in os.getenv(
                "DIAG_SYMBOLS",
                ",".join(DEFAULT_UNIVERSE),
            ).split(",")
            if x.strip()
        ],
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from strategy_us import BENCHMARK_SYMBOL, _extract_symbol_frame, _split_session_for_date
from trade_replay_us import (
    ET,
    ReplayTradeConfig,
    _download_intraday,
    _normalize_symbols,
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
    """지금까지 남은 기준 조합으로 진단한다."""
    return ReplayTradeConfig(
        # BUY1 즉시 50%
        entry_initial_pct=50.0,
        entry_confirm_pct=0.0,

        # BUY1 시장필터/연속대기 없음
        buy1_market_filter_mode="NONE",
        buy1_confirmation_scans=1,
        buy1_confirmation_max_rank=5,
        buy1_require_relative_strength_hold=False,

        # BUY2 = B_STRICT
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


def _ohlc_cols(frame: pd.DataFrame) -> dict[str, object]:
    cols = {str(c).lower(): c for c in frame.columns}
    for name in ("open", "high", "low", "close"):
        if name not in cols:
            raise RuntimeError(f"OHLC 컬럼 누락: {name}")
    return cols


def _first_close_at_or_after(
    session: pd.DataFrame,
    target: pd.Timestamp,
) -> float | None:
    d = _ensure_et_index(session)
    cols = _ohlc_cols(d)
    later = d[d.index >= target]
    if later.empty:
        return None
    values = pd.to_numeric(later[cols["close"]], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.iloc[0])


def _window_metrics(
    session: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    basis_price: float,
) -> dict:
    d = _ensure_et_index(session)
    cols = _ohlc_cols(d)
    w = d[(d.index >= start_ts) & (d.index <= end_ts)].copy()

    if w.empty or basis_price <= 0:
        return {
            "mfe_pct": 0.0,
            "mae_pct": 0.0,
            "mfe_time": "",
            "mae_time": "",
            "minutes_to_mfe": 0.0,
            "minutes_to_mae": 0.0,
        }

    highs = pd.to_numeric(w[cols["high"]], errors="coerce").dropna()
    lows = pd.to_numeric(w[cols["low"]], errors="coerce").dropna()
    if highs.empty or lows.empty:
        return {
            "mfe_pct": 0.0,
            "mae_pct": 0.0,
            "mfe_time": "",
            "mae_time": "",
            "minutes_to_mfe": 0.0,
            "minutes_to_mae": 0.0,
        }

    max_high = float(highs.max())
    min_low = float(lows.min())
    mfe_ts = highs.idxmax()
    mae_ts = lows.idxmin()

    return {
        "mfe_pct": round((max_high / basis_price - 1.0) * 100.0, 3),
        "mae_pct": round((min_low / basis_price - 1.0) * 100.0, 3),
        "mfe_time": mfe_ts.isoformat(),
        "mae_time": mae_ts.isoformat(),
        "minutes_to_mfe": round((mfe_ts - start_ts).total_seconds() / 60.0, 1),
        "minutes_to_mae": round((mae_ts - start_ts).total_seconds() / 60.0, 1),
    }


def _build_episodes(events: list[dict]) -> list[dict]:
    """BUY1부터 완전 청산까지를 한 에피소드로 묶는다."""
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


def diagnose_trade_day(
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
            "version": "trade-diagnose-v1",
            "date": date_text,
            "summary": {"거래에피소드수": 0},
            "episodes": [],
        }

    batch = _download_intraday(date_text, symbols)
    universe_size = len(symbols) + (0 if BENCHMARK_SYMBOL in symbols else 1)

    sessions: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        full = _extract_symbol_frame(batch, symbol, universe_size)
        if full is None or full.empty:
            continue
        session, _, _ = _split_session_for_date(full, date_text)
        if session is not None and not session.empty:
            sessions[symbol] = session

    rows: list[dict] = []

    for ep in episodes:
        symbol = ep["종목코드"]
        session = sessions.get(symbol)
        if session is None:
            continue

        first_ts = pd.Timestamp(ep["first_buy_time"])
        last_buy_ts = pd.Timestamp(ep["buy_events"][-1]["시간ET"])
        exit_ts = pd.Timestamp(ep["exit_time"])

        avg_entry = (
            float(ep["buy_cost"]) / int(ep["buy_qty"])
            if int(ep["buy_qty"]) > 0
            else 0.0
        )
        first_entry = float(ep["first_buy_price"])
        exit_price = float(ep["exit_price"])

        # 첫 진입 기준 움직임 / 모든 분할매수 완료 후 평균단가 기준 움직임을 둘 다 본다.
        first_metrics = _window_metrics(session, first_ts, exit_ts, first_entry)
        avg_metrics = _window_metrics(session, last_buy_ts, exit_ts, avg_entry)

        exit_ret_avg = (
            (exit_price / avg_entry - 1.0) * 100.0
            if avg_entry > 0
            else 0.0
        )

        p30 = _first_close_at_or_after(session, exit_ts + pd.Timedelta(minutes=30))
        p60 = _first_close_at_or_after(session, exit_ts + pd.Timedelta(minutes=60))

        after30 = (
            (p30 / exit_price - 1.0) * 100.0
            if p30 is not None and exit_price > 0
            else None
        )
        after60 = (
            (p60 / exit_price - 1.0) * 100.0
            if p60 is not None and exit_price > 0
            else None
        )

        first_buy = ep["buy_events"][0]
        sell_reason_text = " | ".join(
            str(x.get("이유", "")) for x in ep["sell_events"]
        )

        tags: list[str] = []
        realized = float(ep["realized_pnl"])

        if realized < 0 and first_metrics["mfe_pct"] < 0.8:
            tags.append("진입후상승부족")
        if realized < 0 and first_metrics["mfe_pct"] >= 1.2:
            tags.append("수익구간후손실전환")
        if avg_metrics["mae_pct"] <= -2.0 and avg_metrics["mfe_pct"] <= 0.5:
            tags.append("분할매수후추세악화")
        if ep["exit_action"] == "STOP_LOSS":
            if after30 is not None and after30 >= 1.5:
                tags.append("손절후30분강한회복")
            if after60 is not None and after60 >= 2.0:
                tags.append("손절후60분강한회복")
        if "PROFIT_GUARD" in ep["exit_action"]:
            if after60 is not None and after60 >= 2.0:
                tags.append("수익보호후추가상승큼")
        if "FORCE_SELL" in ep["exit_action"] and realized < 0:
            tags.append("장마감까지회복실패")
        if not tags:
            tags.append("복합/추가검토")

        rows.append({
            "종목코드": symbol,
            "진입시각ET": ep["first_buy_time"],
            "마지막매수시각ET": ep["buy_events"][-1]["시간ET"],
            "청산시각ET": ep["exit_time"],
            "보유분": round((exit_ts - first_ts).total_seconds() / 60.0, 1),
            "매수횟수": len(ep["buy_events"]),
            "매도횟수": len(ep["sell_events"]),
            "총매수수량": int(ep["buy_qty"]),
            "첫진입가": round(first_entry, 4),
            "평균진입가": round(avg_entry, 4),
            "최종청산가": round(exit_price, 4),
            "최종청산사유": ep["exit_action"],
            "실현손익USD": round(realized, 2),
            "청산시점평균단가대비_pct": round(exit_ret_avg, 3),
            "진입점수": first_buy.get("종합점수", ""),
            "진입TOP5순위": first_buy.get("TOP5순위", ""),
            "진입이유": first_buy.get("이유", ""),
            "매도이유상세": sell_reason_text,
            "첫진입기준_MFE_pct": first_metrics["mfe_pct"],
            "첫진입기준_MAE_pct": first_metrics["mae_pct"],
            "첫진입후_MFE까지분": first_metrics["minutes_to_mfe"],
            "첫진입후_MAE까지분": first_metrics["minutes_to_mae"],
            "분할완료후_MFE_pct": avg_metrics["mfe_pct"],
            "분할완료후_MAE_pct": avg_metrics["mae_pct"],
            "분할완료후_MFE까지분": avg_metrics["minutes_to_mfe"],
            "분할완료후_MAE까지분": avg_metrics["minutes_to_mae"],
            "매도30분후등락_pct": None if after30 is None else round(after30, 3),
            "매도60분후등락_pct": None if after60 is None else round(after60, 3),
            "진단": " / ".join(tags),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return {
            "ok": True,
            "version": "trade-diagnose-v1",
            "date": date_text,
            "summary": {"거래에피소드수": 0},
            "episodes": [],
        }

    loss_df = df[df["실현손익USD"] < 0].copy()
    win_df = df[df["실현손익USD"] > 0].copy()

    summary = {
        "거래에피소드수": int(len(df)),
        "수익거래수": int(len(win_df)),
        "손실거래수": int(len(loss_df)),
        "승률_pct": round(len(win_df) / len(df) * 100.0, 1) if len(df) else 0.0,
        "총실현손익USD": round(float(df["실현손익USD"].sum()), 2),
        "전체평균_첫진입MFE_pct": round(float(df["첫진입기준_MFE_pct"].mean()), 3),
        "전체평균_첫진입MAE_pct": round(float(df["첫진입기준_MAE_pct"].mean()), 3),
        "손실평균_첫진입MFE_pct": round(float(loss_df["첫진입기준_MFE_pct"].mean()), 3) if not loss_df.empty else 0.0,
        "손실평균_첫진입MAE_pct": round(float(loss_df["첫진입기준_MAE_pct"].mean()), 3) if not loss_df.empty else 0.0,
        "수익평균_첫진입MFE_pct": round(float(win_df["첫진입기준_MFE_pct"].mean()), 3) if not win_df.empty else 0.0,
        "수익평균_첫진입MAE_pct": round(float(win_df["첫진입기준_MAE_pct"].mean()), 3) if not win_df.empty else 0.0,
        "손실중_MFE0_8미만": int((loss_df["첫진입기준_MFE_pct"] < 0.8).sum()) if not loss_df.empty else 0,
        "손실중_MFE1_2이상": int((loss_df["첫진입기준_MFE_pct"] >= 1.2).sum()) if not loss_df.empty else 0,
        "손절종료수": int((df["최종청산사유"] == "STOP_LOSS").sum()),
        "수익보호종료수": int(df["최종청산사유"].astype(str).str.contains("PROFIT_GUARD", na=False).sum()),
        "강제청산종료수": int(df["최종청산사유"].astype(str).str.contains("FORCE_SELL", na=False).sum()),
    }

    stamp = date_text.replace("-", "")
    csv_path = DIAG_DIR / f"us_trade_diagnose_{stamp}.csv"
    json_path = DIAG_DIR / f"us_trade_diagnose_{stamp}.json"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    payload = {
        "ok": True,
        "version": "trade-diagnose-v1",
        "date": date_text,
        "config": {
            "entry": "BUY1 즉시 50%",
            "buy2": "B_STRICT: 5분 +0.8% TOP3 점수70 5/10분 양수 상대강도 양수",
            "market_filter": "NONE",
            "stop_loss_pct": float(config.stop_loss_pct),
            "take1_pct": float(config.take1_pct),
            "take2_pct": float(config.take2_pct),
            "profit_guard_trigger_pct": float(config.us_profit_guard_trigger_pct),
            "profit_guard_drawdown_pct": float(config.us_profit_guard_drawdown_pct),
        },
        "summary": summary,
        "episodes": df.to_dict("records"),
        "files": {
            "detail_csv": str(csv_path),
            "json": str(json_path),
        },
        "note": "MFE/MAE는 yfinance 1분봉 고가/저가 기준이며 실제 KIS 체결·호가와 차이가 있을 수 있음.",
    }

    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return payload


if __name__ == "__main__":
    result = diagnose_trade_day(
        date_text=os.getenv("DIAG_DATE", "2026-08-14"),
        symbols=[
            x.strip()
            for x in os.getenv("DIAG_SYMBOLS", ",".join(DEFAULT_UNIVERSE)).split(",")
            if x.strip()
        ],
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from strategy_us import (
    BENCHMARK_SYMBOL,
    _benchmark_metrics,
    _extract_symbol_frame,
    _score_frame,
    _split_session_for_date,
)

ET = ZoneInfo("America/New_York")


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
REPLAY_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SYMBOLS = ["TSLA", "NVDA", "AMAT"]


def _normalize_symbols(symbols: Iterable[str] | None) -> list[str]:
    out = []
    seen = set()

    for raw in symbols or DEFAULT_SYMBOLS:
        s = str(raw).strip().upper()
        if s and s not in seen:
            seen.add(s)
            out.append(s)

    return out


def _download_intraday(date_text: str, symbols: list[str]) -> pd.DataFrame:
    """
    목표 날짜 + 그 이전 거래일을 함께 받아 전일 종가를 복원한다.
    """
    try:
        import yfinance as yf
    except Exception as e:
        raise RuntimeError(f"yfinance import 실패: {e}") from e

    day = pd.Timestamp(date_text)
    start_day = day - pd.Timedelta(days=5)
    next_day = day + pd.Timedelta(days=1)

    download_symbols = list(symbols)
    if BENCHMARK_SYMBOL not in download_symbols:
        download_symbols.append(BENCHMARK_SYMBOL)

    try:
        batch = yf.download(
            tickers=download_symbols,
            start=start_day.strftime("%Y-%m-%d"),
            end=next_day.strftime("%Y-%m-%d"),
            interval="1m",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
            prepost=False,
            timeout=20,
        )
    except TypeError:
        batch = yf.download(
            tickers=download_symbols,
            start=start_day.strftime("%Y-%m-%d"),
            end=next_day.strftime("%Y-%m-%d"),
            interval="1m",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
            prepost=False,
        )
    except Exception as e:
        raise RuntimeError(
            f"{date_text} 미국 1분봉 다운로드 실패: {type(e).__name__}: {e}"
        ) from e

    if batch is None or batch.empty:
        raise RuntimeError(f"{date_text} 미국 1분봉 결과가 비어 있음")

    return batch


def _ensure_et_index(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()

    if not isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index)

    if d.index.tz is None:
        d.index = d.index.tz_localize("UTC").tz_convert(ET)
    else:
        d.index = d.index.tz_convert(ET)

    return d.sort_index()


def _slice_until(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    d = _ensure_et_index(frame)
    return d[d.index <= cutoff].copy()


def run_replay(
    date_text: str = "2026-08-14",
    symbols: Iterable[str] | None = None,
    step_minutes: int = 5,
) -> dict:
    """
    지정 날짜 미국장을 V3.1 기준으로 5분 간격 재평가한다.

    핵심:
    - 전일 정규장 마지막 종가를 복원
    - 목표 날짜 정규장 데이터만 점수화
    - TSLA/NVDA/AMAT를 동일 기준으로 비교
    """
    symbols = _normalize_symbols(symbols)
    step_minutes = max(1, int(step_minutes))

    batch = _download_intraday(date_text, symbols)
    universe_size = len(symbols) + (0 if BENCHMARK_SYMBOL in symbols else 1)

    benchmark_full = _extract_symbol_frame(
        batch,
        BENCHMARK_SYMBOL,
        universe_size,
    )
    if benchmark_full is None or benchmark_full.empty:
        raise RuntimeError("QQQ 1분봉을 찾지 못했습니다.")

    qqq_session, qqq_prev_close, qqq_prev_ok = _split_session_for_date(
        benchmark_full,
        date_text,
    )
    if qqq_session is None or qqq_session.empty:
        raise RuntimeError("QQQ 목표일 정규장 데이터를 찾지 못했습니다.")

    symbol_sessions = {}

    for symbol in symbols:
        full = _extract_symbol_frame(
            batch,
            symbol,
            universe_size,
        )
        if full is None or full.empty:
            continue

        session, prev_close, prev_ok = _split_session_for_date(
            full,
            date_text,
        )
        if session is not None and not session.empty:
            symbol_sessions[symbol] = {
                "session": session,
                "prev_close": prev_close,
                "prev_ok": prev_ok,
            }

    if not symbol_sessions:
        raise RuntimeError("리플레이 가능한 종목 데이터가 없습니다.")

    date_ts = pd.Timestamp(date_text, tz=ET)
    start = date_ts.replace(hour=9, minute=35, second=0)
    end = date_ts.replace(hour=15, minute=55, second=0)

    rows = []
    cutoff = start

    while cutoff <= end:
        qqq_slice = _slice_until(qqq_session, cutoff)
        benchmark = _benchmark_metrics(
            qqq_slice,
            prev_close=qqq_prev_close,
            prev_close_available=qqq_prev_ok,
        )

        ranked_rows = []

        for symbol, meta in symbol_sessions.items():
            sliced = _slice_until(
                meta["session"],
                cutoff,
            )

            scored = _score_frame(
                symbol,
                sliced,
                benchmark=benchmark,
                prev_close=float(meta["prev_close"]),
                prev_close_available=bool(meta["prev_ok"]),
            )

            if not scored:
                continue

            row = {
                "리플레이날짜": date_text,
                "리플레이시각ET": cutoff.isoformat(),
                **scored,
            }
            ranked_rows.append(row)

        if ranked_rows:
            snap = pd.DataFrame(ranked_rows)

            snap["_green"] = (
                snap["판정"]
                .astype(str)
                .str.contains("매수 후보", na=False)
                .astype(int)
            )

            snap = snap.sort_values(
                [
                    "_green",
                    "종합점수",
                    "상대강도",
                    "당일등락률",
                    "최근5분수익률",
                    "거래량배수",
                ],
                ascending=[False, False, False, False, False, False],
            ).reset_index(drop=True)

            snap["리플레이순위"] = [
                i + 1 for i in range(len(snap))
            ]
            snap = snap.drop(columns=["_green"])

            rows.extend(
                snap.to_dict("records")
            )

        cutoff += pd.Timedelta(
            minutes=step_minutes
        )

    if not rows:
        raise RuntimeError("리플레이 결과가 없습니다.")

    df = pd.DataFrame(rows)
    summary_rows = []

    for symbol in symbols:
        s = df[df["종목코드"] == symbol].copy()
        if s.empty:
            continue

        greens = s[
            s["판정"]
            .astype(str)
            .str.contains("매수 후보", na=False)
        ]

        top_rank_rows = s[
            s["리플레이순위"] == 1
        ]

        summary_rows.append({
            "종목코드": symbol,
            "전일종가": float(s.iloc[-1]["전일종가"]),
            "전일종가확인": bool(s.iloc[-1]["전일종가확인"]),
            "최고점수": float(s["종합점수"].max()),
            "최고순위": int(s["리플레이순위"].min()),
            "매수후보횟수": int(len(greens)),
            "1위횟수": int(len(top_rank_rows)),
            "최초매수후보시각ET": (
                str(greens.iloc[0]["리플레이시각ET"])
                if not greens.empty
                else ""
            ),
            "마지막당일등락률": float(s.iloc[-1]["당일등락률"]),
            "마지막QQQ당일수익률": float(s.iloc[-1]["QQQ당일수익률"]),
            "마지막상대강도": float(s.iloc[-1]["상대강도"]),
            "급락반등함정횟수": int(
                pd.Series(s["급락반등함정"])
                .fillna(False)
                .astype(bool)
                .sum()
            ),
            "대장주자격횟수": int(
                pd.Series(s["대장주자격"])
                .fillna(False)
                .astype(bool)
                .sum()
            ),
        })

    summary_df = pd.DataFrame(summary_rows)

    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            [
                "매수후보횟수",
                "1위횟수",
                "최고점수",
            ],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    out_csv = REPLAY_DIR / f"us_replay_{date_text}_v31.csv"
    out_summary_csv = REPLAY_DIR / f"us_replay_{date_text}_v31_summary.csv"
    out_json = REPLAY_DIR / f"us_replay_{date_text}_v31_summary.json"

    df.to_csv(
        out_csv,
        index=False,
        encoding="utf-8-sig",
    )

    summary_df.to_csv(
        out_summary_csv,
        index=False,
        encoding="utf-8-sig",
    )

    payload = {
        "ok": True,
        "version": "V3.1-prev-close",
        "date": date_text,
        "symbols": symbols,
        "step_minutes": step_minutes,
        "generated_at": datetime.now(ET).isoformat(timespec="seconds"),
        "summary": summary_df.to_dict("records"),
        "files": {
            "detail_csv": str(out_csv),
            "summary_csv": str(out_summary_csv),
            "summary_json": str(out_json),
        },
    }

    out_json.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    return payload


if __name__ == "__main__":
    result = run_replay(
        date_text=os.getenv(
            "REPLAY_DATE",
            "2026-08-14",
        ),
        symbols=[
            x.strip()
            for x in os.getenv(
                "REPLAY_SYMBOLS",
                "TSLA,NVDA,AMAT",
            ).split(",")
            if x.strip()
        ],
        step_minutes=int(
            os.getenv(
                "REPLAY_STEP_MINUTES",
                "5",
            )
        ),
    )

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )

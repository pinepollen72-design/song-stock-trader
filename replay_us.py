from __future__ import annotations

import json
import os
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from strategy_us import (
    BENCHMARK_SYMBOL,
    _benchmark_metrics,
    _extract_symbol_frame,
    _score_frame,
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
    yfinance에서 지정 날짜 1분봉을 한 번에 받습니다.
    리플레이는 최근 intraday 보존 범위 안에서 사용하는 것을 전제로 합니다.
    """
    try:
        import yfinance as yf
    except Exception as e:
        raise RuntimeError(f"yfinance import 실패: {e}") from e

    day = pd.Timestamp(date_text)
    next_day = day + pd.Timedelta(days=1)

    download_symbols = list(symbols)
    if BENCHMARK_SYMBOL not in download_symbols:
        download_symbols.append(BENCHMARK_SYMBOL)

    try:
        batch = yf.download(
            tickers=download_symbols,
            start=day.strftime("%Y-%m-%d"),
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
            start=day.strftime("%Y-%m-%d"),
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
    지정 날짜 미국장을 5분 간격으로 재평가합니다.

    결과:
    - CSV: 종목/시각별 점수와 탈락 사유
    - JSON: 요약 + 최고점수 + 매수후보 최초시각
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

    symbol_frames = {}
    for symbol in symbols:
        frame = _extract_symbol_frame(batch, symbol, universe_size)
        if frame is not None and not frame.empty:
            symbol_frames[symbol] = frame

    if not symbol_frames:
        raise RuntimeError("리플레이 가능한 종목 데이터가 없습니다.")

    date_ts = pd.Timestamp(date_text, tz=ET)
    start = date_ts.replace(hour=9, minute=35, second=0)
    end = date_ts.replace(hour=15, minute=55, second=0)

    rows = []

    cutoff = start
    while cutoff <= end:
        qqq_slice = _slice_until(benchmark_full, cutoff)
        benchmark = _benchmark_metrics(qqq_slice)

        ranked_rows = []

        for symbol, full_frame in symbol_frames.items():
            sliced = _slice_until(full_frame, cutoff)
            scored = _score_frame(
                symbol,
                sliced,
                benchmark=benchmark,
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
            rows.extend(snap.to_dict("records"))

        cutoff += pd.Timedelta(minutes=step_minutes)

    if not rows:
        raise RuntimeError("리플레이 결과가 없습니다.")

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------
    # 요약
    # ------------------------------------------------------------
    summary_rows = []

    for symbol in symbols:
        s = df[df["종목코드"] == symbol].copy()
        if s.empty:
            continue

        greens = s[
            s["판정"].astype(str).str.contains(
                "매수 후보",
                na=False,
            )
        ]

        top_rank_rows = s[s["리플레이순위"] == 1]

        summary_rows.append({
            "종목코드": symbol,
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

    out_csv = REPLAY_DIR / f"us_replay_{date_text}.csv"
    out_summary_csv = REPLAY_DIR / f"us_replay_{date_text}_summary.csv"
    out_json = REPLAY_DIR / f"us_replay_{date_text}_summary.json"

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
        date_text=os.getenv("REPLAY_DATE", "2026-08-14"),
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

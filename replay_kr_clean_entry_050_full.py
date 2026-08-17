from __future__ import annotations

"""CLEAN ENTRY 0/50 overlay on the verified D5 full-engine harness.

Purpose
- Test entry quality only.
- No 25% probe in the CLEAN variant.
- Good/clean setup => 50%.
- Risky/weak setup => 0% until confirmation.
- STOP_LOSS same-day re-entry blocked.
- PG2 re-entry requires fresh breakout; otherwise no probe.
- Frozen D-v2 STOP/TAKE1/TAKE2/PROFIT_GUARD stays unchanged.
"""

import json
import threading
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import replay_kr
import replay_kr_open_defense_v2 as d2
import replay_kr_d3_edge_entry_full as d3
import replay_kr_d5_profit_shield_combo_full as d5

KST = d2.KST
CLEAN_VERSION = "kr-clean-entry-050-fast-v1"

_THREAD: threading.Thread | None = None
_LOCK = threading.RLock()

_ORIG_RUN = d5.run_kr_d5_profit_shield_combo_replay
_ORIG_TOP5 = d3._top5_cached
_ORIG_MARKET = d3._market_snapshot_cached
_ORIG_BREAKOUT = d3._breakout_allowed
_ORIG_ROW_STATE = d3._row_state
_ORIG_DEFENSE_AMOUNT = d2._defense_entry_amount


def _paths():
    base = d5._base()
    root = base.ROOT / "clean_entry_050_full_engine"
    day_dir = root / "daily"
    state_file = root / "state.json"
    result_file = root / "result.json"
    root.mkdir(parents=True, exist_ok=True)
    day_dir.mkdir(parents=True, exist_ok=True)
    return root, day_dir, state_file, result_file


def _read_json(path: Path, default=None):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


def _write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    tmp.replace(path)


def _public_state() -> dict:
    _, _, state_file, _ = _paths()
    out = dict(_read_json(state_file, {}) or {})
    if not out:
        out = {
            "ok": True,
            "version": CLEAN_VERSION,
            "status": "not_started",
            "result_ready": False,
        }
    out["ok"] = True
    out["version"] = CLEAN_VERSION
    out["thread_alive"] = bool(_THREAD and _THREAD.is_alive())
    return out


def _variant_config(config: d2.OpenDefenseConfig):
    """Disable the 25% opening-defense window for the CLEAN variant."""
    try:
        return replace(
            config,
            defense_start_time="09:09",
            defense_end_time="09:09",
        )
    except Exception:
        cfg = deepcopy(config)
        cfg.defense_start_time = "09:09"
        cfg.defense_end_time = "09:09"
        return cfg


def _run_clean_day(
    date_text: str,
    codes: Iterable[str] | None = None,
    config: d2.OpenDefenseConfig | None = None,
    mode: str = "CONTROL",
) -> dict:
    mode = str(mode or "CONTROL").upper().strip()

    if mode == "CONTROL":
        # Keep sentinel CONTROL on the original verified path.
        return _ORIG_RUN(
            date_text=date_text,
            codes=codes,
            config=config,
            mode="CONTROL",
        )

    if mode != "D5_COMBO":
        raise ValueError("mode must be CONTROL or D5_COMBO")

    top3_streak: dict[str, int] = {}
    deep_locked: set[str] = set()
    overheat_locked: set[str] = set()
    overheat_pullback_ready: set[str] = set()

    def filtered_top5(frames, meta, day, now, scan_count):
        raw = _ORIG_TOP5(frames, meta, day, now, scan_count)
        if raw is None or raw.empty:
            return raw

        current_top3: set[str] = set()
        for _, rr in raw.iterrows():
            sym = str(rr.get("종목코드", "")).zfill(6)
            st = _ORIG_ROW_STATE(rr)
            if st["signal"] and not st["weak"] and st["rank"] <= 3:
                current_top3.add(sym)

        for sym in list(top3_streak):
            if sym not in current_top3:
                top3_streak[sym] = 0
        for sym in current_top3:
            top3_streak[sym] = int(top3_streak.get(sym, 0)) + 1

        snap = _ORIG_MARKET(frames, day, now, raw)
        market_red = bool((snap or {}).get("regime") == "RED")

        keep: list[int] = []

        for idx, row in raw.iterrows():
            sym = str(row.get("종목코드", "")).zfill(6)
            st = _ORIG_ROW_STATE(row)

            if not st["signal"] or st["weak"] or st["rank"] > 3:
                continue

            frame = frames.get(sym)
            if frame is None:
                continue

            ref_price = float(replay_kr._price_at(frame, day, now) or 0.0)
            if ref_price <= 0:
                continue

            vwap_gap = float(
                d2._vwap_gap_pct(frame, day, now, ref_price)
            )

            # Once a dangerous location is observed, require recovery proof.
            if vwap_gap <= -1.00:
                deep_locked.add(sym)

            if vwap_gap >= 3.00:
                overheat_locked.add(sym)

            if sym in overheat_locked and vwap_gap <= 2.20:
                overheat_pullback_ready.add(sym)

            a_plus = d5._is_a_plus(st, vwap_gap)

            # Normal clean location:
            # TOP3 + reasonable score + VWAP 0~2.8 + positive 3m/5m.
            direct_quality = bool(
                st["score"] >= 55.0
                and 0.0 <= vwap_gap <= 2.80
                and st["ret3"] > 0.0
                and st["ret5"] > 0.0
            )

            # Confirmation quality for RED/risky locations.
            confirm_quality = bool(
                st["score"] >= 65.0
                and 0.0 <= vwap_gap <= 2.80
                and st["ret3"] > 0.0
                and st["ret5"] > 0.0
                and st["volume"] >= 1.00
            )

            needs_confirm = bool(
                sym in deep_locked
                or sym in overheat_locked
                or (market_red and not a_plus)
                or (not market_red and not direct_quality)
            )

            # Good location + normal market => immediate 50%.
            if not needs_confirm:
                if direct_quality or a_plus:
                    keep.append(idx)
                continue

            # Risky/RED => no entry until persistent TOP3 + real breakout.
            breakout_ok, _ = _ORIG_BREAKOUT(
                frame,
                day,
                now,
                ref_price,
                st,
                int(top3_streak.get(sym, 0)),
                vwap_gap,
            )

            deep_ok = (
                sym not in deep_locked
                or vwap_gap >= 0.0
            )

            overheat_ok = (
                sym not in overheat_locked
                or sym in overheat_pullback_ready
            )

            if (
                breakout_ok
                and confirm_quality
                and deep_ok
                and overheat_ok
            ):
                keep.append(idx)
                deep_locked.discard(sym)
                overheat_locked.discard(sym)
                overheat_pullback_ready.discard(sym)

        if keep:
            return raw.loc[keep].copy()
        return raw.iloc[0:0].copy()

    def nonred_snapshot(frames, day, now, top5):
        # CLEAN pre-filter already handled RED/risky entry logic.
        # Force D5's extra post-entry shield OFF so exits remain frozen D-v2.
        snap = dict(_ORIG_MARKET(frames, day, now, top5) or {})
        snap["regime"] = "GREEN_OR_NEUTRAL"
        return snap

    cfg = _variant_config(config or d2.OpenDefenseConfig())

    # Reuse D5's proven STOP re-entry block and PG2 rearm logic,
    # but remove every 25% probe by making the defense amount zero.
    d3._top5_cached = filtered_top5
    d3._market_snapshot_cached = nonred_snapshot
    d2._defense_entry_amount = lambda _cfg: 0

    try:
        out = _ORIG_RUN(
            date_text=date_text,
            codes=codes,
            config=cfg,
            mode="D5_COMBO",
        )
    finally:
        d3._top5_cached = _ORIG_TOP5
        d3._market_snapshot_cached = _ORIG_MARKET
        d2._defense_entry_amount = _ORIG_DEFENSE_AMOUNT

    out = dict(out)
    out["version"] = CLEAN_VERSION
    out["strategy"] = "CLEAN_ENTRY_0_50"
    out["mode"] = "CLEAN_050"
    out["rules"] = {
        "entry_size": "0% or 50%; no 25% probe",
        "normal_market": (
            "TOP3 + score>=55 + VWAP 0~+2.8 + ret3/ret5>0 => direct 50%"
        ),
        "red_market": (
            "A+ direct; otherwise persistent TOP3 + fresh 5-bar breakout "
            "+ score>=65 + volume>=1.0"
        ),
        "deep_vwap": (
            "once <=-1.0% observed, VWAP recovery + fresh breakout required"
        ),
        "overheat": (
            "once >=+3.0% observed, pullback <=+2.2% then fresh breakout required"
        ),
        "stop_reentry": "same-day blocked",
        "pg2_reentry": "fresh breakout required; probe fallback is zero",
        "exit_engine": (
            "frozen D-v2 STOP/TAKE1/TAKE2/PROFIT_GUARD; no extra shield exit"
        ),
    }
    return out


def _transform_result() -> None:
    _, _, state_file, result_file = _paths()
    payload = _read_json(result_file, {}) or {}
    if not payload:
        return

    payload["version"] = CLEAN_VERSION
    payload["mode"] = "PATH_CONSISTENT_CLEAN_ENTRY_050_FAST"
    payload["executed_variants"] = ["CLEAN_050"]

    for v in payload.get("variants", []) or []:
        if isinstance(v, dict) and v.get("id") == "D5_COMBO":
            v["id"] = "CLEAN_050"
            v["label"] = "CLEAN ENTRY 0/50 · entry-only test"

    for key in (
        "profit_preservation",
        "profitable_month_preservation",
        "monthly",
    ):
        obj = payload.get(key)
        if isinstance(obj, dict) and "D5_COMBO" in obj:
            obj["CLEAN_050"] = obj.pop("D5_COMBO")

    for row in payload.get("daily", []) or []:
        if not isinstance(row, dict):
            continue
        row["version"] = CLEAN_VERSION
        if "D5_COMBO" in row:
            row["CLEAN_050"] = row.pop("D5_COMBO")

    payload["rules"] = {
        "entry_size": "0% or 50%; no 25% probe",
        "normal_market": "clean location/leadership can enter direct",
        "red_or_risky": "confirmation required unless RED A+",
        "deep_vwap": "<=-1.0% observed => VWAP recovery + breakout",
        "overheat": ">=+3.0% observed => pullback <=+2.2% + re-break",
        "stop_reentry": "blocked same day",
        "pg2_reentry": "fresh breakout only; no probe fallback",
        "exit_engine": "frozen D-v2 exits",
    }

    _write_json(result_file, payload)

    state = _read_json(state_file, {}) or {}
    state.update(
        {
            "version": CLEAN_VERSION,
            "phase": "DONE" if payload.get("ok") else state.get("phase"),
            "message": (
                "CLEAN ENTRY 0/50 검증 완료"
                if payload.get("ok")
                else state.get("message")
            ),
        }
    )
    _write_json(state_file, state)


def _job(
    result,
    provider,
    codes,
    frozen_config,
    protected_window_fn,
) -> None:
    # Reuse D5's proven 147-day cache/parity/aggregation harness,
    # but point it to a separate CLEAN directory and replace only
    # the variant day engine.
    orig_version = d5.D5_VERSION
    orig_paths = d5._paths
    orig_run = d5.run_kr_d5_profit_shield_combo_replay
    orig_state = d5._state

    def clean_state(**updates):
        if str(updates.get("phase", "")) == "D5_PROFIT_SHIELD_COMBO":
            updates["phase"] = "CLEAN_ENTRY_0_50"

        msg = str(updates.get("message", "") or "")
        if msg:
            updates["message"] = (
                msg.replace("D5 COMBO", "CLEAN 0/50")
                .replace("D5", "CLEAN")
            )
        return orig_state(**updates)

    d5.D5_VERSION = CLEAN_VERSION
    d5._paths = _paths
    d5.run_kr_d5_profit_shield_combo_replay = _run_clean_day
    d5._state = clean_state

    try:
        d5._job(
            result,
            provider,
            codes,
            frozen_config,
            protected_window_fn,
        )
        _transform_result()
    finally:
        d5.D5_VERSION = orig_version
        d5._paths = orig_paths
        d5.run_kr_d5_profit_shield_combo_replay = orig_run
        d5._state = orig_state


def ensure_clean_entry_050_started(
    result: dict,
    provider,
    codes,
    frozen_config,
    protected_window_fn,
) -> dict:
    global _THREAD

    _, _, _, result_file = _paths()
    existing = _read_json(result_file, {}) or {}

    base_total = int(
        ((result.get("overall") or {}).get("D2_total_KRW", 0))
        or 0
    )

    if (
        existing.get("ok") is True
        and existing.get("version") == CLEAN_VERSION
        and int(
            (
                (existing.get("parity") or {})
                .get("cached_D2_expected_total_KRW", base_total)
            )
            or base_total
        )
        == base_total
    ):
        compact = dict(existing)
        compact.pop("daily", None)
        return compact

    with _LOCK:
        if _THREAD and _THREAD.is_alive():
            return _public_state()

        _THREAD = threading.Thread(
            target=_job,
            args=(
                dict(result),
                provider,
                list(codes),
                frozen_config,
                protected_window_fn,
            ),
            daemon=True,
            name="kr-clean-entry-050",
        )
        _THREAD.start()

        out = _public_state()
        out["started"] = True
        return out

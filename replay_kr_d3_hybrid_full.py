from __future__ import annotations

"""Compatibility bridge: route the existing d3_hybrid slot to KR SWING V2."""

import os
from datetime import datetime

import replay_kr_swing_v2_full as swing

HYBRID_VERSION = swing.SWING_VERSION

_DEFAULT_KR_CLOSED_DATES = {
    "2026-08-17",  # Liberation Day substitute holiday
}


def _closed_dates() -> set[str]:
    extra = {
        x.strip()
        for x in os.getenv("SWING_KR_CLOSED_DATES", "").split(",")
        if x.strip()
    }
    return set(_DEFAULT_KR_CLOSED_DATES) | extra


def _market_aware_protected_window(original_fn):
    live, label = original_fn()
    if not live:
        return live, label

    if label == "KR_LIVE":
        today = datetime.now(swing.KST).strftime("%Y-%m-%d")
        if today in _closed_dates():
            return False, "KR_MARKET_CLOSED_OVERRIDE"

    return live, label


def ensure_d3_hybrid_started(
    result: dict,
    provider=None,
    codes=None,
    frozen_config=None,
    protected_window_fn=None,
) -> dict:
    if protected_window_fn is None:
        return {
            "ok": False,
            "version": HYBRID_VERSION,
            "status": "error",
            "result_ready": False,
            "started": False,
            "message": "SWING V2 bridge requires protected-window callback.",
        }

    def protected():
        return _market_aware_protected_window(protected_window_fn)

    out = swing.ensure_swing_v2_started(
        result=result,
        provider=provider,
        codes=codes,
        frozen_config=frozen_config,
        protected_window_fn=protected,
    )

    out = dict(out)
    out["compatibility_slot"] = "d3_hybrid_full_engine"
    out["actual_experiment"] = "SWING_V2_DUAL_ENTRY"
    out["speed_profile"] = {
        "strategy_family": "multi_day_swing",
        "entry_routes": [
            "PULLBACK_RECOVERY",
            "BREAKOUT_20D",
        ],
        "daily_bars_only": True,
        "signal_close_fill_next_open": True,
        "max_positions": 3,
        "risk_sized": True,
        "reuses_v1_daily_cache": True,
        "real_orders": False,
        "kr_closed_date_override": sorted(_closed_dates()),
    }
    return out

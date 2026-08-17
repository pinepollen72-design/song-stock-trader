from __future__ import annotations

"""Compatibility bridge: route the existing d3_hybrid slot to CLEAN ENTRY 0/50."""

import os
from datetime import datetime

import replay_kr_clean_entry_050_full as clean

HYBRID_VERSION = clean.CLEAN_VERSION

_DEFAULT_KR_CLOSED_DATES = {
    "2026-08-17",  # Liberation Day substitute holiday
}


def _closed_dates() -> set[str]:
    extra = {
        x.strip()
        for x in os.getenv("CLEAN_KR_CLOSED_DATES", "").split(",")
        if x.strip()
    }
    return set(_DEFAULT_KR_CLOSED_DATES) | extra


def _market_aware_protected_window(original_fn):
    live, label = original_fn()
    if not live:
        return live, label

    if label == "KR_LIVE":
        today = datetime.now(clean.KST).strftime("%Y-%m-%d")
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
    if (
        provider is None
        or codes is None
        or frozen_config is None
        or protected_window_fn is None
    ):
        return {
            "ok": False,
            "version": HYBRID_VERSION,
            "status": "error",
            "result_ready": False,
            "started": False,
            "message": (
                "CLEAN bridge requires cached provider/codes/"
                "frozen config/protected-window callback."
            ),
        }

    def protected():
        return _market_aware_protected_window(protected_window_fn)

    out = clean.ensure_clean_entry_050_started(
        result=result,
        provider=provider,
        codes=codes,
        frozen_config=frozen_config,
        protected_window_fn=protected,
    )

    out = dict(out)
    out["compatibility_slot"] = "d3_hybrid_full_engine"
    out["actual_experiment"] = "CLEAN_ENTRY_0_50"
    out["speed_profile"] = {
        "single_variant_only": True,
        "frozen_control_cache_reused": True,
        "control_spot_audit_only": True,
        "variant_executed": "CLEAN_050",
        "no_25pct_probe": True,
        "kr_closed_date_override": sorted(_closed_dates()),
    }
    return out

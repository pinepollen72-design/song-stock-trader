from __future__ import annotations

"""Compatibility bridge: route the existing d3_hybrid slot to D5 single COMBO."""

import os
from datetime import datetime

import replay_kr_d5_profit_shield_combo_full as d5

HYBRID_VERSION = d5.D5_VERSION

_DEFAULT_KR_CLOSED_DATES = {
    "2026-08-17",  # Liberation Day substitute holiday
}


def _closed_dates() -> set[str]:
    extra = {
        x.strip()
        for x in os.getenv("D5_KR_CLOSED_DATES", "").split(",")
        if x.strip()
    }
    return set(_DEFAULT_KR_CLOSED_DATES) | extra


def _market_aware_protected_window(original_fn):
    live, label = original_fn()
    if not live:
        return live, label

    if label == "KR_LIVE":
        today = datetime.now(d5.KST).strftime("%Y-%m-%d")
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
    if provider is None or codes is None or frozen_config is None or protected_window_fn is None:
        return {
            "ok": False,
            "version": HYBRID_VERSION,
            "status": "error",
            "result_ready": False,
            "started": False,
            "message": "D5 bridge requires cached provider/codes/frozen config/protected-window callback.",
        }

    def protected():
        return _market_aware_protected_window(protected_window_fn)

    out = d5.ensure_d5_profit_shield_combo_started(
        result=result,
        provider=provider,
        codes=codes,
        frozen_config=frozen_config,
        protected_window_fn=protected,
    )
    out = dict(out)
    out["compatibility_slot"] = "d3_hybrid_full_engine"
    out["actual_experiment"] = "D5_PROFIT_SHIELD_COMBO"
    out["speed_profile"] = {
        "single_variant_only": True,
        "frozen_control_cache_reused": True,
        "control_spot_audit_only": True,
        "variant_executed": "D5_COMBO",
        "shared_price_vwap_cache": True,
        "kr_closed_date_override": sorted(_closed_dates()),
    }
    return out

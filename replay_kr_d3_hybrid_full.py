from __future__ import annotations

"""D4 FAST compatibility bridge.

The long backtest already calls the d3_hybrid slot. This bridge routes that
read-only slot to D4 FAST and makes the protected-window callback aware of known
KRX closure overrides, so a weekday substitute holiday does not waste the day.
"""

import os
from datetime import datetime

import replay_kr_d4_loss_router_full as d4

HYBRID_VERSION = d4.D4_VERSION

# Immediate closure override needed for the current run.
# Additional dates can be supplied without another code change:
# D4_KR_CLOSED_DATES=2026-09-24,2026-09-25,...
_DEFAULT_KR_CLOSED_DATES = {
    "2026-08-17",  # Liberation Day substitute holiday
}


def _closed_dates() -> set[str]:
    extra = {
        x.strip()
        for x in os.getenv("D4_KR_CLOSED_DATES", "").split(",")
        if x.strip()
    }
    return set(_DEFAULT_KR_CLOSED_DATES) | extra


def _market_aware_protected_window(original_fn):
    live, label = original_fn()
    if not live:
        return live, label

    if label == "KR_LIVE":
        today = datetime.now(d4.KST).strftime("%Y-%m-%d")
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
            "message": "D4 FAST bridge requires cached provider/codes/frozen config/protected-window callback.",
        }

    def protected():
        return _market_aware_protected_window(protected_window_fn)

    out = d4.ensure_d4_loss_router_started(
        result=result,
        provider=provider,
        codes=codes,
        frozen_config=frozen_config,
        protected_window_fn=protected,
    )
    out = dict(out)
    out["compatibility_slot"] = "d3_hybrid_full_engine"
    out["actual_experiment"] = "D4_LOSS_ROUTER_BAD_DAY_BRAKE_FAST"
    out["speed_profile"] = {
        "frozen_control_cache_reused": True,
        "control_spot_audit_only": True,
        "variants_executed": ["LOSS_ROUTER", "ROUTER_BRAKE"],
        "shared_price_vwap_cache": True,
        "kr_closed_date_override": sorted(_closed_dates()),
    }
    return out

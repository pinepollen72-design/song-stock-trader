from __future__ import annotations

"""Compatibility bridge.

replay_kr_long_backtest.py already calls:
    d3_hybrid_replay.ensure_d3_hybrid_started(...)

To avoid touching Worker/live-order code again, this bridge forwards that existing
read-only result slot to the D4 LOSS ROUTER + BAD DAY BRAKE validation engine.
"""

import replay_kr_d4_loss_router_full as d4

HYBRID_VERSION = d4.D4_VERSION


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
            "message": "D4 bridge requires cached provider/codes/frozen config/protected-window callback.",
        }

    out = d4.ensure_d4_loss_router_started(
        result=result,
        provider=provider,
        codes=codes,
        frozen_config=frozen_config,
        protected_window_fn=protected_window_fn,
    )
    out = dict(out)
    out["compatibility_slot"] = "d3_hybrid_full_engine"
    out["actual_experiment"] = "D4_LOSS_ROUTER_BAD_DAY_BRAKE"
    return out

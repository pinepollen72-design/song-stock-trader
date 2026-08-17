from __future__ import annotations

"""Compatibility stub for an optional D3 hybrid experiment.

The active experiment is D3 EDGE ENTRY. This module exists only so
replay_kr_long_backtest.py revisions that reference d3_hybrid_replay can import
cleanly without starting another backtest or changing trading behavior.
"""

HYBRID_VERSION = "kr-d3-hybrid-compat-stub-v1"


def ensure_d3_hybrid_started(
    result: dict,
    provider=None,
    codes=None,
    frozen_config=None,
    protected_window_fn=None,
) -> dict:
    return {
        "ok": True,
        "version": HYBRID_VERSION,
        "status": "disabled",
        "result_ready": False,
        "started": False,
        "message": "D3 HYBRID is disabled; D3 EDGE ENTRY remains the active experiment.",
        "read_only": True,
    }

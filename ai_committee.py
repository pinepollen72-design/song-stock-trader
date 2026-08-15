from __future__ import annotations

import csv
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests

KST = ZoneInfo("Asia/Seoul")
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
STATE_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = STATE_DIR / "ai_committee_state.json"
LOG_FILE = STATE_DIR / "ai_committee_log.csv"
WORKER_EVENT_FILE = STATE_DIR / "ai_committee_worker_events.csv"


@dataclass(frozen=True)
class CommitteeConfig:
    # V1은 반드시 그림자 모드. 주문 판단에는 사용하지 않는다.
    enabled: bool = os.getenv("AI_COMMITTEE_ENABLED", "true").lower() in (
        "1", "true", "yes", "on"
    )
    shadow_mode: bool = True

    api_key: str = os.getenv("OPENAI_API_KEY", "").strip()
    model: str = os.getenv("AI_COMMITTEE_MODEL", "gpt-5.6").strip() or "gpt-5.6"

    timeout_seconds: float = float(
        os.getenv("AI_COMMITTEE_TIMEOUT_SECONDS", "8")
    )
    cache_ttl_seconds: int = max(
        60, int(os.getenv("AI_COMMITTEE_CACHE_SECONDS", "300"))
    )
    max_candidates_per_scan: int = max(
        1, min(5, int(os.getenv("AI_COMMITTEE_MAX_CANDIDATES", "3")))
    )

    technical_weight: float = 0.45
    market_weight: float = 0.20
    risk_weight: float = 0.35

    approve_score: float = 75.0
    hold_score: float = 60.0
    risk_veto_score: float = 40.0
    risk_reject_score: float = 30.0


CFG = CommitteeConfig()

_LOCK = threading.RLock()
_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="ai-investment-committee",
)
_PENDING: Future | None = None


DECISION_FIELDS = [
    "시간",
    "시장",
    "종목코드",
    "종목명",
    "순위",
    "전략점수",
    "전략판정",
    "현재가",
    "당일등락률",
    "최근3분수익률",
    "최근5분수익률",
    "최근10분수익률",
    "최근20분수익률",
    "거래량배수",
    "고점대비",
    "VWAP괴리율",
    "상대강도",
    "돌파",
    "추세상승",
    "기술점수",
    "기술투표",
    "기술이유",
    "시장점수",
    "시장투표",
    "시장이유",
    "리스크점수",
    "리스크투표",
    "리스크이유",
    "위원회점수",
    "위원회판정",
    "신뢰도",
    "플래그",
    "모델",
    "API지연ms",
    "그림자모드",
    "입력스냅샷",
]


WORKER_EVENT_FIELDS = [
    "시간",
    "시장",
    "종목코드",
    "종목명",
    "액션",
    "상태",
    "수량",
    "가격",
    "손익률",
    "사유",
    "최근AI판정",
    "최근AI점수",
    "최근AI시간",
]


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "evaluations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "technical_score": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100,
                    },
                    "technical_vote": {
                        "type": "string",
                        "enum": ["APPROVE", "HOLD", "REJECT"],
                    },
                    "technical_reason": {"type": "string"},
                    "market_score": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100,
                    },
                    "market_vote": {
                        "type": "string",
                        "enum": ["APPROVE", "HOLD", "REJECT"],
                    },
                    "market_reason": {"type": "string"},
                    "risk_score": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100,
                    },
                    "risk_vote": {
                        "type": "string",
                        "enum": ["APPROVE", "HOLD", "REJECT"],
                    },
                    "risk_reason": {"type": "string"},
                    "confidence": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100,
                    },
                    "flags": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "symbol",
                    "technical_score",
                    "technical_vote",
                    "technical_reason",
                    "market_score",
                    "market_vote",
                    "market_reason",
                    "risk_score",
                    "risk_vote",
                    "risk_reason",
                    "confidence",
                    "flags",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["evaluations"],
    "additionalProperties": False,
}


SYSTEM_PROMPT = """\
너는 단기 자동매매 시스템의 '그림자 AI 투자위원회'다.
실제 주문 권한은 없고, 기존 전략이 이미 뽑은 매수 후보만 평가한다.

서로 다른 세 위원 관점으로 각 후보를 독립적으로 평가하라.

1) 기술·대장주 위원
- 당일 강도, 최근 3/5/10/20분 모멘텀, 거래량 가속,
  장중 고점 근접도, 돌파, VWAP, 상대강도를 본다.
- 급락 뒤 짧은 반등을 '진짜 대장주'로 과대평가하지 않는다.
- 이미 과도하게 튄 뒤 꺾이는 추격 진입도 감점한다.

2) 시장환경 위원
- 오직 입력에 제공된 시장/후보군 정보만 사용한다.
- 뉴스, 지수, 업종 정보를 입력에 없는데 아는 척하거나 만들어내지 않는다.
- 미국 후보는 상대강도 정보가 있으면 중요하게 본다.
- 국내는 시장지수 데이터가 없으면 후보군의 폭과 강도를 제한된 대리변수로 사용하고
  불확실성을 confidence에 반영한다.

3) 리스크관리 위원
- 높은 점수 자체보다 '지금 진입했을 때 틀리면 얼마나 위험한가'를 본다.
- 장 초반 과열, 추격매수, 고점 이탈, 모멘텀 불일치, 거래량 둔화,
  이미 보유 포지션이 많은 상태를 경계한다.
- risk_score는 위험도가 아니라 '안전도'다. 높을수록 상대적으로 안전하다.

중요:
- 수익을 보장한다고 표현하지 않는다.
- 미래 가격을 단정하지 않는다.
- 제공되지 않은 사실은 추측하지 않는다.
- reason은 각 위원당 짧고 구체적인 한 문장으로 쓴다.
- flags는 STRONG_MOMENTUM, NEAR_HIGH, BREAKOUT, CHASE_RISK,
  REBOUND_TRAP_RISK, WEAK_VOLUME, MARKET_UNCERTAIN 같은 짧은 코드형 문자열을 쓴다.
"""


def _load_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data
    except Exception:
        pass
    return default


def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    tmp.replace(path)


def _append_csv(path: Path, fields: list[str], row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        exists = path.exists() and path.stat().st_size > 0
        with path.open("a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fields,
                extrasaction="ignore",
            )
            if not exists:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in fields})


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str):
            value = value.replace("%", "").replace(",", "").strip()
            if not value:
                return default
        x = float(value)
        if pd.isna(x):
            return default
        return x
    except Exception:
        return default


def _safe_int(value: Any, default: int = 999) -> int:
    if isinstance(value, str):
        m = __import__("re").search(r"(\d+)", value)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return default
    try:
        return int(value)
    except Exception:
        return default


def _first(row: dict, *names: str, default: Any = "") -> Any:
    for name in names:
        if name in row:
            value = row.get(name)
            if value is not None and str(value) != "":
                return value
    return default


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    s = str(value or "").strip().lower()
    return s in (
        "1", "true", "yes", "on", "y", "상승", "돌파", "yes",
        "true", "o", "○",
    )


def _candidate_snapshot(row: dict, fallback_rank: int) -> dict:
    symbol = str(
        _first(
            row,
            "종목코드",
            "symbol",
            "ticker",
            default="",
        )
    ).strip().upper()

    name = str(
        _first(
            row,
            "종목명",
            "name",
            default=symbol,
        )
    ).strip()

    signal = str(
        _first(
            row,
            "판정",
            "종합신호",
            "signal",
            default="",
        )
    ).strip()

    rank = _safe_int(
        _first(row, "순위", "rank", default=fallback_rank),
        fallback_rank,
    )

    return {
        "symbol": symbol,
        "name": name or symbol,
        "rank": rank,
        "strategy_score": round(
            _safe_float(
                _first(
                    row,
                    "종합점수",
                    "combined_score",
                    "score",
                    default=0,
                )
            ),
            3,
        ),
        "signal": signal,
        "price": round(
            _safe_float(
                _first(row, "현재가", "price", default=0)
            ),
            6,
        ),
        "day_return": round(
            _safe_float(
                _first(
                    row,
                    "당일등락률",
                    "등락률",
                    "day_return",
                    default=0,
                )
            ),
            3,
        ),
        "ret3": round(
            _safe_float(
                _first(row, "최근3분수익률", "ret3", default=0)
            ),
            3,
        ),
        "ret5": round(
            _safe_float(
                _first(row, "최근5분수익률", "ret5", default=0)
            ),
            3,
        ),
        "ret10": round(
            _safe_float(
                _first(row, "최근10분수익률", "ret10", default=0)
            ),
            3,
        ),
        "ret20": round(
            _safe_float(
                _first(row, "최근20분수익률", "ret20", default=0)
            ),
            3,
        ),
        "volume_ratio": round(
            _safe_float(
                _first(
                    row,
                    "거래량배수",
                    "거래량가속",
                    "volume_ratio",
                    default=0,
                )
            ),
            3,
        ),
        "high_distance": round(
            _safe_float(
                _first(
                    row,
                    "고점대비",
                    "고가근접도",
                    "high_distance",
                    default=0,
                )
            ),
            3,
        ),
        "vwap_gap": round(
            _safe_float(
                _first(
                    row,
                    "VWAP괴리율",
                    "vwap_gap",
                    default=0,
                )
            ),
            3,
        ),
        "relative_strength": round(
            _safe_float(
                _first(
                    row,
                    "상대강도",
                    "나스닥대비상대강도",
                    "relative_strength",
                    default=0,
                )
            ),
            3,
        ),
        "breakout": _boolish(
            _first(
                row,
                "돌파",
                "돌파여부",
                "breakout",
                default=False,
            )
        ),
        "trend_up": _boolish(
            _first(
                row,
                "추세상승",
                "trend_up",
                default=False,
            )
        )
        or "상승" in str(
            _first(row, "추세", default="")
        ),
    }


def _is_buy_candidate(item: dict) -> bool:
    signal = str(item.get("signal", "")).lower()
    if "매수 후보" in signal or "매수후보" in signal:
        return True
    if "buy" in signal:
        return True
    # 현재 전략은 매수 후보가 판정 필드에 명시된다.
    # 신호가 비어 있으면 AI가 임의로 후보를 만들어내지 않는다.
    return False


def _market_context(items: list[dict]) -> dict:
    if not items:
        return {
            "scan_count": 0,
            "buy_candidate_count": 0,
            "positive_day_count": 0,
            "avg_day_return": 0.0,
            "avg_ret5": 0.0,
            "avg_strategy_score": 0.0,
            "avg_relative_strength": 0.0,
            "top_strategy_score": 0.0,
        }

    def avg(key: str) -> float:
        vals = [_safe_float(x.get(key, 0)) for x in items]
        return round(sum(vals) / max(1, len(vals)), 3)

    return {
        "scan_count": len(items),
        "buy_candidate_count": sum(
            1 for x in items if _is_buy_candidate(x)
        ),
        "positive_day_count": sum(
            1 for x in items if _safe_float(x.get("day_return")) > 0
        ),
        "avg_day_return": avg("day_return"),
        "avg_ret5": avg("ret5"),
        "avg_strategy_score": avg("strategy_score"),
        "avg_relative_strength": avg("relative_strength"),
        "top_strategy_score": round(
            max(
                [_safe_float(x.get("strategy_score")) for x in items]
                or [0.0]
            ),
            3,
        ),
    }


def _signature(market: str, item: dict) -> str:
    # 작은 틱 변화마다 다시 AI를 부르지 않도록 핵심 값만 거칠게 반올림한다.
    payload = {
        "market": market,
        "symbol": item.get("symbol"),
        "rank": item.get("rank"),
        "score": round(_safe_float(item.get("strategy_score")), 0),
        "day": round(_safe_float(item.get("day_return")), 1),
        "r3": round(_safe_float(item.get("ret3")), 1),
        "r5": round(_safe_float(item.get("ret5")), 1),
        "r10": round(_safe_float(item.get("ret10")), 1),
        "vr": round(_safe_float(item.get("volume_ratio")), 1),
        "hd": round(_safe_float(item.get("high_distance")), 1),
        "rs": round(_safe_float(item.get("relative_strength")), 1),
        "bo": bool(item.get("breakout")),
        "tu": bool(item.get("trend_up")),
    }
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _state() -> dict:
    state = _load_json(
        STATE_FILE,
        {
            "version": "ai-committee-v1-shadow",
            "cache": {},
            "recent": [],
            "worker_events": [],
            "counts": {
                "total": 0,
                "APPROVE": 0,
                "HOLD": 0,
                "REJECT": 0,
                "errors": 0,
            },
        },
    )
    if not isinstance(state, dict):
        state = {}
    state.setdefault("version", "ai-committee-v1-shadow")
    state.setdefault("cache", {})
    state.setdefault("recent", [])
    state.setdefault("worker_events", [])
    state.setdefault(
        "counts",
        {
            "total": 0,
            "APPROVE": 0,
            "HOLD": 0,
            "REJECT": 0,
            "errors": 0,
        },
    )
    return state


def _save_state(state: dict) -> None:
    state["updated_at"] = datetime.now(KST).isoformat(timespec="seconds")
    state["configured"] = bool(CFG.api_key)
    state["enabled"] = bool(CFG.enabled)
    state["shadow_mode"] = True
    state["model"] = CFG.model
    _atomic_json(STATE_FILE, state)


def _chair(
    technical_score: float,
    market_score: float,
    risk_score: float,
) -> tuple[float, str, str]:
    weighted = (
        technical_score * CFG.technical_weight
        + market_score * CFG.market_weight
        + risk_score * CFG.risk_weight
    )
    weighted = round(weighted, 1)

    # 리스크 위원에게 제한적 거부권.
    if risk_score < CFG.risk_reject_score:
        return weighted, "REJECT", "RISK_VETO_STRONG"

    if risk_score < CFG.risk_veto_score:
        return weighted, "HOLD", "RISK_VETO"

    if weighted >= CFG.approve_score:
        return weighted, "APPROVE", "WEIGHTED_APPROVE"

    if weighted >= CFG.hold_score:
        return weighted, "HOLD", "WEIGHTED_HOLD"

    return weighted, "REJECT", "WEIGHTED_REJECT"


def _extract_output_text(data: dict) -> str:
    # Responses API REST 응답의 output -> message -> output_text를 안전하게 탐색.
    for output in data.get("output", []) or []:
        if not isinstance(output, dict):
            continue
        for content in output.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            if content.get("type") == "output_text":
                text = content.get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()

    # 일부 SDK/프록시 형식 대비
    text = data.get("output_text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    raise ValueError("Responses API에서 output_text를 찾지 못했습니다.")


def _call_openai(
    market: str,
    now_iso: str,
    candidates: list[dict],
    market_context: dict,
    portfolio_context: dict,
) -> tuple[dict, int]:
    if not CFG.api_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")

    user_payload = {
        "market": market,
        "time": now_iso,
        "shadow_mode": True,
        "candidates": candidates,
        "market_context": market_context,
        "portfolio_context": portfolio_context,
        "scoring_contract": {
            "technical_score": "0~100, 높을수록 진입 기술품질 우수",
            "market_score": "0~100, 높을수록 제공된 시장환경이 우호적",
            "risk_score": "0~100, 높을수록 현재 진입 위험이 상대적으로 낮음",
        },
    }

    body = {
        "model": CFG.model,
        "input": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": json.dumps(
                    user_payload,
                    ensure_ascii=False,
                    default=str,
                ),
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "ai_investment_committee",
                "strict": True,
                "schema": OUTPUT_SCHEMA,
            }
        },
        "max_output_tokens": 1400,
    }

    started = time.perf_counter()
    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {CFG.api_key}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=max(2.0, CFG.timeout_seconds),
    )
    latency_ms = int(
        round((time.perf_counter() - started) * 1000)
    )

    if not response.ok:
        body_text = response.text[:500]
        raise RuntimeError(
            f"OpenAI API HTTP {response.status_code}: {body_text}"
        )

    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("OpenAI API 응답이 JSON 객체가 아닙니다.")

    text = _extract_output_text(data)
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise RuntimeError("AI 위원회 응답이 객체가 아닙니다.")
    return parsed, latency_ms


def _record_error(message: str) -> None:
    with _LOCK:
        state = _state()
        counts = state.setdefault("counts", {})
        counts["errors"] = int(counts.get("errors", 0) or 0) + 1
        state["last_error"] = str(message)[:1000]
        state["last_error_at"] = datetime.now(KST).isoformat(
            timespec="seconds"
        )
        _save_state(state)


def _persist_decision(
    market: str,
    now_iso: str,
    item: dict,
    evaluation: dict,
    latency_ms: int,
) -> dict:
    tech = max(
        0.0, min(100.0, _safe_float(evaluation.get("technical_score")))
    )
    mkt = max(
        0.0, min(100.0, _safe_float(evaluation.get("market_score")))
    )
    risk = max(
        0.0, min(100.0, _safe_float(evaluation.get("risk_score")))
    )
    weighted, decision, chair_code = _chair(tech, mkt, risk)

    flags = [
        str(x).strip()
        for x in evaluation.get("flags", []) or []
        if str(x).strip()
    ]
    if chair_code not in flags:
        flags.append(chair_code)

    confidence = max(
        0.0, min(100.0, _safe_float(evaluation.get("confidence")))
    )

    record = {
        "시간": now_iso,
        "시장": market,
        "종목코드": item.get("symbol", ""),
        "종목명": item.get("name", ""),
        "순위": item.get("rank", ""),
        "전략점수": item.get("strategy_score", 0),
        "전략판정": item.get("signal", ""),
        "현재가": item.get("price", 0),
        "당일등락률": item.get("day_return", 0),
        "최근3분수익률": item.get("ret3", 0),
        "최근5분수익률": item.get("ret5", 0),
        "최근10분수익률": item.get("ret10", 0),
        "최근20분수익률": item.get("ret20", 0),
        "거래량배수": item.get("volume_ratio", 0),
        "고점대비": item.get("high_distance", 0),
        "VWAP괴리율": item.get("vwap_gap", 0),
        "상대강도": item.get("relative_strength", 0),
        "돌파": bool(item.get("breakout")),
        "추세상승": bool(item.get("trend_up")),
        "기술점수": round(tech, 1),
        "기술투표": evaluation.get("technical_vote", ""),
        "기술이유": str(evaluation.get("technical_reason", ""))[:240],
        "시장점수": round(mkt, 1),
        "시장투표": evaluation.get("market_vote", ""),
        "시장이유": str(evaluation.get("market_reason", ""))[:240],
        "리스크점수": round(risk, 1),
        "리스크투표": evaluation.get("risk_vote", ""),
        "리스크이유": str(evaluation.get("risk_reason", ""))[:240],
        "위원회점수": weighted,
        "위원회판정": decision,
        "신뢰도": round(confidence, 1),
        "플래그": "|".join(flags[:12]),
        "모델": CFG.model,
        "API지연ms": latency_ms,
        "그림자모드": True,
        "입력스냅샷": json.dumps(
            item,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        ),
    }

    _append_csv(LOG_FILE, DECISION_FIELDS, record)

    compact = {
        "time": now_iso,
        "market": market,
        "symbol": item.get("symbol", ""),
        "name": item.get("name", ""),
        "rank": item.get("rank", ""),
        "strategy_score": item.get("strategy_score", 0),
        "technical_score": round(tech, 1),
        "market_score": round(mkt, 1),
        "risk_score": round(risk, 1),
        "committee_score": weighted,
        "decision": decision,
        "confidence": round(confidence, 1),
        "flags": flags[:12],
        "model": CFG.model,
        "latency_ms": latency_ms,
    }

    with _LOCK:
        state = _state()
        recent = list(state.get("recent", []) or [])
        recent.append(compact)
        state["recent"] = recent[-50:]

        last_map = dict(state.get("last_by_symbol", {}) or {})
        last_map[f"{market}:{item.get('symbol', '')}"] = compact
        state["last_by_symbol"] = last_map

        counts = state.setdefault("counts", {})
        counts["total"] = int(counts.get("total", 0) or 0) + 1
        counts[decision] = int(counts.get(decision, 0) or 0) + 1

        state["last_decision_at"] = now_iso
        state["last_error"] = ""
        _save_state(state)

    return compact


def _run_scan_job(
    market: str,
    now_iso: str,
    candidates: list[dict],
    market_context: dict,
    portfolio_context: dict,
) -> None:
    try:
        parsed, latency_ms = _call_openai(
            market=market,
            now_iso=now_iso,
            candidates=candidates,
            market_context=market_context,
            portfolio_context=portfolio_context,
        )

        evaluations = parsed.get("evaluations", []) or []
        by_symbol = {
            str(x.get("symbol", "")).strip().upper(): x
            for x in evaluations
            if isinstance(x, dict)
        }

        for item in candidates:
            symbol = str(item.get("symbol", "")).upper()
            evaluation = by_symbol.get(symbol)
            if not evaluation:
                _record_error(
                    f"{market} {symbol}: AI 응답에 해당 종목 평가가 없습니다."
                )
                continue
            _persist_decision(
                market=market,
                now_iso=now_iso,
                item=item,
                evaluation=evaluation,
                latency_ms=latency_ms,
            )

    except Exception as e:
        _record_error(f"{type(e).__name__}: {e}")


def submit_shadow_scan(
    market: str,
    scan_df: pd.DataFrame,
    now: datetime | None = None,
    portfolio_context: dict | None = None,
) -> dict:
    """
    Worker 메인 루프를 절대 기다리게 하지 않는다.
    가능한 후보가 있을 때만 background executor에 제출한다.
    """
    global _PENDING

    market = str(market or "").upper().strip()
    now = now or datetime.now(KST if market == "KR" else ET)
    now_iso = now.isoformat(timespec="seconds")
    portfolio_context = dict(portfolio_context or {})

    if not CFG.enabled:
        return {
            "accepted": False,
            "reason": "AI_COMMITTEE_ENABLED=false",
        }

    if not CFG.api_key:
        # API 키가 없어도 Worker는 정상 진행.
        with _LOCK:
            state = _state()
            state["last_error"] = "OPENAI_API_KEY 미설정"
            state["last_error_at"] = now_iso
            _save_state(state)
        return {
            "accepted": False,
            "reason": "OPENAI_API_KEY 미설정",
        }

    if scan_df is None or scan_df.empty:
        return {"accepted": False, "reason": "후보 데이터 없음"}

    raw_rows = scan_df.to_dict("records")
    items = [
        _candidate_snapshot(row, i + 1)
        for i, row in enumerate(raw_rows)
    ]
    items = [x for x in items if x.get("symbol")]

    candidates = [
        x for x in items if _is_buy_candidate(x)
    ]
    candidates.sort(
        key=lambda x: (
            int(x.get("rank", 999) or 999),
            -_safe_float(x.get("strategy_score", 0)),
        )
    )
    candidates = candidates[: CFG.max_candidates_per_scan]

    if not candidates:
        return {"accepted": False, "reason": "매수 후보 없음"}

    with _LOCK:
        if _PENDING is not None and not _PENDING.done():
            return {
                "accepted": False,
                "reason": "이전 AI 위원회 평가 실행 중",
            }

        state = _state()
        cache = dict(state.get("cache", {}) or {})
        now_epoch = time.time()
        fresh = []

        for item in candidates:
            key = f"{market}:{item.get('symbol', '')}"
            sig = _signature(market, item)
            prev = cache.get(key, {})
            age = now_epoch - _safe_float(prev.get("at_epoch", 0))
            if (
                prev.get("signature") == sig
                and age <= CFG.cache_ttl_seconds
            ):
                continue

            fresh.append(item)
            cache[key] = {
                "signature": sig,
                "at_epoch": now_epoch,
                "at": now_iso,
            }

        state["cache"] = cache
        _save_state(state)

        if not fresh:
            return {
                "accepted": False,
                "reason": "5분 캐시 내 동일 후보",
            }

        context = _market_context(items)
        _PENDING = _EXECUTOR.submit(
            _run_scan_job,
            market,
            now_iso,
            fresh,
            context,
            portfolio_context,
        )

    return {
        "accepted": True,
        "market": market,
        "candidates": [
            x.get("symbol", "") for x in fresh
        ],
        "shadow_mode": True,
    }


def record_worker_result(
    market: str,
    result: dict | None,
) -> None:
    """
    실제 Worker가 어떤 주문을 냈는지도 별도 기록.
    AI 판정은 여전히 주문에 영향이 없다.
    """
    if not isinstance(result, dict):
        return

    actions = result.get("actions", []) or []
    if not actions:
        return

    market = str(market or "").upper().strip()
    now_iso = str(
        result.get("time")
        or datetime.now(KST).isoformat(timespec="seconds")
    )

    with _LOCK:
        state = _state()
        last_map = dict(state.get("last_by_symbol", {}) or {})

    for action in actions:
        if not isinstance(action, dict):
            continue

        symbol = str(
            action.get("symbol", "")
            or action.get("종목코드", "")
        ).strip().upper()
        if not symbol:
            continue

        ai = last_map.get(f"{market}:{symbol}", {}) or {}

        row = {
            "시간": now_iso,
            "시장": market,
            "종목코드": symbol,
            "종목명": action.get("name", ""),
            "액션": action.get("action", ""),
            "상태": action.get("status", ""),
            "수량": action.get("qty", ""),
            "가격": action.get("price", ""),
            "손익률": action.get("pnl", ""),
            "사유": action.get("reason", ""),
            "최근AI판정": ai.get("decision", ""),
            "최근AI점수": ai.get("committee_score", ""),
            "최근AI시간": ai.get("time", ""),
        }
        _append_csv(
            WORKER_EVENT_FILE,
            WORKER_EVENT_FIELDS,
            row,
        )

        compact_event = {
            "time": now_iso,
            "market": market,
            "symbol": symbol,
            "action": action.get("action", ""),
            "status": action.get("status", ""),
            "ai_decision": ai.get("decision", ""),
            "ai_score": ai.get("committee_score", ""),
        }

        with _LOCK:
            state = _state()
            events = list(state.get("worker_events", []) or [])
            events.append(compact_event)
            state["worker_events"] = events[-50:]
            _save_state(state)


def committee_status() -> dict:
    global _PENDING

    with _LOCK:
        state = _state()
        counts = dict(state.get("counts", {}) or {})
        recent = list(state.get("recent", []) or [])[-12:]
        worker_events = list(
            state.get("worker_events", []) or []
        )[-12:]

    return {
        "ok": True,
        "version": "ai-committee-v1-shadow",
        "enabled": bool(CFG.enabled),
        "configured": bool(CFG.api_key),
        "shadow_mode": True,
        "model": CFG.model,
        "pending": bool(
            _PENDING is not None and not _PENDING.done()
        ),
        "cache_seconds": CFG.cache_ttl_seconds,
        "max_candidates_per_scan": CFG.max_candidates_per_scan,
        "weights": {
            "technical": CFG.technical_weight,
            "market": CFG.market_weight,
            "risk": CFG.risk_weight,
        },
        "thresholds": {
            "approve": CFG.approve_score,
            "hold": CFG.hold_score,
            "risk_veto": CFG.risk_veto_score,
            "risk_reject": CFG.risk_reject_score,
        },
        "counts": counts,
        "last_decision_at": state.get("last_decision_at"),
        "last_error": state.get("last_error", ""),
        "last_error_at": state.get("last_error_at"),
        "recent": recent,
        "worker_events": worker_events,
        "storage": {
            "persistent": not str(STATE_DIR).startswith("/tmp/"),
            "decision_log": "ai_committee_log.csv",
            "worker_event_log": "ai_committee_worker_events.csv",
            "state": "ai_committee_state.json",
        },
    }


def committee_shutdown() -> None:
    try:
        _EXECUTOR.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass

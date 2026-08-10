from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from openai import OpenAI


CACHE_DIR = Path(os.getenv("SONG_TRADER_AI_CACHE_DIR", "/tmp/song_trader_ai"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _secret(secrets, name: str, default: str = "") -> str:
    try:
        value = secrets.get(name, default)
        return str(value) if value is not None else default
    except Exception:
        return os.getenv(name, default)


def _extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("AI 응답에서 JSON 객체를 찾지 못했습니다.")
    return json.loads(m.group(0))


def _candidate_key(
    leader_df: pd.DataFrame,
    market: str,
    strategy_name: str,
    max_candidates: int,
) -> str:
    cols = [
        c for c in [
            "종목코드","종목명","종합점수","판정","RSI",
            "거래량배수","매수점수","매도점수","진입근거"
        ]
        if c in leader_df.columns
    ]
    payload = {
        "market": market,
        "strategy": strategy_name,
        "max_candidates": max_candidates,
        "rows": leader_df[cols].head(max_candidates).to_dict("records"),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_file(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def _load_cache(key: str, ttl_seconds: int) -> Dict[str, Any] | None:
    path = _cache_file(key)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        saved_at = float(data.get("_saved_at", 0))
        if time.time() - saved_at <= ttl_seconds:
            data.pop("_saved_at", None)
            data["_cached"] = True
            return data
    except Exception:
        return None

    return None


def _save_cache(key: str, data: Dict[str, Any]) -> None:
    payload = dict(data)
    payload["_saved_at"] = time.time()
    _cache_file(key).write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def _prefilter_candidates(
    leader_df: pd.DataFrame,
    max_candidates: int = 3,
) -> pd.DataFrame:
    """
    느린 웹검색 전에 후보 수를 줄입니다.

    우선순위:
    1) 판정에 '매수'가 들어간 후보
    2) 종합점수 높은 후보
    3) 그래도 없으면 상위 max_candidates개
    """
    if leader_df is None or leader_df.empty:
        return pd.DataFrame()

    d = leader_df.copy()

    if "판정" in d.columns:
        buy_like = d[
            d["판정"].astype(str).str.contains("매수", na=False)
        ].copy()
    else:
        buy_like = pd.DataFrame()

    if not buy_like.empty:
        if "종합점수" in buy_like.columns:
            buy_like = buy_like.sort_values("종합점수", ascending=False)
        return buy_like.head(max_candidates).reset_index(drop=True)

    if "종합점수" in d.columns:
        d = d.sort_values("종합점수", ascending=False)

    return d.head(max_candidates).reset_index(drop=True)


def analyze_market_with_ai(
    leader_df: pd.DataFrame,
    secrets,
    strategy_name: str = "대장주 추세매매 모드",
    market: str = "국내",
) -> Dict[str, Any]:
    """
    빠른 AI 위험 필터.

    - AI 단독 주문 금지
    - 기술/모멘텀 후보 중 최대 3개만 검사
    - 최근 48시간 공개 뉴스 위주
    - 동일 후보는 15분 캐시
    - 한 번의 웹검색 포함 요청으로 묶어서 판단
    """
    if leader_df is None or leader_df.empty:
        return {"ok": False, "error": "후보 데이터가 없습니다."}

    api_key = _secret(secrets, "OPENAI_API_KEY")
    if not api_key:
        return {
            "ok": False,
            "error": "OPENAI_API_KEY가 없습니다. Streamlit Secrets에 추가해주세요."
        }

    model = _secret(secrets, "OPENAI_MODEL", "gpt-5")
    client = OpenAI(api_key=api_key)

    max_candidates = int(_secret(secrets, "AI_MAX_CANDIDATES", "3") or "3")
    cache_minutes = int(_secret(secrets, "AI_CACHE_MINUTES", "15") or "15")

    filtered = _prefilter_candidates(
        leader_df,
        max_candidates=max(1, min(max_candidates, 5)),
    )

    if filtered.empty:
        return {"ok": False, "error": "AI 검사 대상 후보가 없습니다."}

    cache_key = _candidate_key(
        filtered,
        market=market,
        strategy_name=strategy_name,
        max_candidates=max_candidates,
    )

    cached = _load_cache(cache_key, ttl_seconds=cache_minutes * 60)
    if cached:
        return cached

    cols = [
        "순위","종목코드","종목명","등락률","주도주점수","추세점수",
        "VWAP괴리율","당일고가거리","돌파","눌림재상승","종합점수",
        "판정","진입근거","RSI","거래량배수","매수점수","매도점수"
    ]
    usable = [c for c in cols if c in filtered.columns]
    candidates = filtered[usable].to_dict("records")

    prompt = f"""
당신은 {market} 단기매매용 '뉴스 위험 필터'입니다.
전략명: {strategy_name}

아래 후보들은 이미 기술/모멘텀 규칙으로 선별되었습니다.
당신은 최신 공개 웹 정보만 확인해 '최근 48시간 중심'으로
각 후보의 악재·급변 리스크·재료 신뢰도를 빠르게 점검하세요.

속도 우선 원칙:
- 후보 전체를 한 번에 판단하세요.
- 불필요한 배경 설명은 최소화하세요.
- 오래된 기사보다 최근 48시간 이슈를 우선하세요.
- 확실한 최신 이슈가 없으면 억지로 찾지 말고 CAUTION으로 두세요.
- 매수 추천, 주문, 수량, 가격 지시는 금지합니다.
- AI는 추가 위험 필터일 뿐입니다.

판정 기준:
- ALLOW: 뚜렷한 추가 위험 신호 없음
- CAUTION: 정보 부족/혼재/불확실성
- BLOCK: 명확한 악재·규제·회계·대규모 희석·급변 위험

후보 데이터:
{json.dumps(candidates, ensure_ascii=False)}

반드시 JSON만 출력:
{{
  "market_summary": "최근 시장 맥락 1~3문장",
  "candidates": [
    {{
      "code": "국내는 6자리 종목코드, 미국은 티커",
      "name": "종목명",
      "verdict": "ALLOW|CAUTION|BLOCK",
      "ai_score": 0,
      "confidence": 0,
      "theme": "핵심 이슈",
      "reason": "핵심 이유 1문장",
      "risks": ["위험1", "위험2"],
      "news_quality": "HIGH|MEDIUM|LOW"
    }}
  ]
}}

ai_score와 confidence는 0~100 정수입니다.
"""

    response = client.responses.create(
        model=model,
        tools=[{"type": "web_search"}],
        input=prompt,
        store=False,
    )

    data = _extract_json(response.output_text)
    rows = data.get("candidates", []) or []

    clean_rows: List[Dict[str, Any]] = []
    for r in rows:
        verdict = str(r.get("verdict", "CAUTION")).upper()
        if verdict not in ("ALLOW", "CAUTION", "BLOCK"):
            verdict = "CAUTION"

        try:
            ai_score = max(0, min(100, int(r.get("ai_score", 0))))
        except Exception:
            ai_score = 0

        try:
            confidence = max(0, min(100, int(r.get("confidence", 0))))
        except Exception:
            confidence = 0

        clean_rows.append({
            "종목코드": str(r.get("code", "")).upper(),
            "AI판정": verdict,
            "AI점수": ai_score,
            "AI확신도": confidence,
            "AI테마": str(r.get("theme", "")),
            "AI이유": str(r.get("reason", "")),
            "AI위험": ", ".join([str(x) for x in (r.get("risks", []) or [])]),
            "뉴스품질": str(r.get("news_quality", "")),
        })

    result = {
        "ok": True,
        "model": model,
        "market_summary": str(data.get("market_summary", "")),
        "rows": clean_rows,
        "checked_count": len(filtered),
        "_cached": False,
    }

    _save_cache(cache_key, result)
    return result


def merge_ai_filter(
    leader_df: pd.DataFrame,
    ai_result: Dict[str, Any],
    min_ai_score: int = 60,
    min_confidence: int = 55,
) -> pd.DataFrame:
    out = leader_df.copy()

    if not ai_result or not ai_result.get("ok"):
        out["AI통과"] = False
        out["AI판정"] = "UNAVAILABLE"
        return out

    ai_df = pd.DataFrame(ai_result.get("rows", []))
    if ai_df.empty:
        out["AI통과"] = False
        out["AI판정"] = "UNAVAILABLE"
        return out

    out["종목코드"] = out["종목코드"].astype(str).str.upper()
    ai_df["종목코드"] = ai_df["종목코드"].astype(str).str.upper()

    out["종목코드"] = out["종목코드"].apply(
        lambda x: x.zfill(6) if x.isdigit() else x
    )
    ai_df["종목코드"] = ai_df["종목코드"].apply(
        lambda x: x.zfill(6) if x.isdigit() else x
    )

    out = out.merge(ai_df, how="left", on="종목코드")

    # AI가 실제로 검사한 후보만 통과 가능.
    out["AI판정"] = out["AI판정"].fillna("NOT_CHECKED")
    out["AI점수"] = pd.to_numeric(out["AI점수"], errors="coerce").fillna(0)
    out["AI확신도"] = pd.to_numeric(out["AI확신도"], errors="coerce").fillna(0)

    out["AI통과"] = (
        (out["AI판정"] == "ALLOW")
        & (out["AI점수"] >= min_ai_score)
        & (out["AI확신도"] >= min_confidence)
    )

    return out

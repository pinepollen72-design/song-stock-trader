from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

import pandas as pd
from openai import OpenAI


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

    # Markdown code fence / surrounding prose fallback.
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("AI 응답에서 JSON 객체를 찾지 못했습니다.")
    return json.loads(m.group(0))


def analyze_market_with_ai(
    leader_df: pd.DataFrame,
    secrets,
    strategy_name: str = "대장주 추세매매 모드",
) -> Dict[str, Any]:
    """
    AI는 뉴스/공시/시장 맥락을 요약하고 후보를 '추가 필터'합니다.
    AI 단독으로 주문을 생성하지 않습니다.
    """
    if leader_df is None or leader_df.empty:
        return {"ok": False, "error": "대장주 TOP5 데이터가 없습니다."}

    api_key = _secret(secrets, "OPENAI_API_KEY")
    if not api_key:
        return {
            "ok": False,
            "error": "OPENAI_API_KEY가 없습니다. Streamlit Secrets에 추가해주세요."
        }

    model = _secret(secrets, "OPENAI_MODEL", "gpt-5")
    client = OpenAI(api_key=api_key)

    cols = [
        "순위","종목코드","종목명","등락률","주도주점수","추세점수",
        "VWAP괴리율","당일고가거리","돌파","눌림재상승","종합점수","판정","진입근거"
    ]
    usable = [c for c in cols if c in leader_df.columns]
    candidates = leader_df[usable].head(5).to_dict("records")

    prompt = f"""
당신은 한국 주식 단기매매용 '시장 맥락 위험 필터'입니다.
전략명: {strategy_name}

아래 후보들은 이미 거래대금/거래량/추세 규칙을 통과해 선별된 후보입니다.
당신은 최신 공개 웹 정보를 확인해서 각 종목의 당일/최근 이슈의 질과 위험도를 평가하세요.

중요 원칙:
- 매수 추천을 하지 마세요. 주문을 생성하거나 수량/가격을 지시하지 마세요.
- AI 판단은 규칙 기반 자동매매의 추가 안전 필터일 뿐입니다.
- 단순 루머, 출처 불명, 오래된 기사, 이미 주가에 과도하게 반영된 재료는 감점하세요.
- 거래정지/관리종목/유상증자/대규모 오버행/회계·소송·규제·급격한 변동성 등 위험 신호를 우선 확인하세요.
- 'ALLOW'는 "추가 위험 신호가 뚜렷하지 않음"이라는 뜻일 뿐 매수 추천이 아닙니다.
- 확신이 낮거나 정보가 충돌하면 'CAUTION' 또는 'BLOCK'을 사용하세요.

후보 데이터:
{json.dumps(candidates, ensure_ascii=False)}

반드시 아래 JSON 형식만 출력하세요.
{{
  "market_summary": "오늘 시장/테마 맥락 2~4문장",
  "candidates": [
    {{
      "code": "6자리 종목코드",
      "name": "종목명",
      "verdict": "ALLOW|CAUTION|BLOCK",
      "ai_score": 0,
      "confidence": 0,
      "theme": "핵심 테마/이슈",
      "reason": "핵심 판단 이유 1~2문장",
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
            "종목코드": str(r.get("code", "")).zfill(6),
            "AI판정": verdict,
            "AI점수": ai_score,
            "AI확신도": confidence,
            "AI테마": str(r.get("theme", "")),
            "AI이유": str(r.get("reason", "")),
            "AI위험": ", ".join([str(x) for x in (r.get("risks", []) or [])]),
            "뉴스품질": str(r.get("news_quality", "")),
        })

    return {
        "ok": True,
        "model": model,
        "market_summary": str(data.get("market_summary", "")),
        "rows": clean_rows,
    }


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

    out["종목코드"] = out["종목코드"].astype(str).str.zfill(6)
    out = out.merge(ai_df, how="left", on="종목코드")
    out["AI통과"] = (
        (out["AI판정"] == "ALLOW")
        & (pd.to_numeric(out["AI점수"], errors="coerce").fillna(0) >= min_ai_score)
        & (pd.to_numeric(out["AI확신도"], errors="coerce").fillna(0) >= min_confidence)
    )
    return out

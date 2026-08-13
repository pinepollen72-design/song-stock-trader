from __future__ import annotations

import pandas as pd

from trader_core import discover_domestic_candidates, score_ticker


def build_kr_top5(client, top_n: int = 5) -> pd.DataFrame:
    """한국투자 거래량/거래대금 후보 중 개별주만 골라 기술점수와 합산한다."""
    base = discover_domestic_candidates(client, top_n=25)
    if base is None or base.empty:
        return pd.DataFrame()

    rows = []
    for _, r in base.iterrows():
        code = str(r.get("종목코드", "")).zfill(6)
        if not (len(code) == 6 and code.isdigit()):
            continue
        try:
            tech = score_ticker(code, market="국내") or {}
        except Exception:
            tech = {}

        net = float(tech.get("순점수", 0) or 0)
        tech100 = max(0.0, min(100.0, (net + 6.0) / 12.0 * 100.0))
        lead = float(r.get("주도주점수", 0) or 0)
        combined = lead * 0.55 + tech100 * 0.45

        rows.append({
            "종목코드": code,
            "종목명": str(r.get("종목명", code)),
            "현재가": float(r.get("현재가", 0) or 0),
            "등락률": float(r.get("등락률", 0) or 0),
            "거래대금": int(float(r.get("거래대금", 0) or 0)),
            "누적거래량": int(float(r.get("누적거래량", 0) or 0)),
            "주도주점수": round(lead, 1),
            "기술순점수": round(net, 1),
            "종합점수": round(combined, 1),
            "판정": "🟢 매수검토" if combined >= 50 else "🟡 관망",
        })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values(
        ["종합점수", "주도주점수", "거래대금"],
        ascending=[False, False, False],
    ).head(top_n).reset_index(drop=True)
    labels = ["👑 1위", "🥈 2위", "🥉 3위", "4위", "5위"]
    out.insert(0, "순위", labels[:len(out)])
    return out

from __future__ import annotations

import pandas as pd

from trader_core import score_ticker


def build_us_top5(symbols, top_n: int = 5) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        s = str(symbol).strip().upper()
        if not s:
            continue
        try:
            row = score_ticker(s, market="미국")
        except Exception:
            row = None
        if not row:
            continue
        net = float(row.get("순점수", 0) or 0)
        vol = float(row.get("거래량배수", 0) or 0)
        tech100 = max(0.0, min(100.0, (net + 6.0) / 12.0 * 100.0))
        volume_bonus = max(0.0, min(10.0, vol / 2.0 * 10.0))
        combined = tech100 * 0.90 + volume_bonus
        rows.append({
            "종목코드": s,
            "종목명": s,
            "현재가": float(row.get("현재가", 0) or 0),
            "RSI": float(row.get("RSI", 0) or 0),
            "거래량배수": vol,
            "기술순점수": net,
            "종합점수": round(combined, 1),
            "판정": "🟢 매수검토" if combined >= 50 else "🟡 관망",
        })

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(
        ["종합점수", "거래량배수"], ascending=[False, False]
    ).head(top_n).reset_index(drop=True)
    labels = ["⭐ 1위", "⭐ 2위", "⭐ 3위", "4위", "5위"]
    out.insert(0, "순위", labels[:len(out)])
    return out

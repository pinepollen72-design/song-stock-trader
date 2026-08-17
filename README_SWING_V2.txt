KR SWING V2 — Dual Entry

V1 결과:
- 147일
- signals 5
- 실제 entries 4
- 4전 4패
- STOP 3회, TIME_STOP 1회
=> 전략 자체보다 진입 필터가 지나치게 좁아 표본 부족.

V2에서 바꾸는 것 = 진입만.
손절 / 1% risk sizing / 최대 3종목 / 다음날 시가 체결 /
+3,+5 고정익절 없음 / ATR trailing / time stop은 V1 그대로 유지.

두 진입 통로:
A. PULLBACK_RECOVERY
- 상승추세 Close > SMA20 > SMA60, SMA20 상승
- 최근 눌림 1~12%
- SMA10 위 회복
- 전일 고점 돌파 또는 3일 고점 근처까지 의미 있게 회복
- RISK_OFF에서는 사용하지 않음

B. BREAKOUT_20D
- 상승추세 유지
- 직전 20일 고점 신규 돌파
- 눌림 조건 불필요
- 너무 멀리 추격하는 돌파는 제외
- RISK_OFF에서도 상대강도/거래량이 아주 강하면 1종목까지 허용

완화:
- min_score 58 -> 42
- RS 기준 완화
- 거래량 기준 완화
- market breadth: 0.55/0.40 -> 0.50/0.30
- RISK_OFF 완전 금지 -> 강한 20D breakout만 허용
- 목표 표본: 147일에 대략 20~40 거래를 기대하지만 강제로 맞추지는 않음

꼭 볼 결과:
- net_krw
- trades
- pullback_entries / breakout_entries
- win_rate
- profit_factor
- payoff_ratio
- avg_win / avg_loss
- max_drawdown
- 월별 손익
- 각 trade의 setup_type

업로드:
1) replay_kr_swing_v2_full.py       새 파일
2) replay_kr_d3_hybrid_full.py      기존 파일 덮어쓰기

버전:
kr-swing-v2-dual-entry-fast-v1

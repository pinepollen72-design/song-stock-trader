# 교체용 app.py
# 기존 app.py에서 아래 네 값만 변경하세요.
#
# 국내:
# value=1000000
# value=300000
#
# 미국 분기에서 국내 fallback:
# budget = 1000000
# per_stock = 300000
#
# 아래 코드는 그대로 붙여넣기 쉬운 변경 구간입니다.

# 국내 자금 설정
if market == "국내":
    budget = st.number_input(
        "1일 최대 신규매수 금액(원)",
        min_value=10000,
        value=1000000,
        step=10000,
    )

    per_stock = st.number_input(
        "종목당 최대 금액(원)",
        min_value=10000,
        value=300000,
        step=10000,
    )

    us_daily_budget = 0.0
    us_per_stock_budget = 0.0

else:
    us_daily_budget = st.number_input(
        "미국 1일 최대 신규매수 금액(USD)",
        min_value=0.0,
        value=1500.0,
        step=100.0,
    )

    us_per_stock_budget = st.number_input(
        "미국 종목당 최대 금액(USD)",
        min_value=0.0,
        value=600.0,
        step=50.0,
    )

    budget = 1000000
    per_stock = 300000

# 쏭 자동매매 v1

국내/미국, 모의/실전 공용 구조의 첫 통합 버전입니다.

## 들어있는 기능
- 한국투자 Open API 토큰 자동 발급 및 파일 캐시
- 모의/실전 서버와 키 분리
- 실전 잠금
- 국내 거래대금/거래량 순위 기반 당일 후보 탐색
- 5분봉 RSI/볼린저/거래량/추세 기술점수
- 분할매수 금액 계산
- 국내 현금 매수/매도 API 함수
- 미국 지정가 주문 API 함수
- Streamlit 대시보드(app.py)
- 상시 실행용 워커(worker.py)
- 주문/후보 로그 저장

## 중요한 현재 상태
이 버전은 "통합 골격 + 국내 자동 후보탐지" v1입니다.

`worker.py`는 기본적으로 후보만 찾아 로그를 남기며,
`ENABLE_AUTO_ORDERS=true`를 명시해야 국내 1차 자동주문을 보냅니다.
실전은 추가로 `ALLOW_LIVE_TRADING=true`와 실전 확인문구가 일치해야 합니다.

미국 주문 함수는 포함되어 있지만, 미국 자동주문은 지정가 산정을 위한
실시간 호가/체결조회 검증을 더 한 뒤 켜는 것이 안전합니다.

또한 완전한 당일 자동청산에는 반드시 다음 모듈을 추가 검증해야 합니다.
1. 실제 체결/미체결 조회
2. 보유잔고 동기화
3. 분할매수 상태 저장
4. 익절/손절 분할매도
5. 국내/미국 휴장·단축장 캘린더
6. 마감 전 강제청산 실패 재시도

즉, 파일 안에 실전 주문 함수는 있지만 기본값은 실전 잠금 상태입니다.

## Streamlit Cloud
1. app.py, trader_core.py, requirements.txt를 GitHub 저장소에 올립니다.
2. Streamlit Secrets에 `secrets.toml.example` 내용을 참고해서 키를 입력합니다.
3. 실전 키는 모의운용이 충분히 끝난 뒤 넣는 것을 권장합니다.

## 24시간 실행
Streamlit 대시보드와 별개로 항상 켜진 서버에서:

    pip install -r requirements.txt
    python worker.py

를 실행합니다.

서버에서는 `.env.example` 항목들을 환경변수로 설정하세요.
Streamlit Community Cloud 화면 자체는 24시간 워커 용도로 사용하지 않습니다.

## 기존 secrets와 호환
현재 앱이 아래 키를 쓰고 있다면:
- KIS_APP_KEY
- KIS_APP_SECRET
- KIS_ACCOUNT_NO
- KIS_ACCOUNT_PRODUCT_CODE

새 `Settings`는 모의투자 값에 한해 이 이름들도 fallback으로 읽습니다.

## 권장 진행 순서
1. 대시보드 모의투자 API 연결
2. 국내 후보탐색 확인
3. worker에서 ENABLE_AUTO_ORDERS=false로 하루 로그 수집
4. 모의투자에서 ENABLE_AUTO_ORDERS=true
5. 체결/잔고/청산 모듈 완성 후 여러 거래일 모의운용
6. 실전 소액 검증

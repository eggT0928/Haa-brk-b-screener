# 2026-09-08 시세·휴장 안내 수정

- Firebase Hosting: https://haa-portfolio-260830.web.app/
- Streamlit: https://haa-brk-b-screener-0928.streamlit.app/
- Streamlit main: `403c51e`; Firebase frontend: `e68cfb5` (fix/market-closure).

## 확인된 원인과 변경

Streamlit은 9월 30일에서 DateOffset(months=1)을 빼면 8월 30일이 되어, 8월 31일 월말 인덱스를 찾지 못했다. MonthEnd 오프셋으로 모든 비교 월의 실제 달력상 월말을 찾도록 수정했다. 월초 휴장에도 현재 월을 예상 신호 대상으로 삼고 실제 마지막 가격 날짜를 유지한다. 미국 동부시간 NYSE 달력으로 휴장·장외를 안내하고 그 시간에는 fast_info 조회와 당일 가상 가격 행 생성을 생략한다.

Firebase의 기존 '최저 시세 시각'은 가격 최저점이 아니라 oldestPriceAt(종목별 최신 시세 중 가장 오래된 시각)이었다. 이제 가장 최근 시세 시각과 구성 종목 중 가장 오래된 시세를 별도로 표시한다. NYSE 달력을 이용해 주말·공휴일·장외 시간을 안내한다. 실제 갱신 실패 경고는 유지한다.

브라우저용 NYSE 일정은 backend와 같은 pandas-market-calendars로 생성했다. 범위는 2025~2035년이며, 범위 밖에서는 휴장이라고 단정하지 않고 달력 확인 필요를 표시한다. 임시 휴장이 새로 발표되면 라이브러리 갱신 후 `python scripts/generate_market_calendar.py`로 재생성하고 Hosting을 배포한다. 시계는 미국 동부시간과 서머타임, 조기폐장을 반영하며 화면에서 1분마다 갱신한다.

## 운영 예약 확인

Cloud Scheduler의 실제 ENABLED 작업 두 개를 확인했다. 미국 동부시간 기준:

| 구분 | 정상 거래일 | 평일 휴장일 | 주말 |
| --- | --- | --- | --- |
| 장중 09:30~15:45, 15분 간격 | 26회 | 0회 | 0회 |
| 일일 17:30 | 1회 | 1회(최근 종가 재확인) | 0회 |
| 가격 갱신 작업 합계 | 27회 | 1회 | 0회 |

예약 함수 호출은 평일 09:00/09:15를 포함해 장중 28회+일일 1회이나, 개장 전·휴장일 장중 함수는 외부 시세 조회 전에 종료한다. 조기폐장일은 실제 개장 구간만 장중 조회한다. 각 작업에서 여러 종목·일봉·분봉을 요청하므로 Yahoo HTTP 요청 수 자체가 27회라는 뜻은 아니다. 제공자 내부 재시도, Scheduler 재시도, 수진 수동 조회는 추가될 수 있다. HAA 화면 새로고침은 Firestore 캐시를 읽으며 Yahoo 조회를 추가하지 않는다.

운영 status/daily의 2026-09-07T21:30:35Z 갱신은 성공했다. market/current 및 signals/preview는 미국 9월 4일 16:00 종가(한국 9월 5일 05:00)를 유지하고 있었다. 9월 5~6일 주말 및 9월 7일 노동절 이후 새 정규장 시세가 없는 정상 상태였다.

## 검증·배포

- Streamlit 테스트 14개 통과 및 GitHub main CI 성공.
- Firebase 통합 Python 테스트 118개, 프런트엔드 테스트 67개 통과.
- production 빌드 성공 및 Firebase Hosting 배포 완료. 함수·예약·권한·사용자 포트폴리오는 변경하지 않았다.
- 실제 Firebase 로그인 화면에서 가장 최근 시세 시각, 장외 안내, 9월 예상 신호가 표시되는 것을 확인했다.

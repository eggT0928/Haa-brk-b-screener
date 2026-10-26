# 연금 포트폴리오 통합과 HAA 소수점 수량

작업 브랜치 `feat/pension-fractional`. 기존 운영 후속 `fix/market-closure`의 미국 휴장 안내·시세 시각 수정까지 포함한 상태에서 확장합니다. 사용자 요청의 기존 Firebase 프로젝트 `haa-portfolio-260830`을 사용하며 main 머지는 별도입니다.

## 원본과 유지한 계산

기준 저장소: [eggT0928/magic_pension_portfolio](https://github.com/eggT0928/magic_pension_portfolio), 커밋 `02eabec53f88a51405447dceb4c74373029b144d`, `magic_pension_portfolio.py` blob `787e7955aaef39c535f713eb3352598a4ebd8aa2`.

README의 오래된 종목·비중 설명보다 실행 코드의 `PORTFOLIO_CONFIG`를 기준으로 했습니다. 상품명은 원본 표기이며 종목코드를 식별자로 사용합니다. 운용보수·추천 상품 정보는 새로 표시하지 않습니다.

| 종목코드 | 원본 상품 표기 | 기본 목표 |
|---|---|---|
| 379800 | KODEX 미국 S&P500TR | 24% |
| 294400 | KOSEF 200TR | 8% |
| 283580 | KODEX 차이나CSI300 | 8% |
| 453810 | KODEX 인도 NIFTY50 | 8% |
| 411060 | ACE KRX금현물 | 19% |
| 385560 | RISE KIS 국고채30년 Enhanced | 14% |
| 308620 | KODEX 미국채 10년선물 | 7% |
| 453850 | ACE 미국30년 국채액티브(H) | 7% |
| 449170 | TIGER KOFR금리액티브(합성) | 5% |

총자산·원금은 KRW, 보유·최종 목표는 국내 ETF 정수 수량입니다. `목표 배분액 = 총자산 × 조정 비중`, `자동 목표 = floor(목표 배분액 / 현재 가격)`, `매매수량 = 최종 목표 − 보유수량`입니다. 예수금을 포함한 총자산을 직접 입력하고 종목별 목표 비중을 조정할 수 있습니다. 합계 100%가 아니거나 최종 목표금액이 총자산을 초과하면 계산안 저장을 막습니다. 총자산이 입력한 ETF 평가액보다 작은 경우도 확인을 요구합니다.

‘목표 보유수량’은 추가 매수량이 아니라 매매 후 보유할 **전체 수량**입니다. 빈칸(null)은 자동 목표, 직접 입력한 0은 명시적 목표 0으로 구분합니다. 직접 입력한 수량은 시세나 총자산 변경 후에도 유지합니다. 기본 비중 복원과 자동 목표로 복원 버튼을 제공합니다.

원본에서 수정한 부분:

- 세션의 `purchase_quantities=0` 초기값과 매번 재계산된 편집 기본값이 충돌하던 부분을 nullable 수동 목표로 구분했습니다.
- 원본은 화면의 총자산에 예수금을 포함하면서 손익에서는 제외했습니다. 통합 앱은 `손익 = 입력 총자산 − 원금`, `수익률 = 총자산/원금 − 1`로 통일하고 ETF 평가액을 별도로 표시합니다. 원금 0은 미입력으로 표시합니다.
- 조회 실패를 0원으로 대입하지 않습니다. 9개 정상 가격과 시각이 없으면 계산을 중단합니다.
- 원본 저장 버튼은 세션만 유지했지만 통합 앱은 보유·총자산·원금·비중·직접 목표를 Firestore에 실제 저장합니다. 과거 Streamlit 세션은 자동 이관할 수 없으므로 최초 한 번 입력·저장이 필요합니다.

연금 원본에는 장기 백테스트가 없어 신규 연금 백테스트를 임의로 추가하지 않았습니다. 기존 현근·수진 백테스트와 전략은 유지합니다.

## 화면·가족 공유·저장

상단에서 현근 / 수진 / 연금 탭을 전환합니다. 방문한 탭은 유지되어 미저장 입력이 사라지지 않으며, 로그인 계정·공유 공간 변경 시는 초기화합니다. 모바일에서는 종목별 입력 카드를 사용하고 계산표만 내부 스크롤합니다.

기존 가족 공유 승인 `access/{callerUid}.ownerUid`를 동일하게 적용합니다. 관리자와 구성원이 세 포트폴리오를 함께 편집합니다. 프로필 저장이나 계산안 저장은 실제 주문 또는 보유수량 자동 변경을 수행하지 않습니다.

| 경로 | 내용 |
|---|---|
| `users/{ownerUid}` | 기존 HAA 보유수량·설정, 선택 quantityMode 추가 |
| `users/{ownerUid}/rebalances/*` | 기존 HAA 계산안, 선택 quantityMode 추가 |
| `users/{ownerUid}/portfolios/pension` | 연금 strategy/currency/totalBalance/principal/holdings/weights/targetOverrides/updatedAt |
| `users/{ownerUid}/portfolios/pension/rebalances/*` | 연금 계산안, 읽기·추가만, 최근 24건 표시 |
| `market/pension` / `status/pensionQuotes` | 공용 원화 가격·종목별 시각·조회 상태 |
| `internal/pensionQuotesLease` | 연금 수동 조회 잠금·재시도 제한, 서버만 접근 |

연금 계산안은 strategy=`magic-pension-v1`, currency=`KRW`, month, equity, principal, remainingCash, weights, 가격 시각 2개를 저장합니다. `lines`는 위 표의 고정 9종목 순서로 `{held, price, target}`만 보존하고, 표시 종목과 `trade=target-held`는 이 고정 계약으로 복원합니다. 중복 필드를 줄여 Firestore 규칙 표현식 한도 안에서 모든 행의 정수 수량·유효 가격과 9개 비중의 범위·합계를 검사합니다. 같은 전략 버전의 종목 순서는 변경하지 않습니다.

규칙은 기본 거부, 공유 승인 범위, 생성·갱신 스키마 검증, 이력 불변성을 적용하고 에뮬레이터로 검증합니다. 가족 외 공개 범위를 넓힐 때는 접근 정책과 규칙을 다시 검토해야 하는 초기 규칙입니다.

## 시세와 비용

`POST /api/pension/refresh`는 기존 API 함수에 추가합니다. Firebase ID 토큰·승인·선택 App Check를 적용하고 사용자가 임의의 조회 종목이나 다른 사용자 UID를 지정할 수 없습니다.

- Yahoo `.KS` 원시가격과 KRX 달력을 사용합니다. 미국 NYSE 일정으로 한국 휴장을 추정하지 않습니다.
- 한국 장중에는 실제 시각이 확인된 최근 25분 이내 분봉을 우선 사용합니다. 장중 응답을 사용할 수 없으면 최근 완료된 종가를 `quoteMode=close`로 구분해 표시합니다.
- 종가는 KRX 폐장 후 60분 경과와 해당 일봉 존재를 모두 확인합니다. 당일 종가가 완료되지 않았으면 최근 완료 종가를 표시합니다.
- 장중 성공 캐시는 10분, 장외에는 같은 마지막 완료 종가를 재사용합니다. 요청 종류별 60초 재시도 간격·2분 잠금·Yahoo 각 조회 최대 2회 재시도로 연속 호출을 제한합니다.
- 조회 실패 시 가격 문서와 원래 성공·가격 시각을 유지하고 상태만 실패로 기록합니다. 첫 성공값이 없으면 오류를 안내합니다.
- 연금 탭 방문·입력 변경만으로 시세를 새로 조회하지 않습니다. **추가 Scheduler와 함수는 없으며** 기존 미국 HAA·수진 자동 갱신 주기도 유지합니다. 연금은 버튼 갱신을 사용합니다.

추가 요청의 Functions 실행·Firestore 읽기/쓰기 사용량은 발생할 수 있습니다. 전용 메일 서비스나 새 상시 서버는 추가하지 않습니다.

## HAA 정수·소수점

현근 보유자산의 **HAA 수량 단위**에서 정수 1주 / 소수점 0.000001주를 선택합니다. 소수점 목표도 반올림하지 않고 6자리 내림하여 자산별 목표금액을 초과하지 않습니다. 기존 소수점 보유·매매수량은 최대 8자리로 표시합니다. 소수점 매매 가능 여부와 최소 단위는 증권사에서 확인합니다.

선택은 프로필과 계산안 이력의 `quantityMode`에 저장합니다. 이전 문서·이력에 해당 필드가 없으면 종전 정수 방식으로 해석합니다. 기존 보유량을 자동으로 반올림하거나 변경하지 않습니다. 신호·자산선택·SPY/SPYM 실행 구분·SPY 백테스트·DBC 프록시·^IRX는 그대로입니다.

## 검증·배포

```bash
pytest -q
ruff check functions/haa functions/sujin functions/pension functions/main.py tests/test_firebase*.py
npm test
npm run test:rules
npx firebase emulators:exec --project demo-haa --only firestore "python scripts/check_emulators.py"
npm run build
# 운영 변경: 사용자가 요청한 기존 전용 프로젝트에 규칙·API·Hosting만 배포
npx firebase deploy --only firestore:rules,functions:haa:api,hosting --project haa-portfolio-260830
```

신규 환경변수·결제 연결·예약 작업은 필요하지 않습니다. 운영 빌드는 `VITE_DEMO_MODE=false`, `VITE_USE_EMULATORS=false`로 확인합니다. 새 브랜치에서 검증 후 배포하며 GitHub Actions는 검증만 수행합니다. main 머지는 배포와 별도입니다. 롤백 시 앞선 규칙·API·Hosting을 함께 재배포하며 포트폴리오 문서를 삭제하지 않습니다.

2026-10-26 실제 Yahoo 읽기 검증에서 9개 ETF의 해당 KRX 종가와 시각을 정상 확인했습니다. 현근·수진 실제 보유량은 테스트 값으로 수정하지 않습니다. 검증 개수·운영 확인 결과는 배포 후 아래에 기록합니다.

배포 전 로컬 검증: Python 140개, 화면·계산 93개, Firestore 규칙 23개(합계 256개) 통과. 실제 Firestore 에뮬레이터에서 연금 잠금·캐시·실패 시각 보존과 기존 미국 문서 분리를 검증했습니다. 별도 로컬 샘플 화면에서 폭 390px/1280px의 가로 넘침 없음, 현근·연금 전환 후 미저장 입력·수량 단위 보존, 콘솔 오류 없음도 확인했습니다.

### 운영 배포 확인 (2026-10-26)

- 구현 커밋 `891fcf5`의 API·규칙·Hosting 배포 완료. [운영 웹앱](https://haa-portfolio-260830.web.app/?v=891fcf5), [GitHub Actions 전체 성공](https://github.com/eggT0928/Haa-brk-b-screener/actions/runs/40298947629). main 미병합.
- 관리자 기존 Google 로그인 세션에서 현근·수진·연금 탭과 HAA 수량 선택을 확인했습니다. HAA 소수점 목표·잔여 현금 표시가 실제 보유량으로 바뀌는지 화면에서 확인했으며, 설정 저장은 누르지 않았습니다.
- 운영 연금 시세 버튼으로 실제 API→Yahoo→Firestore 성공. 한국시간 **18:27:09** 요청에서 9종목의 **2026-10-26 15:30 KST 종가**를 모두 받았고 `status/pensionQuotes.ok=true`, 종가 기준 표시를 확인했습니다. 로컬 성공만으로 운영 성공을 추정하지 않았습니다.
- 배포 전후 현근 보유량 문서(최종 수정 2026-10-16), 수진 문서(2026-08-31), 확정 신호(2026-09-30)의 갱신시각·내용 지문이 일치했습니다. 연금 사용자 프로필은 임의 생성하지 않았습니다. 실제 연금 보유량 입력·저장과 운영 계산안 이력 저장은 사용자가 첫 입력 후 진행합니다. 해당 저장 경로 자체는 규칙 에뮬레이터와 화면 테스트에서 검증했습니다.
- 기존 Scheduler 2개의 이름·일정·활성 상태를 유지했습니다. 비로그인 HAA/수진/연금 API 요청은 401이며, Hosting은 새 제목·새 자산 파일을 제공합니다.
- Firebase 연결 도구가 Hosting 대상 이름을 찾지 못해 배포를 시작하지 못했으며, 같은 프로젝트를 명시한 기존 Firebase CLI 배포로 완료했습니다. 새 프로젝트·결제·권한·함수를 생성한 우회는 하지 않았습니다.

# 부부 리밸런서 · HAA / 수진 포트폴리오 Firebase 웹앱

기존 Streamlit `haa_brk-b_screener_web.py`를 유지하면서 Firebase 웹앱을 추가했습니다.
기준 원본: `main` 커밋 `7f4b4cac1070eef23cbcfd1369cd83c632b3d05b`.
작업 브랜치: `feat/firebase-haa-webapp`. 자동 배포·자동 머지는 없습니다.
차트·모바일 후속 개선 브랜치: `feat/haa-interactive-charts`.
수진 통합 브랜치: `feat/couple-rebalancer` — **2026-08-30 사용자 승인 후 운영 배포 완료, main 미병합**입니다.
기존 자동 갱신 2개를 함께 사용하고 수동 갱신은 보조로 제공합니다. 별도 Scheduler는 추가하지 않습니다.
전략 보존 범위·분리 저장·추가 검증·배포 체크리스트는 [수진 통합 안내](docs/SUJIN_INTEGRATION.md)를 참고하세요.
가족 권한 후속 브랜치: `feat/family-access`. 앱 안에서 이메일 초대·승인·해제를 관리하고 두 Google 계정이 같은 포트폴리오를 사용합니다. [초대 사용법·초기 관리자 설정·보안](docs/FAMILY_ACCESS.md)을 참고하세요.

2026-08-30 사용자 승인 후 별도 Firebase 프로젝트 `haa-portfolio-260830`에 배포했습니다.
웹 주소: <https://haa-portfolio-260830.web.app/>. 기존 Streamlit 앱·다른 Firebase 프로젝트와 `main`은 변경하지 않았습니다.
수진 공유 갱신·수동 갱신·운영 백테스트를 확인했습니다. 수진의 실제 보유수량은 첫 사용 시 입력하세요.
이전 화면이 남아 있으면 강력 새로고침(Ctrl+F5)을 하세요. 이후 배포부터 즉시 재확인되도록 루트 HTML에 `no-cache`를 적용했습니다.
실제 확인 범위와 남은 운영 점검은 [검증 기록](docs/VALIDATION.md)을 참고하세요.

### 백테스트 그래프 사용법

- 평가금액·낙폭 그래프에 마우스를 올리면 해당 월말 날짜, USD 평가금액, 낙폭이 표시됩니다.
- 모바일에서는 그래프를 터치하거나 좌우로 움직여 월을 선택합니다. 선택 수치는 그래프 위에 고정 표시되며 페이지의 위아래 스크롤도 유지합니다.
- X축은 `YYYY.MM` 형식으로 중간 월도 표시하며, 기간과 화면 폭에 맞춰 눈금 개수를 자동 조절합니다. 생략되는 것은 날짜 눈금뿐이며 월별 관측값은 모두 유지합니다.
- 아래 탐색 막대와 이전·다음 버튼으로 모든 월을 확인할 수 있습니다. 키보드는 방향키로 한 달, Home/End로 처음/끝, PageUp/PageDown으로 12개월씩 이동합니다.
- 차트 탐색은 이미 받은 결과를 화면에서만 사용하므로 추가 서버 호출이나 저장이 발생하지 않습니다. 전략·SPY 기준·DBC 프록시·^IRX 계산은 변경하지 않습니다.

## 1. 전략과 원본 재사용 범위

- **HAA 80% + BRK-B 20%**, HAA 안에서 균등 배분합니다.
- 모멘텀은 월말 수정주가의 **1·3·6·12개월 수익률 단순 평균(13612U)** 입니다.
- TIP > 0: SPY, VEA, VWO, IWM, TLT, PDBC, VNQ, IEF 중 양수 점수 상위 최대 4개.
- TIP ≤ 0: IEF/BIL 중 점수가 높은 1개. 동점은 기존 목록 순서를 유지합니다.
- 양수 공격자산이 4개 미만이면 해당 개수로 80%를 나눕니다. 0개이면 HAA 몫은 현금(무이자)입니다.
- 실전만 SPY/SPYM 선택 가능. 신호·랭킹·백테스트는 언제나 SPY. 실제 수량은 선택 ETF의 **원시 현재가격**으로 계산합니다. SPYM 시세가 없을 때 SPY 가격으로 대신 계산하지 않습니다.
- **DBC 프록시 주의:** 현재 원본은 PDBC 상장 이전만 연결하는 방식이 아니라 **백테스트 전 기간에 DBC를 PDBC 슬롯으로 사용**합니다. 같은 동작을 유지합니다. 실전 신호는 PDBC입니다.
- 무위험수익률은 선택된 백테스트 기간의 일별 `^IRX Close` 평균 / 100. 샤프는 기존 `(CAGR − 무위험수익률) / 연환산 변동성`을 유지합니다. IRX가 없으면 임의 2.5%를 넣지 않고 미표시·경고합니다.
- 기존 순수 계산을 `functions/haa/core.py`로 이식했고, 원본 함수와의 모멘텀·자산 선택·프록시 결과 대조 테스트를 포함했습니다. Streamlit 원본과 기존 테스트는 변경하지 않았습니다.

### 신호 시점과 수정한 경계 처리

**확정:** NYSE 달력의 그 달 마지막 거래일 종가. 마지막 거래일이 주말 월말보다 앞서도 정상 반영합니다. 최소 폐장 60분 이후에만 확정 후보가 되고, 실제 저장은 일일 작업의 미국 동부 17:30입니다. 최종 거래일의 모든 필수 종목 일봉과 12개월 비교가격이 있어야 합니다. 확정 월별 문서는 이후 일일 가격 수정으로 덮어쓰지 않습니다.

**예상:** 미국 동부시간의 당월 가상 월말에 Yahoo 정규장 최신 1분봉 Close를 넣습니다. 같은 갱신에서 받은 최근 2년 수정주가로 정확히 1·3·6·12개월 전 달력 월말과 비교하고, TIP 판정·공격/방어·선택 자산까지 전부 재계산합니다. 21/63/126/252 거래일 근사를 사용하지 않습니다. Yahoo 데이터는 지연될 수 있으며 체결 틱이 아닙니다. 25분 이상 된 필수 시세는 거부합니다.

월초 휴장일에 전월 가격으로 새로운 달 예상 신호를 만들어내지 않습니다. 화면에 이전 예상본이 남아 있으면 카드의 **월·시세 기준시각**을 확인하세요. 현재 일봉에는 앞으로 발생할 배당을 가정하지 않습니다. 과거 수정주가를 매 갱신 재조회하여 이미 발생한 배당·분할을 반영합니다.

원본에서 발견한 두 경계 문제는 신규 엔진에서 수정했습니다.

1. 4월 30일에서 한 달을 빼면 3월 30일을 찾던 문제: 달력 월 단위 연산으로 3월 31일을 찾습니다.
2. 첫 손실 월이 최대 낙폭에서 빠지던 문제: 초기 원금을 누적 최고값에 포함합니다. 따라서 이 경계에 해당하는 MDD는 원본과 달라질 수 있습니다.

## 2. 아키텍처와 비용

| 구성 | 역할 | 선택 이유 |
|---|---|---|
| Firebase Hosting | React/Vite 정적 웹, `/api/**` 함수 연결 | 정적 파일은 서버 실행 없이 제공 |
| Firebase Authentication | Google 로그인 | 기기 간 사용자 식별 |
| Firestore | 개인 설정·보유수량, 공용 신호·시세, 이력, 압축 장기 데이터 | 방문자마다 Yahoo를 조회하지 않음 |
| Python Cloud Functions 2세대 | 공유 예약 함수 2개 + 백테스트·수동 갱신 HTTP 함수 1개 | 기존 Python/pandas 재사용, 별도 Docker 관리 불필요 |
| Cloud Scheduler | 장중·일일 실행 | Firebase 배포가 예약 작업과 IAM 연결을 관리 |

별도 Cloud Run 컨테이너도 가능하지만 Docker 이미지, 배포 명령, Scheduler 인증 연결을 따로 관리해야 합니다. Functions 2세대도 Cloud Run 기반이므로 이 규모에서는 Firebase CLI 한 흐름이 단순합니다. 프런트엔드용 상시 서버는 없습니다.

예약 작업은 다음 **2개**입니다.

- `refresh_intraday`: 동부시간 평일 09~15시 매 15분. 09:00/09:15, 휴일·조기폐장 이후는 NYSE 달력 검사 후 바로 종료합니다. 보통 실제 장중 조회는 09:30~15:45 하루 26회입니다.
- `refresh_daily`: 동부시간 평일 17:30. 2000년부터의 장기 수정주가·IRX를 한 번 갱신하고, 새로 완료된 월이 있으면 확정 신호를 저장합니다. 월말 장애 시 다음 평일에도 다시 확정을 시도합니다. 일일 데이터 캐시와 월말 확정을 한 작업으로 결합했습니다.

두 작업은 하나의 Firestore 임대 잠금을 공유합니다. 같은 15분 실행 구간은 중복 발행하지 않습니다. Yahoo 조회는 최대 3회(1·2초 대기), Scheduler 추가 재시도는 장중 1회·일일 2회로 제한했습니다. 실패하면 기존 성공 문서·시각을 유지하고 `status`만 실패로 기록합니다. 초기 성공값이 아예 없으면 데이터를 꾸며내지 않고 대기 상태를 표시합니다.

**Blaze 결제 연결이 필요하며 완전 무료를 보장하지 않습니다.** Functions는 최소 인스턴스 0, 최대 1, 512MiB, `gcf_gen1` CPU, 동시성 1입니다. Scheduler는 무료 제공량이 남아 있으면 2개 작업을 그 안에서 운영할 수 있습니다. 공식 안내상 Scheduler 작업은 월 $0.10/개이며 Google 계정에 3개 무료 제공량이 있습니다. 실제 공유 무료량과 과금은 콘솔에서 확인하세요. [Firebase 예약 함수 안내](https://firebase.google.com/docs/functions/schedule-functions)

사용자별 반복 Yahoo 요청·백테스트 결과 저장을 하지 않고, 공용 압축 데이터에서 요청 기간만 계산합니다. Firestore 읽기/쓰기, Functions 실행·네트워크, Cloud Build, Artifact Registry 저장, Hosting 전송은 별도 과금 가능성이 있습니다. 예산 알림은 **지출의 강제 상한이 아닙니다**. 소수 승인 계정용이며 공개 대규모 서비스라면 요청 제한·비동기 백테스트 작업을 추가하세요. 최대 인스턴스 1도 모든 종류의 지출 상한은 아닙니다.

관련 공식 문서: [Functions 리소스 설정](https://firebase.google.com/docs/functions/manage-functions), [Hosting 함수 연결 및 60초 제한](https://firebase.google.com/docs/hosting/functions), [Yahoo 다운로드 파라미터](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html).

## 3. 파일 구조

```text
haa_brk-b_screener_web.py   기존 Streamlit 앱 (보존)
frontend/                 한국어 React UI, 수량 계산, UI 테스트
functions/main.py         공유 예약 함수 2개, 인증된 백테스트·수동 갱신 API
functions/sujin/          수진 고정비중 백테스트·공통 시세 재사용·독립 캐시
functions/haa/core.py      13612U, 선택, 실전/백테스트 분리
functions/haa/calendar.py  NYSE 휴장·조기폐장·서머타임 경계
functions/haa/market.py    yfinance 조회, 데이터 검증, 재시도
functions/haa/store.py     Firestore 잠금, 압축 캐시, 원자적 발행
functions/haa/service.py   공용 갱신 및 캐시 기반 백테스트
functions/haa/access.py    Google 이메일 초대·승인·가족 관리자 API
firestore.rules           승인 계정의 서버 지정 공유 공간만 허용
firebase.json             Hosting, Python 런타임, 에뮬레이터
.firebaserc.example       배포 프로젝트 선택 예시
tests/                    원본 및 신규 엔진/API/장애복구/규칙 테스트
scripts/check_yahoo.py     Firestore 쓰기 없는 실데이터 점검
.github/workflows/        기존 CI + 신규 Firebase 검증 (배포 없음)
docs/VALIDATION.md         확인 결과 및 배포 전 점검 목록
```

Firestore 데이터 모델:

| 경로 | 내용 | 클라이언트 권한 |
|---|---|---|
| `access/{uid}` | enabled, role, ownerUid, email | 본인 상태 읽기만, 서버 쓰기 |
| `users/{ownerUid}` | `holdings`, USD `cash`, `sp500`, 백테스트 기간, 갱신시각 | 승인된 공유 구성원 읽기·쓰기 |
| `users/{ownerUid}/rebalances/{id}` | 사용자가 저장한 계산안 | 공유 구성원 읽기·추가만, 변경·삭제 금지 |
| `signals/confirmed`, `signals/preview` | 분리된 공용 신호 | 승인 사용자 읽기만 |
| `signalHistory/{YYYY-MM}` | 운영 이후 월별 확정본 | 승인 사용자 읽기만 |
| `market/current` | 원시 시세와 종목별 시각 | 승인 사용자 읽기만 |
| `status/intraday`, `status/daily` | 최근 작업 성공/실패 | 승인 사용자 읽기만 |
| `internal/history` | 압축 수정주가와 일별 IRX, 기준일 | 서버 전용 |
| `internal/refreshLease` | 중복 실행 방지 임대 잠금 | 서버 전용 |

압축 데이터가 900KB를 넘으면 갱신을 중단합니다(Firestore 문서 1MiB 한도 대비 여유). 이 경우 데이터 분할 저장을 추가해야 합니다. 월별 이력은 운영 시작 이후의 확정본이며, 운영 전 과거 신호를 당시에 저장한 것처럼 만들지 않습니다. 화면 이력은 최근 24건, 전체 장기 시뮬레이션 이력은 백테스트 결과에서 확인합니다.

## 4. 로컬 설치·검증

필요: Python **3.12**, Node.js **22 이상**, Java **21 이상**(Firestore 에뮬레이터), Git.

저장소 루트에서:

```bash
python3.12 -m venv functions/venv
source functions/venv/bin/activate
pip install -r requirements-dev.txt
npm ci
pytest -q
ruff check functions/haa functions/sujin functions/main.py tests/test_firebase*.py
npm test
npm run build
npm run test:rules
```

Windows PowerShell은 가상환경 생성·활성화 부분을 다음으로 바꿉니다.

```powershell
py -3.12 -m venv functions/venv
.\functions\venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

활성화가 제한되면 `functions/venv/Scripts/python.exe`를 직접 사용하세요. CLI 실행 시 `npm.cmd` / `npx.cmd`를 사용할 수 있습니다.

### Firebase 없이 화면만 검토

`frontend/.env.example`을 `frontend/.env.local`로 복사하고 `VITE_DEMO_MODE=true`로 설정한 뒤:

```bash
npm run dev
```

이 모드는 **가상 샘플임을 화면에 명시**하며 실제 로그인·저장·서버 백테스트를 실행하지 않습니다. 배포 빌드에는 반드시 `VITE_DEMO_MODE=false`를 사용하세요. Firebase 설정이 빠진 일반 모드는 오류 안내를 보여주며 샘플을 실제 신호로 대신 표시하지 않습니다.

### 전체 로컬 Firebase 에뮬레이터

`frontend/.env.local`에 예를 들어 다음을 지정합니다. 이 파일은 Git에서 제외됩니다.

```dotenv
VITE_FIREBASE_API_KEY=demo-key
VITE_FIREBASE_AUTH_DOMAIN=demo-haa.firebaseapp.com
VITE_FIREBASE_PROJECT_ID=demo-haa
VITE_FIREBASE_APP_ID=demo-app
VITE_USE_EMULATORS=true
VITE_DEMO_MODE=false
```

Python 가상환경이 활성화된 루트 터미널에서:

```bash
npm run build
npx firebase emulators:start --project demo-haa --only auth,firestore,functions,hosting
```

Hosting `http://127.0.0.1:5000` 또는 별도 `npm run dev`를 엽니다. Google 로그인 팝업은 Auth 에뮬레이터의 테스트 계정 선택 화면입니다. 에뮬레이터 UI에서 `internal/family`의 ownerUid·ownerEmail과 `access/{uid}`의 enabled·role·ownerUid·email을 가상 관리자 계정에 맞춰 설정합니다([필드 안내](docs/FAMILY_ACCESS.md)). 실제 계정과 에뮬레이터를 혼용하지 않도록 코드가 `demo-` 프로젝트를 강제합니다.

에뮬레이터를 종료한 뒤 아래 명령으로 가상 Yahoo 응답을 사용한 로컬 통합 검증도 실행할 수 있습니다. 실제 Google 자격증명 없이 `demo-haa` 로컬 저장·인증만 테스트합니다.

```bash
npx firebase emulators:exec --project demo-haa --only firestore,auth,functions "python scripts/check_emulators.py"
```

예약 트리거의 시간 흐름을 에뮬레이터가 자동 재현하지는 않습니다. `pytest`에서 시간과 Yahoo 응답을 주입하여 경계를 검증합니다. 실제 Yahoo 읽기만 확인하려면:

```bash
python scripts/check_yahoo.py
# 과거 시점 기준 재현 점검 (실시간 검증이 아님)
python scripts/check_yahoo.py --as-of 2026-08-28T15:00:00Z
```

이 명령은 Firebase에 쓰지 않습니다. 비어 있는 필수 일봉이나 누락 종목이 있으면 실패하는 것이 정상적인 보호 동작입니다. CI는 Yahoo 네트워크에 의존하지 않습니다.

## 5. 실제 Firebase 배포 절차 — 사용자 승인 후 실행

아래는 새 환경에 재현하거나 변경사항을 재배포할 때의 절차입니다. 현재 전용 프로젝트 생성·Blaze 연결·첫 배포는 사용자 승인 후 완료했습니다. main 머지는 별도 승인이 필요합니다.

1. Firebase 콘솔에서 별도 테스트/운영 프로젝트를 만들거나 선택하고 **Blaze 결제·예산 알림**을 설정합니다. Firestore는 기본 `(default)` DB를 프로덕션 모드로 생성합니다. Functions와 데이터 위치를 맞추려면 `us-central1` 지역을 선택하세요. 기존 DB 위치는 변경되지 않습니다.
2. Authentication → Sign-in method에서 **Google**을 활성화합니다. 웹앱을 등록해 Firebase 웹 설정을 복사합니다. 배포 Hosting 도메인과 개발용 `localhost`가 승인 도메인인지 확인합니다.
3. `frontend/.env.example` → `frontend/.env.local`로 복사하고 네 가지 Firebase 웹 설정값을 채웁니다. `VITE_USE_EMULATORS=false`, `VITE_DEMO_MODE=false`를 확인합니다. 웹 API 키는 공개 설정이며 서버 비밀키가 아닙니다. 서비스 계정 JSON을 프런트엔드나 Git에 넣지 마세요.
4. `functions/.env.example` → `functions/.env`로 복사합니다. 초기 개인용은 `ENFORCE_APP_CHECK=false`. 공개 범위를 넓힐 때는 Firebase App Check에 웹 앱을 등록하고 `VITE_RECAPTCHA_SITE_KEY`를 채운 뒤 서버 검증을 활성화하세요. Firestore의 App Check 강제 적용은 콘솔에서 별도로 설정합니다. App Check는 사용자 승인 규칙을 대체하지 않습니다.
5. Python 가상환경을 준비하고 아래를 실행합니다. 예시 프로젝트 ID는 반드시 본인 것으로 바꾸세요. 기존 `firebase.json`을 `firebase init`으로 덮어쓸 필요가 없습니다.

```bash
npx firebase login
npx firebase use --add
# 또는 .firebaserc.example을 .firebaserc로 복사하고 프로젝트 ID 입력
pip install -r functions/requirements.txt
npm ci
npm test
pytest -q
npm run test:rules
npm run build

# 아래 명령부터 실제 리소스·비용이 발생할 수 있음: 검토 승인 후만 실행
npx firebase deploy --only firestore:rules,firestore:indexes,functions,hosting --project YOUR_PROJECT_ID
```

6. Cloud Scheduler / Cloud Functions API가 활성화되었는지 확인합니다. Firebase CLI가 예약 작업과 호출 권한을 생성합니다. 예약 함수에 직접 공개 호출 권한을 추가하지 마세요. HTTP `api`는 Hosting 연결을 위해 공개 진입점이지만 모든 요청에서 ID 토큰·계정 승인 여부를 확인합니다.
7. Cloud Scheduler 콘솔에서 `firebase-schedule-refresh_daily-us-central1`의 **지금 실행**으로 초기 데이터를 만듭니다. 이름은 배포 로그와 콘솔에서 확인하세요. `status/daily.ok`, `internal/history`, `signals/confirmed`, `market/current`를 확인합니다. Yahoo 실패 시 로그에 누락 일자/티커가 남으며 빈 캐시를 임의 값으로 채우지 않습니다. 문제 해결 후 예약 작업을 다시 실행합니다.
8. 관리자 본인이 Hosting에서 Google 로그인한 뒤 최초 한 번만 [가족 관리자 초기 설정](docs/FAMILY_ACCESS.md)을 수행합니다. 이후 앱의 가족 관리에서 이메일 초대·승인·해제를 처리합니다. 일반 사용자는 자신이나 다른 사람의 승인을 변경할 수 없습니다. Firebase IAM 소유자 초대는 앱 이용에 필요하지 않습니다.
9. 보유수량 저장→재로그인 복원, SPY/SPYM 수량 차이, 백테스트 최대/사용자 기간, 확정·예상·시세 시각, 이력 저장을 확인합니다. 신규 일일 갱신의 가격이 증권사/Yahoo 화면과 맞는지 확인한 뒤 실전에 사용하세요.
10. 비용 알림과 함수 오류 로그를 확인합니다. Artifact Registry 이미지 정리 정책도 확인하세요. Scheduler 설정을 변경할 때는 `functions/main.py`를 수정해 재배포합니다. 콘솔에서 임의 수정한 일정은 다음 배포에서 되돌아갈 수 있습니다.

### 장애·운영 주의사항

- `Yahoo`는 비공식 접근 경로이며 가용성·지연·상업적 사용 권한을 보장하지 않습니다. 개인용을 전제로 하며 공개 서비스 전에는 이용조건과 데이터 라이선스를 별도 확인하세요.
- 서비스 계정은 Admin SDK로 규칙을 우회합니다. 서버 런타임 계정의 IAM 권한도 검토하고 클라이언트에 노출하지 마세요.
- API는 캐시만 읽습니다. 인증된 사용자가 페이지를 새로고침한다고 Yahoo 갱신이 실행되지는 않습니다.
- 당일 데이터 실패 시 직전 성공값을 유지합니다. 성공값의 나이가 오래되면 거래 판단을 중지하고 데이터부터 확인하세요. 화면은 30분 이상 된 실행 시세로 계산안을 저장할 때 확인 체크를 요구합니다(장외도 포함).
- 지연 시세, 체결가격 차이, 수수료, 세금, 환율은 계산에 포함되지 않습니다. 현금은 USD 기준이며 자동 주문·증권사 연동은 없습니다. 목표 수량은 정수, 기존 소수 보유량의 매도는 소수 단위가 될 수 있습니다.
- 백테스트는 월말 종가 즉시 리밸런싱을 가정한 이론값입니다. 실제 신호 확인 후 다음 거래일 체결 시 성과가 다를 수 있습니다. 일중 낙폭은 계산하지 않습니다.
- 전 기간 DBC를 사용하는 백테스트와 PDBC를 사용하는 실전 신호가 다를 수 있습니다. 과거 백테스트 시계열은 Yahoo의 배당·분할 수정으로 이후 바뀔 수 있고, 별도 저장한 월별 확정본은 그대로 보존합니다.
- 60초 이상 걸리는 백테스트는 Hosting 시간 제한에 걸릴 수 있습니다. 소수 사용자·현재 월별 데이터 규모를 대상으로 했습니다. 규모 확대 시 비동기 작업·결과 캐시를 검토하세요.
- 프런트엔드 운영 의존성 검사와 CLI 개발 의존성 검사를 구분하세요. `npm audit --omit=dev`와 전체 `npm audit` 결과는 다를 수 있습니다. 무조건적인 `npm audit fix --force`로 CLI 버전을 바꾸지 마세요.

## 6. GitHub 검토

신규 `Firebase 웹앱 검증` 워크플로는 `feat/**` 푸시, PR, main 푸시에서 기존+신규 Python 테스트, 정적 검사, UI 테스트, 타입 검사·빌드, 실제 Firestore 에뮬레이터 규칙 테스트를 실행합니다. Firebase 자격증명은 필요 없고 배포 단계도 없습니다.

검토 순서: 새 브랜치 변경사항·CI → 별도 Firebase 테스트 프로젝트 배포 승인 → 실데이터/로그인/비용 검증 → main 머지 승인. 기존 Streamlit 실행 방법은 그대로입니다.

```bash
pip install -r requirements.txt
streamlit run haa_brk-b_screener_web.py
```

"""demo-haa 로컬 Firestore만 대상으로 저장·잠금·장애복구 통합 검증."""

import os
import sys
from pathlib import Path

if os.getenv("FIRESTORE_EMULATOR_HOST") not in {"127.0.0.1:8080", "localhost:8080"}:
    raise SystemExit("로컬 Firestore 에뮬레이터 환경에서만 실행할 수 있습니다.")
if os.getenv("GCLOUD_PROJECT", os.getenv("GOOGLE_CLOUD_PROJECT")) != "demo-haa":
    raise SystemExit("demo-haa 프로젝트에서만 실행할 수 있습니다.")

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / "functions"), str(root / "tests")]

import requests  # noqa: E402
from haa.service import cached_backtest, refresh  # noqa: E402
from haa.store import create_store  # noqa: E402
from test_firebase_service import FakeMarket  # noqa: E402

store = create_store()
provider = FakeMarket()
assert refresh(store, provider, "2026-08-28T21:30:00Z", "daily") == "updated"
before = store.get("signals/confirmed")
assert cached_backtest(store)["basis"] == "SPY"
assert refresh(store, provider, "2026-08-28T21:30:00Z", "daily") == "duplicate"


def broken(*args, **kwargs):
    raise RuntimeError("검증용 Yahoo 장애")


provider.daily = broken
try:
    refresh(store, provider, "2026-08-28T21:45:00Z", "daily")
except RuntimeError:
    pass
else:
    raise AssertionError("장애가 전파되지 않았습니다.")
assert store.get("signals/confirmed") == before
assert store.get("status/daily")["ok"] is False
print("로컬 Firestore 통합 검증 성공: 원자적 저장, 압축 캐시, 잠금, 중복 억제, 실패 fallback")

if os.getenv("FIREBASE_AUTH_EMULATOR_HOST") in {"127.0.0.1:9099", "localhost:9099"}:
    # 오직 로컬 Auth 에뮬레이터에만 존재하는 테스트 계정이다.
    signup = requests.post(
        "http://127.0.0.1:9099/identitytoolkit.googleapis.com/v1/accounts:signUp?key=demo-key",
        json={"email": "smoke@example.test", "password": "local-emulator-only", "returnSecureToken": True},
        timeout=10,
    )
    signup.raise_for_status()
    user = signup.json()
    endpoint = "http://127.0.0.1:5001/demo-haa/us-central1/api/api/backtest"
    assert requests.get(endpoint, timeout=60).status_code == 401
    headers = {"Authorization": "Bearer " + user["idToken"]}
    assert requests.get(endpoint, headers=headers, timeout=60).status_code == 403
    store.db.document(f"access/{user['localId']}").set({"enabled": True})
    response = requests.get(endpoint, headers=headers, timeout=60)
    assert response.status_code == 200, response.text
    assert response.json()["basis"] == "SPY"
    print("로컬 Auth→HTTP API→Firestore 백테스트 통합 검증 성공: 401/403/200")

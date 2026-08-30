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

# 수진과 HAA가 같은 시세를 쓰더라도 잠금·문서·실패 처리는 분리한다.
import pandas as pd  # noqa: E402
from sujin.service import refresh_quotes, run_backtest  # noqa: E402
from test_firebase_sujin import FakeSujin  # noqa: E402

now = pd.Timestamp("2026-08-28T15:00:00Z")
token = store.acquire_manual("sujinQuotes", now)
assert token is not None
assert store.acquire_manual("sujinQuotes", now) is None
assert store.acquire_manual("sujinHistory", now) is not None
store.release_manual("sujinQuotes", "wrong-token")
assert store.acquire_manual("sujinQuotes", now + pd.Timedelta(seconds=70)) is None
store.release_manual("sujinQuotes", token)
assert store.acquire_manual("sujinQuotes", now + pd.Timedelta(seconds=30)) is None
sujin = FakeSujin()
good = refresh_quotes(store, sujin, now + pd.Timedelta(minutes=3))
assert good["source"] == "refreshed"
assert store.get("signals/confirmed") == before
sujin.fail = True
fallback = refresh_quotes(store, sujin, now + pd.Timedelta(minutes=20))
assert fallback["source"] == "fallback" and fallback["market"] == good["market"]
sujin.fail = False
assert run_backtest(store, sujin, now + pd.Timedelta(minutes=3))["basis"] == "SPY 10% + SPYM 10%"
print("수진 실제 Firestore 통합 검증 성공: 독립 잠금·토큰·재시도 제한·캐시 보존·장기 이력")

# 이메일 초대의 두 동시 수락 중 하나만 공유 구성원으로 등록되는지 실제 트랜잭션으로 검증한다.
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from haa import access as family  # noqa: E402
from test_firebase_access import OWNER, claims, lookup  # noqa: E402

family_repo = family.FamilyRepository(store.db)
store.db.document(family.CONFIG).set({"ownerUid": "owner", "ownerEmail": OWNER["email"]})
store.db.document("access/owner").set({"enabled": True, "role": "admin", "ownerUid": "owner", "email": OWNER["email"]})
store.db.document("users/owner").set({"preserve": "실제 계정과 무관한 에뮬레이터 검증값"})
family.manage(family_repo, OWNER, {"action": "invite", "email": claims()["email"]}, now, lookup)
def competing_claim(uid):
    try:
        return family.session(family_repo, claims(uid), now)
    except LookupError:
        # 에뮬레이터의 비관적 잠금 경합도 허용하되, 실패 시 권한이 중복 발급되면 안 된다.
        return {"uid": uid, "enabled": False, "retry": True}


with ThreadPoolExecutor(max_workers=2) as pool:
    concurrent = list(pool.map(competing_claim, ["wife", "duplicate"]))
assert sum(result["enabled"] for result in concurrent) <= 1
# 잠금 경합을 안내받은 요청은 사용자가 다시 확인한 것처럼 재실행한다.
results = [family.session(family_repo, claims(uid), now) for uid in ["wife", "duplicate"]]
assert sum(result["enabled"] for result in results) == 1
winner = next(result["uid"] for result in results if result["enabled"])
family.manage(family_repo, OWNER, {"action": "revoke", "uid": winner}, now, lookup)
assert not family.session(family_repo, claims(winner), now)["enabled"]
assert store.get("users/owner") == {"preserve": "실제 계정과 무관한 에뮬레이터 검증값"}
assert store.get("signals/confirmed") == before
print("가족 권한 실제 Firestore 검증 성공: 동시 초대 수락 1명, 접근 해제, 포트폴리오·신호 보존")

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
    sujin_endpoint = endpoint.replace("/api/backtest", "/api/sujin/backtest")
    # 과거 가상 가격 캐시는 현재 cutoff와 다르므로, 새 조회를 금지하는 현재 잠금으로 fallback 경로를 검증한다.
    current = pd.Timestamp.now(tz="UTC")
    store.db.document("internal/sujinHistoryLease").set(
        {
            "token": "fixture-lock",
            "lastAttempt": current.isoformat(),
            "until": (current + pd.Timedelta(minutes=2)).isoformat(),
        }
    )
    assert requests.post(sujin_endpoint, json={}, timeout=60).status_code == 401
    response = requests.post(sujin_endpoint, headers=headers, json={"initial": 10000}, timeout=60)
    assert response.status_code == 200, response.text
    assert response.json()["basis"] == "SPY 10% + SPYM 10%"
    assert response.json()["source"] == "fallback"
    print("로컬 Auth→HTTP API→Firestore 백테스트 통합 검증 성공: 401/403/200")

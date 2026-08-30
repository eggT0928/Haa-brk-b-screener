"""Firestore에 검증된 마지막 성공값과 중복 실행 방지 잠금을 저장한다."""

import io
import json
import os
import uuid
import zlib

import pandas as pd
from firebase_admin import firestore
from google.auth.credentials import AnonymousCredentials
from google.cloud.firestore import Client


def pack(adjusted, raw):
    # 실행 수량용 원시가격 전체는 필요 없다. IRX 일별 원시 수익률만 보존한다.
    raw = raw.reindex(columns=["^IRX"])
    adjusted = adjusted.drop(columns=["SPYM", "^IRX"], errors="ignore")
    payload = {
        "adjusted": adjusted.to_json(orient="split", date_format="iso"),
        "raw": raw.to_json(orient="split", date_format="iso"),
    }
    blob = zlib.compress(json.dumps(payload).encode(), level=6)
    if len(blob) > 900_000:
        raise ValueError("가격 캐시가 문서 안전 한도 900KB를 넘었습니다. 분할 저장이 필요합니다.")
    return blob


def unpack(blob):
    payload = json.loads(zlib.decompress(blob))
    tables = []
    for key in ("adjusted", "raw"):
        table = pd.read_json(io.StringIO(payload[key]), orient="split")
        table.index = pd.to_datetime(table.index).tz_localize(None)
        tables.append(table)
    return tuple(tables)


class FirestoreStore:
    def __init__(self, db):
        self.db = db

    def get(self, path):
        snapshot = self.db.document(path).get()
        return snapshot.to_dict() if snapshot.exists else None

    def acquire(self, run_key, now):
        ref = self.db.document("internal/refreshLease")
        token = str(uuid.uuid4())

        @firestore.transactional
        def acquire(transaction):
            old = ref.get(transaction=transaction).to_dict() or {}
            if old.get("lastRun") == run_key:
                return None
            if old.get("until") and pd.Timestamp(old["until"]) > now:
                return None
            transaction.set(
                ref, {"token": token, "until": (now + pd.Timedelta(minutes=9)).isoformat()}, merge=True
            )
            return token

        return acquire(self.db.transaction())

    def release(self, token, run_key, success):
        ref = self.db.document("internal/refreshLease")

        @firestore.transactional
        def release(transaction):
            old = ref.get(transaction=transaction).to_dict() or {}
            if old.get("token") == token:
                update = {"until": None, "token": None}
                if success:
                    update["lastRun"] = run_key
                transaction.set(ref, update, merge=True)

        release(self.db.transaction())

    def commit(self, documents):
        batch = self.db.batch()
        for path, value in documents.items():
            batch.set(self.db.document(path), value)
        batch.commit()

    def acquire_manual(self, key, now):
        # HAA 예약 잠금과 다른 문서. 실패한 조회도 60초 동안 연속 재실행을 막는다.
        if key not in {"sujinQuotes", "sujinHistory"}:
            raise ValueError("허용하지 않은 수동 갱신 작업입니다.")
        ref = self.db.document(f"internal/{key}Lease")
        token = str(uuid.uuid4())

        @firestore.transactional
        def acquire(transaction):
            old = ref.get(transaction=transaction).to_dict() or {}
            if old.get("until") and pd.Timestamp(old["until"]) > now:
                return None
            if old.get("lastAttempt") and (now - pd.Timestamp(old["lastAttempt"])).total_seconds() < 60:
                return None
            transaction.set(
                ref,
                {
                    "token": token,
                    "lastAttempt": now.isoformat(),
                    "until": (now + pd.Timedelta(minutes=2)).isoformat(),
                },
            )
            return token

        return acquire(self.db.transaction())

    def release_manual(self, key, token):
        ref = self.db.document(f"internal/{key}Lease")

        @firestore.transactional
        def release(transaction):
            old = ref.get(transaction=transaction).to_dict() or {}
            if old.get("token") == token:
                transaction.update(ref, {"token": None, "until": None})

        release(self.db.transaction())

    def failure(self, job, now):
        # 성공 신호와 시세는 손대지 않아 다음 읽기가 직전 성공값으로 fallback된다.
        self.db.document(f"status/{job}").set(
            {
                "lastAttemptAt": now.isoformat(),
                "ok": False,
                "message": "Yahoo 조회 또는 검증 실패. 직전 성공값을 표시합니다. 서버 로그를 확인하세요.",
            }
        )


def create_store():
    """로컬 demo 에뮬레이터에서는 실제 Google 자격증명을 요구하지 않는다."""
    emulator = os.getenv("FIRESTORE_EMULATOR_HOST")
    if emulator:
        project = os.getenv("GCLOUD_PROJECT", os.getenv("GOOGLE_CLOUD_PROJECT", ""))
        if emulator not in {"127.0.0.1:8080", "localhost:8080"} or not project.startswith("demo-"):
            raise RuntimeError("에뮬레이터는 로컬 demo- 프로젝트에서만 허용됩니다.")
        return FirestoreStore(Client(project=project, credentials=AnonymousCredentials()))
    return FirestoreStore(firestore.client())

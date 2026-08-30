"""검증된 Google 이메일 초대와 한 가족의 공유 권한. 클라이언트는 권한 문서를 쓸 수 없다."""

import hashlib
import re
import uuid

import pandas as pd
from firebase_admin import firestore
from google.api_core.exceptions import Aborted, DeadlineExceeded

CONFIG = "internal/family"


def email_key(email):
    return hashlib.sha256(normalize_email(email).encode()).hexdigest()


def normalize_email(email):
    if not isinstance(email, str):
        raise ValueError("Google 이메일을 입력하세요.")
    email = email.strip().lower()
    if len(email) > 254 or not re.fullmatch(
        r"[a-z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-z0-9-]+(?:\.[a-z0-9-]+)+", email
    ):
        raise ValueError("이메일 형식을 확인하세요.")
    return email


def identity(claims):
    uid = claims.get("uid", "")
    if not isinstance(uid, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", uid):
        raise PermissionError("유효한 로그인 계정이 필요합니다.")
    if (
        claims.get("email_verified") is not True
        or claims.get("firebase", {}).get("sign_in_provider") != "google.com"
    ):
        raise PermissionError("이메일이 확인된 Google 계정으로 로그인하세요.")
    return uid, normalize_email(claims.get("email"))


class FamilyRepository:
    def __init__(self, db):
        self.db = db

    def get(self, path):
        return self.db.document(path).get().to_dict()

    def rows(self, collection):
        # 개인 가족용 관리 목록. 무제한 클라이언트 쿼리나 일반 사용자 목록 노출은 하지 않는다.
        return [{**doc.to_dict(), "id": doc.id} for doc in self.db.collection(collection).limit(100).stream()]

    def mutate(self, paths, operation):
        @firestore.transactional
        def run(transaction):
            refs = [self.db.document(p) for p in sorted(set(paths))]
            values = {
                snapshot.reference.path: snapshot.to_dict()
                for snapshot in self.db.get_all(refs, transaction=transaction)
            }
            result, writes = operation(values)
            for path, value in writes.items():
                transaction.set(self.db.document(path), value)
            return result

        try:
            return run(self.db.transaction(max_attempts=3))
        except (Aborted, DeadlineExceeded) as exc:
            raise LookupError("권한 변경이 겹쳤습니다. 목록을 새로고침한 후 다시 시도하세요.") from exc
        except ValueError as exc:
            if isinstance(exc.__cause__, Aborted):
                raise LookupError("권한 변경이 겹쳤습니다. 목록을 새로고침한 후 다시 시도하세요.") from exc
            raise


def configured(values):
    config = values.get(CONFIG)
    if not config or not config.get("ownerUid"):
        raise LookupError("가족 관리자 초기 설정이 필요합니다. 자동으로 관리자를 지정하지 않습니다.")
    return config


def admin(values, uid):
    config = configured(values)
    grant = values.get(f"access/{uid}") or {}
    if (
        uid != config["ownerUid"]
        or grant.get("enabled") is not True
        or grant.get("role") != "admin"
        or grant.get("ownerUid") != uid
        or grant.get("email") != config.get("ownerEmail")
    ):
        raise PermissionError("초대·승인 관리는 가족 관리자만 할 수 있습니다.")
    return config


def public_grant(grant, uid, email, status=None):
    return {
        "uid": uid,
        "email": email,
        "enabled": grant.get("enabled") is True,
        "role": grant.get("role", "member"),
        "ownerUid": grant.get("ownerUid"),
        "requestStatus": status,
    }


def session(repo, claims, now):
    uid, email = identity(claims)
    now = pd.Timestamp(now)
    access_path, invite_path, request_path = (
        f"access/{uid}",
        f"familyInvites/{email_key(email)}",
        f"familyRequests/{uid}",
    )

    def resolve(values):
        config = configured(values)
        grant, invite, request = (
            values.get(access_path) or {},
            values.get(invite_path) or {},
            values.get(request_path) or {},
        )
        if (
            grant.get("enabled") is True
            and grant.get("ownerUid") == config["ownerUid"]
            and grant.get("email") == email
        ):
            return public_grant(grant, uid, email, "approved"), {}
        if (
            invite.get("status") == "pending"
            and invite.get("ownerUid") == config["ownerUid"]
            and invite.get("email") == email
            and pd.Timestamp(invite["expiresAt"]) > now
            and invite.get("claimedUid") in (None, uid)
        ):
            grant = {
                "enabled": True,
                "role": "member",
                "ownerUid": config["ownerUid"],
                "email": email,
                "updatedAt": now.isoformat(),
                "approvedBy": config["ownerUid"],
            }
            writes = {
                access_path: grant,
                invite_path: {
                    **invite,
                    "status": "accepted",
                    "claimedUid": uid,
                    "acceptedAt": now.isoformat(),
                },
            }
            if request:
                writes[request_path] = {**request, "status": "approved", "updatedAt": now.isoformat()}
            return public_grant(grant, uid, email, "approved"), writes
        status = "revoked" if grant.get("enabled") is False else request.get("status")
        return public_grant({}, uid, email, status), {}

    return repo.mutate([CONFIG, access_path, invite_path, request_path], resolve)


def request_access(repo, claims, now):
    uid, email = identity(claims)
    now = pd.Timestamp(now)
    path = f"familyRequests/{uid}"

    def create(values):
        config = configured(values)
        grant, old = values.get(f"access/{uid}") or {}, values.get(path) or {}
        if grant.get("enabled") is True:
            return {"message": "이미 승인된 계정입니다."}, {}
        if grant.get("enabled") is False:
            raise PermissionError("접근이 해제된 계정입니다. 관리자에게 재초대를 요청하세요.")
        if old.get("status") == "pending":
            return {"message": "승인 요청이 접수되어 있습니다. 관리자의 승인을 기다려 주세요."}, {}
        if old.get("updatedAt") and (now - pd.Timestamp(old["updatedAt"])).total_seconds() < 86400:
            raise ValueError("최근 요청이 처리되었습니다. 하루 후 다시 요청하거나 관리자에게 문의하세요.")
        value = {
            "uid": uid,
            "email": email,
            "ownerUid": config["ownerUid"],
            "status": "pending",
            "updatedAt": now.isoformat(),
        }
        return {"message": "승인을 요청했습니다. 관리자 확인 후 이용할 수 있습니다."}, {path: value}

    return repo.mutate([CONFIG, f"access/{uid}", path], create)


def management_list(repo, claims):
    uid, _ = identity(claims)
    config = repo.mutate([CONFIG, f"access/{uid}"], lambda values: (admin(values, uid), {}))
    owner = config["ownerUid"]
    result = {"ownerUid": owner}
    for key, collection in [
        ("members", "access"),
        ("invites", "familyInvites"),
        ("requests", "familyRequests"),
    ]:
        result[key] = sorted(
            [v for v in repo.rows(collection) if v.get("ownerUid") == owner],
            key=lambda v: v.get("updatedAt", ""),
            reverse=True,
        )
    return result


def manage(repo, claims, data, now, lookup_user):
    uid, actor_email = identity(claims)
    if not isinstance(data, dict) or set(data) - {"action", "email", "uid"}:
        raise ValueError("권한 변경 요청 형식을 확인하세요.")
    action, now = data.get("action"), pd.Timestamp(now)
    # 관리자 확인 전에 Auth 사용자 조회나 다른 계정의 문서 탐색을 하지 않는다.
    repo.mutate([CONFIG, f"access/{uid}"], lambda values: (admin(values, uid), {}))
    paths = [CONFIG, f"access/{uid}"]
    if action in ("invite", "cancel"):
        email = normalize_email(data.get("email"))
        target = None
    elif action in ("approve", "reject", "revoke"):
        target = data.get("uid", "")
        if not isinstance(target, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", target):
            raise ValueError("대상 계정을 확인하세요.")
        if target == uid:
            raise ValueError("본인 관리자 권한은 이 화면에서 변경할 수 없습니다.")
        source = repo.get(f"{'access' if action == 'revoke' else 'familyRequests'}/{target}") or {}
        email = normalize_email(source.get("email"))
        if action == "approve":
            user = lookup_user(target)
            if (
                user.disabled
                or not user.email_verified
                or normalize_email(user.email) != email
                or not any(p.provider_id == "google.com" for p in user.provider_data)
            ):
                raise ValueError(
                    "현재 Google 계정 정보가 승인 요청과 일치하지 않습니다. 다시 로그인해 주세요."
                )
        paths += [f"access/{target}", f"familyRequests/{target}"]
    else:
        raise ValueError("지원하지 않는 권한 관리 작업입니다.")
    invitation = f"familyInvites/{email_key(email)}"
    paths.append(invitation)

    def change(values):
        config = admin(values, uid)
        if email == actor_email or email == config.get("ownerEmail") or target == config["ownerUid"]:
            raise ValueError("가족 관리자 본인은 초대·해제 대상이 아닙니다.")
        invite, writes = values.get(invitation) or {}, {}
        if action == "invite":
            if invite.get("status") == "accepted":
                raise ValueError("이미 참여한 이메일입니다. 구성원 목록에서 상태를 확인하세요.")
            writes[invitation] = {
                "email": email,
                "ownerUid": config["ownerUid"],
                "status": "pending",
                "invitedBy": uid,
                "updatedAt": now.isoformat(),
                "expiresAt": (now + pd.Timedelta(days=14)).isoformat(),
            }
            message = "초대를 등록했습니다. 이메일 자동 발송은 하지 않으며, 링크를 전달하면 해당 Google 계정 로그인 시 승인됩니다."
        elif action == "cancel":
            if invite.get("status") != "pending" or invite.get("ownerUid") != config["ownerUid"]:
                raise ValueError("취소할 대기 초대가 없습니다. 이미 가입했다면 구성원 접근을 해제하세요.")
            writes[invitation] = {**invite, "status": "cancelled", "updatedAt": now.isoformat()}
            message = "초대를 취소했습니다."
        else:
            request_path, access_path = f"familyRequests/{target}", f"access/{target}"
            old, pending = values.get(access_path) or {}, values.get(request_path) or {}
            source = old if action == "revoke" else pending
            if source.get("ownerUid") != config["ownerUid"] or source.get("email") != email:
                raise PermissionError("이 가족의 대상 계정이 아닙니다.")
            if old.get("role") == "admin":
                raise ValueError("관리자 권한은 변경할 수 없습니다.")
            if action in ("approve", "reject") and pending.get("status") != "pending":
                raise ValueError("이미 처리된 요청입니다. 목록을 새로고침하세요.")
            if action == "approve":
                writes[access_path] = {
                    "enabled": True,
                    "role": "member",
                    "ownerUid": config["ownerUid"],
                    "email": email,
                    "approvedBy": uid,
                    "updatedAt": now.isoformat(),
                }
                writes[request_path] = {**pending, "status": "approved", "updatedAt": now.isoformat()}
                if invite:
                    writes[invitation] = {
                        **invite,
                        "status": "accepted",
                        "claimedUid": target,
                        "updatedAt": now.isoformat(),
                    }
                message = "공유 포트폴리오 이용을 승인했습니다."
            elif action == "reject":
                writes[request_path] = {**pending, "status": "rejected", "updatedAt": now.isoformat()}
                if invite.get("status") == "pending":
                    writes[invitation] = {**invite, "status": "cancelled", "updatedAt": now.isoformat()}
                message = "승인 요청을 거절했습니다."
            else:
                writes[access_path] = {**old, "enabled": False, "updatedAt": now.isoformat()}
                if invite:
                    writes[invitation] = {**invite, "status": "revoked", "updatedAt": now.isoformat()}
                message = "접근을 해제했습니다. 저장된 포트폴리오와 이력은 삭제하지 않았습니다."
        writes[f"familyAudit/{uuid.uuid4().hex}"] = {
            "action": action,
            "actorUid": uid,
            "email": email,
            "targetUid": target,
            "createdAt": now.isoformat(),
        }
        return {"message": message}, writes

    return repo.mutate(paths, change)

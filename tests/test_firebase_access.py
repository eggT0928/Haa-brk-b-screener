"""실제 계정·포트폴리오를 건드리지 않는 가족 권한 회귀 검증."""

from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest
from haa import access

NOW = pd.Timestamp("2026-08-30T00:00:00Z")


def claims(uid="wife", email="wife@example.com"):
    return {
        "uid": uid,
        "email": email,
        "email_verified": True,
        "firebase": {"sign_in_provider": "google.com"},
    }


OWNER = claims("owner", "owner@example.com")


class Repository:
    def __init__(self):
        self.docs = {
            access.CONFIG: {"ownerUid": "owner", "ownerEmail": OWNER["email"]},
            "access/owner": {"enabled": True, "role": "admin", "ownerUid": "owner", "email": OWNER["email"]},
            "users/owner": {"holdings": {"SPY": 7}},
            "users/owner/portfolios/sujin": {"holdings": {"SPYM": 9}},
        }

    def get(self, path):
        return deepcopy(self.docs.get(path))

    def rows(self, collection):
        return [
            {**deepcopy(v), "id": p.split("/")[1]}
            for p, v in self.docs.items()
            if p.startswith(collection + "/")
        ]

    def mutate(self, paths, operation):
        result, writes = operation({p: self.get(p) for p in paths})
        self.docs.update(deepcopy(writes))
        return result


def lookup(uid):
    return SimpleNamespace(
        disabled=False,
        email_verified=True,
        email=claims()["email"],
        provider_data=[SimpleNamespace(provider_id="google.com")],
    )


def change(repo, action, **kwargs):
    return access.manage(repo, OWNER, {"action": action, **kwargs}, NOW, lookup)


def invited():
    repo = Repository()
    change(repo, "invite", email="  WIFE@example.com  ")
    return repo


def test_invitation_accepts_verified_google_email_and_preserves_portfolios():
    repo = invited()
    before = {k: v for k, v in repo.docs.items() if k.startswith("users/")}
    result = access.session(repo, claims(), NOW)
    assert result["enabled"] is True and result["ownerUid"] == "owner" and result["role"] == "member"
    assert access.session(repo, claims(), NOW) == result
    assert {k: v for k, v in repo.docs.items() if k.startswith("users/")} == before
    invite = repo.get("familyInvites/" + access.email_key(claims()["email"]))
    assert invite["status"] == "accepted" and invite["claimedUid"] == "wife"
    assert not access.session(repo, claims("imposter"), NOW)["enabled"]
    assert len(repo.rows("familyAudit")) == 1


@pytest.mark.parametrize(
    "update",
    [
        {"email_verified": False},
        {"firebase": {"sign_in_provider": "password"}},
        {"uid": "../owner"},
        {"uid": None},
    ],
)
def test_invalid_identity_never_claims_invitation(update):
    repo = invited()
    with pytest.raises(PermissionError):
        access.session(repo, {**claims(), **update}, NOW)
    assert repo.get("access/wife") is None


def test_uninvited_expired_cancelled_and_different_email_do_not_gain_access():
    repo = invited()
    assert not access.session(repo, claims(email="other@gmail.com"), NOW)["enabled"]
    assert repo.get("familyRequests/wife") is None  # 방문 자체는 승인 요청도 만들지 않는다.
    assert not access.session(repo, claims(), NOW + pd.Timedelta(days=14))["enabled"]
    change(repo, "cancel", email=claims()["email"])
    assert not access.session(repo, claims(), NOW)["enabled"]
    assert repo.get("access/wife") is None


def test_request_approval_revocation_and_reinvitation():
    repo = Repository()
    access.request_access(repo, claims(), NOW)
    access.request_access(repo, claims(), NOW)
    assert len(repo.rows("familyRequests")) == 1
    change(repo, "approve", uid="wife")
    assert access.session(repo, claims(), NOW)["enabled"]
    change(repo, "revoke", uid="wife")
    assert access.session(repo, claims(), NOW)["requestStatus"] == "revoked"
    with pytest.raises(PermissionError):
        access.request_access(repo, claims(), NOW)
    change(repo, "invite", email=claims()["email"])
    assert access.session(repo, claims(), NOW)["enabled"]


def test_reject_also_cancels_pending_invite_and_enforces_cooldown():
    repo = invited()
    access.request_access(repo, claims(), NOW)
    change(repo, "reject", uid="wife")
    assert not access.session(repo, claims(), NOW)["enabled"]
    with pytest.raises(ValueError):
        access.request_access(repo, claims(), NOW + pd.Timedelta(hours=23))
    access.request_access(repo, claims(), NOW + pd.Timedelta(days=1))
    assert repo.get("familyRequests/wife")["status"] == "pending"


@pytest.mark.parametrize("actor", [claims(), claims("stranger", "stranger@gmail.com")])
def test_members_and_strangers_cannot_manage_or_enumerate(actor):
    repo = invited()
    access.session(repo, claims(), NOW)
    with pytest.raises(PermissionError):
        access.management_list(repo, actor)
    with pytest.raises(PermissionError):
        access.manage(repo, actor, {"action": "invite", "email": "other@gmail.com"}, NOW, lookup)


def test_owner_self_removal_and_arbitrary_role_or_root_are_forbidden():
    repo = invited()
    for data in [
        {"action": "revoke", "uid": "owner"},
        {"action": "invite", "email": OWNER["email"]},
        {"action": "invite", "email": claims()["email"], "role": "admin"},
        {"action": "approve", "uid": "wife", "ownerUid": "other"},
    ]:
        with pytest.raises(ValueError):
            access.manage(repo, OWNER, data, NOW, lookup)
    repo.docs["access/wife"] = {**repo.docs["access/owner"], "email": claims()["email"]}
    with pytest.raises(PermissionError):
        access.management_list(repo, claims())


@pytest.mark.parametrize(
    "update",
    [{"disabled": True}, {"email_verified": False}, {"email": "changed@gmail.com"}, {"provider_data": []}],
)
def test_approval_checks_current_auth_user(update):
    repo = Repository()
    access.request_access(repo, claims(), NOW)
    user = lookup("wife")
    for k, v in update.items():
        setattr(user, k, v)
    with pytest.raises(ValueError):
        access.manage(repo, OWNER, {"action": "approve", "uid": "wife"}, NOW, lambda _: user)
    assert repo.get("access/wife") is None


def test_missing_configuration_never_assigns_first_user_as_owner():
    repo = Repository()
    del repo.docs[access.CONFIG]
    with pytest.raises(LookupError):
        access.session(repo, claims(), NOW)
    assert repo.get("access/wife") is None


def test_cancel_after_accept_does_not_silently_revoke_and_scopes_are_checked():
    repo = invited()
    access.session(repo, claims(), NOW)
    with pytest.raises(ValueError):
        change(repo, "cancel", email=claims()["email"])
    repo.docs["access/wife"]["ownerUid"] = "other"
    with pytest.raises(PermissionError):
        change(repo, "revoke", uid="wife")


def test_disabled_admin_is_rechecked_for_every_action():
    repo = Repository()
    repo.docs["access/owner"]["enabled"] = False
    with pytest.raises(PermissionError):
        change(repo, "invite", email=claims()["email"])


@pytest.mark.parametrize("email", [None, "", "bad", "a@localhost", "x" * 250 + "@gmail.com", "a b@gmail.com"])
def test_email_validation(email):
    with pytest.raises(ValueError):
        access.normalize_email(email)

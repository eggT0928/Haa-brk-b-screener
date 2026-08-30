from unittest.mock import Mock

import main
import pytest
from flask import Flask, request
from test_firebase_access import OWNER, Repository, claims

app = Flask(__name__)


def call(path="session", method="POST", data=None, token=True):
    with app.test_request_context(
        "/api/access/" + path,
        method=method,
        json=data,
        headers={"Authorization": "Bearer fixture"} if token else {},
    ):
        return main.api(request)


@pytest.fixture
def setup(monkeypatch):
    repo = Repository()
    monkeypatch.setattr(main, "store", lambda: Mock())
    monkeypatch.setattr(main.family_access, "FamilyRepository", lambda _: repo)
    monkeypatch.setattr(main.auth, "verify_id_token", lambda *a, **kw: claims())
    return repo


def test_session_and_request_are_available_before_approval_but_manage_is_not(setup):
    assert call().json["enabled"] is False
    assert call("request").status_code == 200
    assert call("manage", method="GET").status_code == 403
    assert call("manage", data={"action": "invite", "email": "other@gmail.com"}).status_code == 403
    assert call(token=False).status_code == 401
    assert call(method="GET").status_code == 405


def test_manage_validates_body_and_does_not_trust_role_claims(setup, monkeypatch):
    monkeypatch.setattr(main.auth, "verify_id_token", lambda *a, **kw: OWNER)
    assert call("manage", method="GET").status_code == 200
    for data in [None, [], {"action": "invite", "email": "wife@gmail.com", "role": "admin"}]:
        assert call("manage", data=data).status_code == 400
    assert call("manage", data={"extra": "x" * 3000}).status_code == 413
    assert call("manage", data={"action": "invite", "email": claims()["email"]}).status_code == 200


def test_app_check_runs_before_access_mutation(setup, monkeypatch):
    from firebase_admin import app_check

    monkeypatch.setenv("ENFORCE_APP_CHECK", "true")

    def invalid(*a, **kw):
        raise ValueError("invalid")

    monkeypatch.setattr(app_check, "verify_token", invalid)
    assert call("request").status_code == 403
    assert setup.get("familyRequests/wife") is None

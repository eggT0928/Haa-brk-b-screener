from unittest.mock import Mock

import main
import pytest
from flask import Flask, request
from sujin.service import RefreshBusy

app = Flask(__name__)


def call(path="/api/pension/refresh", headers=None, method="POST", body=None):
    with app.test_request_context(path, headers=headers or {}, method=method, json=body):
        return main.api(request)


@pytest.fixture
def authorized(monkeypatch):
    monkeypatch.setattr(main.auth, "verify_id_token", lambda *a, **kw: {"uid": "owner"})
    store = Mock()
    store.get.return_value = {"enabled": True}
    monkeypatch.setattr(main, "store", lambda: store)
    monkeypatch.setattr(
        main, "pension_refresh", lambda *a, **kw: {"source": "cache", "market": {"currency": "KRW"}}
    )
    return {"Authorization": "Bearer fixture"}


def test_auth_approval_and_post_required(authorized):
    assert call().status_code == 401
    assert call(headers=authorized, method="GET").status_code == 405
    assert call(headers=authorized, body={}).json["market"]["currency"] == "KRW"
    assert call("/pension/refresh", headers=authorized, body={}).status_code == 200
    main.store().get.return_value = None
    assert call(headers=authorized, body={}).status_code == 403


def test_requests_cannot_choose_symbols_or_other_users(authorized, monkeypatch):
    monkeypatch.setattr(main, "pension_refresh", lambda *a, **kw: pytest.fail("조회 금지"))
    for body in [{"uid": "other"}, {"symbols": ["SPY"]}, [], "bad"]:
        assert call(headers=authorized, body=body).status_code == 400
    assert call(headers=authorized, body={"x": "x" * 2000}).status_code == 413


def test_app_check_enforced_before_quotes(authorized, monkeypatch):
    from firebase_admin import app_check

    monkeypatch.setenv("ENFORCE_APP_CHECK", "true")
    monkeypatch.setattr(app_check, "verify_token", lambda _: (_ for _ in ()).throw(ValueError()))
    monkeypatch.setattr(main, "pension_refresh", lambda *a, **kw: pytest.fail("조회 금지"))
    assert call(headers=authorized).status_code == 403


def test_cache_missing_and_retry_after_are_actionable(authorized, monkeypatch):
    def fail(*args, **kwargs):
        raise LookupError("연금 캐시 없음")

    monkeypatch.setattr(main, "pension_refresh", fail)
    assert call(headers=authorized).status_code == 503

    def busy(*args, **kwargs):
        raise RefreshBusy("잠시 후 재시도")

    monkeypatch.setattr(main, "pension_refresh", busy)
    result = call(headers=authorized)
    assert result.status_code == 429 and result.headers["Retry-After"] == "60"


def test_pension_endpoint_does_not_change_haa_or_sujin_routes(authorized, monkeypatch):
    monkeypatch.setattr(main, "refresh_quotes", lambda *a, **kw: {"basis": "sujin"})
    monkeypatch.setattr(main, "cached_backtest", lambda *a, **kw: {"basis": "SPY"})
    assert call("/api/sujin/refresh", headers=authorized).json == {"basis": "sujin"}
    assert call("/api/backtest", headers=authorized, method="GET").json == {"basis": "SPY"}

import main
import pytest
from flask import Flask, request
from sujin.service import RefreshBusy

app = Flask(__name__)


def call(path="/api/sujin/refresh", headers=None, method="POST", body=None):
    with app.test_request_context(path, headers=headers or {}, method=method, json=body):
        return main.api(request)


@pytest.fixture
def authorized(monkeypatch):
    from unittest.mock import Mock

    monkeypatch.setattr(main.auth, "verify_id_token", lambda token, **kwargs: {"uid": "owner"})
    store = Mock()
    store.get.return_value = {"enabled": True}
    monkeypatch.setattr(main, "store", lambda: store)
    monkeypatch.setattr(main, "refresh_quotes", lambda *a, **k: {"source": "cache"})
    monkeypatch.setattr(main, "sujin_backtest", lambda *a, **k: {"basis": "SPY+SPYM"})
    return {"Authorization": "Bearer fixture"}


def test_manual_routes_require_auth_approval_and_post(authorized):
    assert call().status_code == 401
    assert call(headers=authorized, method="GET").status_code == 405
    assert call(headers=authorized).json == {"source": "cache"}
    main.store().get.return_value = None
    assert call(headers=authorized).status_code == 403


@pytest.mark.parametrize(
    "body",
    [
        None,
        [],
        {"initial": -1},
        {"initial": True},
        {"initial": "bad"},
        {"start": "2026-02-31"},
        {"start": "2026-07-01", "end": "2025-01-01"},
        {"uid": "other"},
        {"start": 15},
    ],
)
def test_invalid_backtest_request_never_fetches_yahoo(authorized, monkeypatch, body):
    monkeypatch.setattr(main, "sujin_backtest", lambda *a, **k: pytest.fail("조회 금지"))
    assert call("/api/sujin/backtest", authorized, body=body).status_code == 400


def test_sujin_and_haa_routes_remain_separate(authorized, monkeypatch):
    monkeypatch.setattr(main, "cached_backtest", lambda *a, **k: {"basis": "SPY"})
    assert call("/api/sujin/backtest", authorized, body={"initial": 10000}).json["basis"] == "SPY+SPYM"
    assert call("/api/backtest", authorized, method="GET").json["basis"] == "SPY"
    assert call("/api/backtest", authorized).status_code == 405


def test_cooldown_has_retry_after(authorized, monkeypatch):
    def busy(*args, **kwargs):
        raise RefreshBusy("잠시 후 재시도")

    monkeypatch.setattr(main, "refresh_quotes", busy)
    response = call(headers=authorized)
    assert response.status_code == 429 and response.headers["Retry-After"] == "60"


def test_first_failure_returns_no_fake_success(authorized, monkeypatch):
    def failure(*args, **kwargs):
        raise LookupError("캐시 없음")

    monkeypatch.setattr(main, "refresh_quotes", failure)
    assert call(headers=authorized).status_code == 503


def test_app_check_enforced_for_sujin_too(authorized, monkeypatch):
    from firebase_admin import app_check

    monkeypatch.setenv("ENFORCE_APP_CHECK", "true")
    monkeypatch.setattr(app_check, "verify_token", lambda token: (_ for _ in ()).throw(ValueError()))
    assert call(headers=authorized).status_code == 403

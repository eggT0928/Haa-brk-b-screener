from unittest.mock import Mock

import main
import pytest
from flask import Flask, request

app = Flask(__name__)


def call(path="/api/backtest", headers=None, method="GET"):
    with app.test_request_context(path, headers=headers or {}, method=method):
        return main.api(request)


@pytest.fixture
def authorized(monkeypatch):
    monkeypatch.setattr(main.auth, "verify_id_token", lambda token, **kwargs: {"uid": "owner"})
    store = Mock()
    store.get.return_value = {"enabled": True}
    monkeypatch.setattr(main, "store", lambda: store)
    monkeypatch.setattr(main, "cached_backtest", lambda *a, **k: {"basis": "SPY"})
    return {"Authorization": "Bearer fixture-token"}


def test_unauthenticated_and_invalid_tokens_rejected(monkeypatch):
    assert call().status_code == 401

    def invalid(*args, **kwargs):
        raise ValueError("invalid token")

    monkeypatch.setattr(main.auth, "verify_id_token", invalid)
    assert call(headers={"Authorization": "Bearer invalid"}).status_code == 401


def test_unapproved_users_cannot_read_cache(authorized, monkeypatch):
    main.store().get.return_value = None
    assert call(headers=authorized).status_code == 403


def test_authenticated_backtest_and_no_shared_http_cache(authorized):
    response = call(headers=authorized)
    assert response.status_code == 200
    assert response.json == {"basis": "SPY"}
    assert "no-store" in response.headers["Cache-Control"]


def test_invalid_date_unknown_route_and_write_rejected(authorized):
    assert call("/api/backtest?start=abc", authorized).status_code == 400
    assert call("/api/other", authorized).status_code == 404
    assert call(headers=authorized, method="POST").status_code == 405


def test_missing_cache_returns_actionable_error(authorized, monkeypatch):
    def missing(*args, **kwargs):
        raise LookupError("초기 갱신 필요")

    monkeypatch.setattr(main, "cached_backtest", missing)
    result = call(headers=authorized)
    assert result.status_code == 503
    assert "초기" in result.json["error"]

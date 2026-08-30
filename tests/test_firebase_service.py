from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from haa.core import APP_TICKERS
from haa.market import YahooMarket, retry, validate_month_ends, validate_row
from haa.service import cached_backtest, refresh
from haa.store import pack, unpack


class MemoryStore:
    def __init__(self):
        self.docs, self.last, self.lock = {}, None, False

    def get(self, path):
        return deepcopy(self.docs.get(path))

    def acquire(self, run_key, now):
        if self.lock or run_key == self.last:
            return None
        self.lock = True
        return "test-token"

    def release(self, token, run_key, success):
        self.lock = False
        if success:
            self.last = run_key

    def commit(self, documents):
        self.docs.update(deepcopy(documents))

    def failure(self, job, now):
        self.docs[f"status/{job}"] = {"ok": False, "lastAttemptAt": now.isoformat()}


class FakeMarket:
    def __init__(self):
        days = pd.bdate_range("2024-01-01", "2026-08-28")
        self.adjusted = pd.DataFrame(
            {t: 100 + np.arange(len(days)) / 100 for t in APP_TICKERS + ["DBC", "^IRX"]}, index=days
        )
        self.raw = self.adjusted.copy()
        self.raw["^IRX"] = 3
        self.raw["SPYM"] = 50

    def daily(self, now, full=False):
        return self.adjusted, self.raw

    def quotes(self, now):
        return {t: 110.0 for t in APP_TICKERS}, {t: now.isoformat() for t in APP_TICKERS}


def test_retry_then_success_and_exhaustion():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("일시 오류")
        return 42

    waits = []
    assert retry(flaky, sleep=waits.append) == 42
    assert waits == [1, 2]
    with pytest.raises(ValueError):
        retry(lambda: int("실패"), sleep=lambda _: None)


def test_daily_success_cache_and_immutable_confirmed_history():
    store, provider = MemoryStore(), FakeMarket()
    refresh(store, provider, "2026-08-28T21:30:00Z", "daily")
    before = store.get("signals/confirmed")
    assert before["month"] == "2026-07"
    assert store.get("signalHistory/2026-07") == before
    assert cached_backtest(store)["basis"] == "SPY"
    provider.adjusted["TIP"] *= 2
    refresh(store, provider, "2026-08-28T21:45:00Z", "daily")
    assert store.get("signals/confirmed") == before
    assert store.get("signalHistory/2026-07") == before


def test_failure_keeps_last_good_snapshot_and_timestamp(monkeypatch):
    store, provider = MemoryStore(), FakeMarket()
    refresh(store, provider, "2026-08-28T15:00:00Z")
    before = store.get("signals/preview")
    prices = store.get("market/current")

    def broken(*args, **kwargs):
        raise RuntimeError("Yahoo 장애")

    monkeypatch.setattr(provider, "daily", broken)
    with pytest.raises(RuntimeError):
        refresh(store, provider, "2026-08-28T15:15:00Z")
    assert store.get("signals/preview") == before
    assert store.get("market/current") == prices
    assert store.get("status/intraday")["ok"] is False
    assert not store.lock


def test_missing_ticker_fails_atomically_not_partial_publication():
    store, provider = MemoryStore(), FakeMarket()
    provider.adjusted.loc[pd.Timestamp("2026-07-31"), "TIP"] = np.nan
    with pytest.raises(ValueError):
        refresh(store, provider, "2026-08-28T21:30:00Z", "daily")
    assert store.get("signals/confirmed") is None
    assert store.get("internal/history") is None


def test_duplicate_job_and_market_closed_skip_provider(monkeypatch):
    store, provider = MemoryStore(), FakeMarket()
    assert refresh(store, provider, "2026-08-28T15:00:00Z") == "updated"
    monkeypatch.setattr(provider, "daily", lambda *a, **k: pytest.fail("조회하면 안 됨"))
    assert refresh(store, provider, "2026-08-28T15:00:00Z") == "duplicate"
    assert refresh(store, provider, "2026-08-29T15:00:00Z") == "closed"


def test_roundtrip_compressed_cache_and_raw_irx_preserved():
    source = FakeMarket()
    blob = pack(source.adjusted, source.raw)
    adjusted, raw = unpack(blob)
    assert len(blob) < 900000
    assert adjusted.loc[pd.Timestamp("2026-08-28"), "SPY"] == pytest.approx(source.adjusted.SPY.iloc[-1])
    assert list(raw.columns) == ["^IRX"]
    assert raw["^IRX"].iloc[-1] == 3


def test_missing_close_and_no_bootstrap_cache_are_errors():
    with pytest.raises(ValueError):
        validate_row(FakeMarket().adjusted, pd.Timestamp("2026-08-31"), ["SPY"])
    with pytest.raises(LookupError):
        cached_backtest(MemoryStore())


def test_provider_rejects_stale_intraday_bars(monkeypatch):
    import haa.market as module

    stamp = pd.Timestamp("2026-08-28T15:00:00Z")
    table = pd.DataFrame(100.0, index=[stamp], columns=APP_TICKERS)
    frame = pd.concat({"Close": table}, axis=1)
    monkeypatch.setattr(module.yf, "download", lambda *a, **k: frame)
    monkeypatch.setattr(module, "retry", lambda op: op())
    with pytest.raises(ValueError, match="오래"):
        YahooMarket().quotes(stamp + pd.Timedelta(hours=1))
    prices, times = YahooMarket().quotes(stamp + pd.Timedelta(minutes=1))
    assert prices["SPY"] == 100
    assert times["SPY"] == stamp.isoformat()


def test_partial_optional_spym_does_not_block_signals(monkeypatch):
    import haa.market as module

    stamp = pd.Timestamp("2026-08-28T15:00:00Z")
    table = pd.DataFrame(100.0, index=[stamp], columns=APP_TICKERS)
    table["SPYM"] = np.nan
    monkeypatch.setattr(module.yf, "download", lambda *a, **k: pd.concat({"Close": table}, axis=1))
    monkeypatch.setattr(module, "retry", lambda op: op())
    prices, _ = YahooMarket().quotes(stamp)
    assert "SPYM" not in prices and "SPY" in prices


def test_missing_historical_month_end_date_is_not_previous_day_close():
    data = FakeMarket().adjusted.drop(pd.Timestamp("2026-07-31"))
    with pytest.raises(ValueError, match="2026-07-31"):
        validate_month_ends(data, "2026-08-28T15:00:00Z", ["SPY"])

"""KRX 달력, 원시가격 검증, 독립 수동 캐시와 실패 복구 검증."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from pension import market
from pension.market import SYMBOLS, TICKERS, PensionMarket, is_open, last_close, quote_payload
from pension.service import recent_quotes, refresh_quotes
from sujin.service import RefreshBusy
from test_firebase_sujin import ManualStore


class FakePension:
    def __init__(self, mode="intraday"):
        self.mode, self.calls, self.fail = mode, 0, False

    def quotes(self, now):
        self.calls += 1
        if self.fail:
            raise ValueError("검증용 Yahoo 장애")
        stamp = now if self.mode == "intraday" else last_close(now)[1]
        return quote_payload(
            dict.fromkeys(TICKERS, 10000.0), dict.fromkeys(TICKERS, stamp.isoformat()), now, self.mode
        )


def yahoo_frame(index, price=10000):
    frame = pd.DataFrame({symbol: price for symbol in SYMBOLS}, index=pd.DatetimeIndex(index))
    frame.columns = pd.MultiIndex.from_product([["Close"], frame.columns])
    return frame


@pytest.fixture(autouse=True)
def no_retry_delay(monkeypatch):
    monkeypatch.setattr(market, "retry", lambda operation, attempts=2: operation())


def test_krx_holiday_weekend_and_settlement_do_not_follow_us_calendar():
    assert is_open("2026-10-26T03:00:00Z")
    assert not is_open("2026-10-26T07:00:00Z")
    assert not is_open("2026-10-24T03:00:00Z")
    assert not is_open("2026-10-09T03:00:00Z")  # 한글날: 미국 거래일이어도 KRX는 휴장.
    assert last_close("2026-10-26T07:00:00Z")[0] == pd.Timestamp("2026-10-23")
    assert last_close("2026-10-26T07:30:00Z")[0] == pd.Timestamp("2026-10-26")
    assert last_close("2026-10-09T10:00:00Z")[0] == pd.Timestamp("2026-10-08")


def test_current_intraday_prices_use_krx_codes_and_actual_timestamps(monkeypatch):
    now = pd.Timestamp("2026-10-26T03:00:00Z")
    calls = []

    def download(symbols, **kwargs):
        calls.append((symbols, kwargs))
        return yahoo_frame([now - pd.Timedelta(minutes=5)])

    monkeypatch.setattr(market.yf, "download", download)
    result = PensionMarket().quotes(now)
    assert result["currency"] == "KRW" and result["quoteMode"] == "intraday"
    assert set(result["prices"]) == set(TICKERS)
    assert set(result["priceTimes"].values()) == {"2026-10-26T02:55:00+00:00"}
    assert calls[0][0] == SYMBOLS and calls[0][1]["auto_adjust"] is False


def test_failed_intraday_falls_back_to_last_completed_close_not_current_daily(monkeypatch):
    now = pd.Timestamp("2026-10-26T03:00:00Z")
    calls = []

    def download(symbols, **kwargs):
        calls.append(kwargs["interval"])
        if kwargs["interval"] == "1m":
            return yahoo_frame([now - pd.Timedelta(hours=2)])
        return yahoo_frame(["2026-10-23", "2026-10-26"], price=15000)

    monkeypatch.setattr(market.yf, "download", download)
    result = PensionMarket().quotes(now)
    assert result["quoteMode"] == "close" and result["marketDate"] == "2026-10-23"
    assert set(result["priceTimes"].values()) == {"2026-10-23T06:30:00+00:00"}
    assert calls == ["1m", "1d"]


def test_closing_requires_actual_last_session_not_previous_available_row(monkeypatch):
    monkeypatch.setattr(market.yf, "download", lambda *a, **kw: yahoo_frame(["2026-10-22"]))
    with pytest.raises(ValueError, match="종가"):
        PensionMarket().quotes("2026-10-24T08:00:00Z")


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf, True, "10000"])
def test_zero_nonfinite_or_invalid_prices_never_become_success(bad):
    now = pd.Timestamp("2026-10-26T03:00:00Z")
    prices = dict.fromkeys(TICKERS, 10000.0)
    prices[TICKERS[-1]] = bad
    with pytest.raises(ValueError):
        quote_payload(prices, dict.fromkeys(TICKERS, now.isoformat()), now, "intraday")


def test_timezone_missing_and_future_quotes_are_rejected():
    now = pd.Timestamp("2026-10-26T03:00:00Z")
    for stamp in ["2026-10-26T03:00:00", "2026-10-26T03:02:00Z", "2026-10-23T06:30:00Z"]:
        with pytest.raises(ValueError):
            quote_payload(dict.fromkeys(TICKERS, 10000.0), dict.fromkeys(TICKERS, stamp), now, "intraday")


def test_intraday_ten_minute_shared_cache_and_independent_documents():
    now = pd.Timestamp("2026-10-26T03:00:00Z")
    store, provider = ManualStore(), FakePension()
    store.docs["market/current"] = {"preserved": True}
    store.docs["market/sujin"] = {"preserved": "sujin"}
    first = refresh_quotes(store, provider, now)
    assert first["source"] == "refreshed"
    assert refresh_quotes(store, provider, now + pd.Timedelta(minutes=9))["source"] == "cache"
    assert provider.calls == 1
    assert refresh_quotes(store, provider, now + pd.Timedelta(minutes=10))["source"] == "refreshed"
    assert provider.calls == 2
    assert store.docs["market/current"] == {"preserved": True}
    assert store.docs["market/sujin"] == {"preserved": "sujin"}
    assert set(store.gates) == {"pensionQuotes"}


def test_weekend_close_cache_preserves_original_updated_at_without_yahoo():
    now = pd.Timestamp("2026-10-24T08:00:00Z")
    store, provider = ManualStore(), FakePension("close")
    first = refresh_quotes(store, provider, now)
    second = refresh_quotes(store, provider, now + pd.Timedelta(days=1))
    assert second["source"] == "cache" and provider.calls == 1
    assert second["market"] == first["market"]


def test_failure_preserves_all_success_values_and_timestamp_then_throttles():
    now = pd.Timestamp("2026-10-26T03:00:00Z")
    store, provider = ManualStore(), FakePension()
    first = refresh_quotes(store, provider, now)
    before = deepcopy(store.docs["market/pension"])
    provider.fail = True
    result = refresh_quotes(store, provider, now + pd.Timedelta(minutes=20))
    assert result["source"] == "fallback" and result["market"] == before == first["market"]
    assert store.docs["status/pensionQuotes"]["ok"] is False
    assert refresh_quotes(store, provider, now + pd.Timedelta(minutes=20, seconds=30))["source"] == "fallback"
    assert provider.calls == 2


def test_first_failure_has_no_fake_price_and_is_throttled():
    now = pd.Timestamp("2026-10-26T03:00:00Z")
    store, provider = ManualStore(), FakePension()
    provider.fail = True
    with pytest.raises(LookupError):
        refresh_quotes(store, provider, now)
    assert "market/pension" not in store.docs
    with pytest.raises(RefreshBusy):
        refresh_quotes(store, provider, now + pd.Timedelta(seconds=30))
    assert provider.calls == 1


def test_cache_requires_krw_complete_prices_and_matching_session():
    now = pd.Timestamp("2026-10-26T03:00:00Z")
    value = FakePension().quotes(now)
    assert recent_quotes(value, now)
    assert not recent_quotes({**value, "currency": "USD"}, now)
    assert not recent_quotes({**value, "prices": {}}, now)
    assert not recent_quotes(value, now + pd.Timedelta(days=1))


def test_provider_retries_are_bounded_to_two(monkeypatch):
    attempts = []
    original_retry = __import__("haa.market", fromlist=["retry"]).retry
    monkeypatch.setattr(
        market, "retry", lambda op, attempts=2: original_retry(op, attempts=attempts, sleep=lambda _: None)
    )

    def fail(*args, **kwargs):
        attempts.append(1)
        raise ValueError("missing")

    monkeypatch.setattr(market.yf, "download", fail)
    with pytest.raises(ValueError):
        PensionMarket().quotes("2026-10-24T08:00:00Z")
    assert len(attempts) == 2

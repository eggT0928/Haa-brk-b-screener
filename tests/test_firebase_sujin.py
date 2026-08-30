from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from haa.calendar import schedule
from sujin.core import TICKERS, WEIGHTS, backtest
from sujin.market import SujinMarket
from sujin.service import RefreshBusy, pack, refresh_quotes, refresh_with_haa, run_backtest, unpack
from test_firebase_service import FakeMarket, MemoryStore


def monthly_data():
    dates = pd.date_range("2025-01-31", "2026-07-31", freq="ME")
    table = pd.DataFrame(
        {t: 100 * (1.01 + i / 1000) ** np.arange(len(dates)) for i, t in enumerate(TICKERS)}, index=dates
    )
    return table, pd.Series(3.0, index=pd.date_range("2025-01-01", "2026-07-31"))


class ManualStore(MemoryStore):
    def __init__(self):
        super().__init__()
        self.gates = {}

    def acquire_manual(self, key, now):
        previous = self.gates.get(key)
        if previous and (now - previous).total_seconds() < 60:
            return None
        self.gates[key] = now
        return "manual-token"

    def release_manual(self, key, token):
        pass


class FakeSujin:
    def __init__(self):
        self.quote_calls = self.history_calls = 0
        self.fail = False
        self.base = None

    def quotes(self, now, base=None):
        self.quote_calls += 1
        self.base = base
        if self.fail:
            raise ValueError("수진 전용 Yahoo 오류")
        return dict.fromkeys(TICKERS, 100.0), dict.fromkeys(TICKERS, now.isoformat())

    def history(self, now):
        self.history_calls += 1
        if self.fail:
            raise ValueError("수진 이력 조회 오류")
        return monthly_data()


def test_original_weights_and_weighted_monthly_formula_preserved():
    table, rates = monthly_data()
    result = backtest(table, rates)
    assert WEIGHTS["SPY"] == WEIGHTS["SPYM"] == 0.1
    assert WEIGHTS["PDBC"] == 0.05 and "DBC" not in WEIGHTS
    expected = 10000.0
    for i in range(1, len(table)):
        expected *= 1 + sum(w * (table.iloc[i][t] / table.iloc[i - 1][t] - 1) for t, w in WEIGHTS.items())
    assert result["curve"][-1]["value"] == pytest.approx(expected)
    assert result["metrics"]["riskFree"] == pytest.approx(0.03)


def test_sujin_spym_changes_its_backtest_but_dbc_does_not():
    table, rates = monthly_data()
    first = backtest(table, rates)
    table["DBC"] = np.arange(len(table)) + 1
    assert backtest(table, rates) == first
    table.loc[table.index[-1], "SPYM"] *= 1.2
    assert backtest(table, rates)["metrics"]["totalReturn"] > first["metrics"]["totalReturn"]


@pytest.mark.parametrize("mode", ["ticker", "middle", "last", "nan", "negative"])
def test_missing_data_is_never_reweighted_or_treated_as_zero(mode):
    table, rates = monthly_data()
    if mode == "ticker":
        table = table.drop(columns="JEPQ")
    elif mode == "middle":
        table = table.drop(table.index[4])
    elif mode == "last":
        table.iloc[-1, 0] = np.nan
    else:
        table.iloc[4, 0] = np.nan if mode == "nan" else -1
    with pytest.raises(ValueError):
        backtest(table, rates)


def test_range_first_loss_and_irx_missing():
    table, rates = monthly_data()
    table.iloc[1:] *= 0.5
    result = backtest(table, rates.iloc[:0], end="2025-05-31")
    assert result["actualEnd"] == "2025-05-31"
    assert result["metrics"]["mdd"] < -0.4
    assert result["metrics"]["riskFree"] is None and result["metrics"]["sharpe"] is None
    assert result["warnings"]
    for kwargs in [
        {"start": "2026-01-01", "end": "2025-01-01"},
        {"initial": float("nan")},
        {"start": "2030-01-01"},
        {"initial": -1},
    ]:
        with pytest.raises(ValueError):
            backtest(table, rates, **kwargs)


def test_cache_preserves_spym_and_irx():
    monthly, rates = monthly_data()
    copy, irx = unpack(pack(monthly, rates))
    assert "SPYM" in copy
    np.testing.assert_allclose(copy.to_numpy(), monthly.to_numpy())
    np.testing.assert_allclose(irx.to_numpy(), rates.to_numpy())


def test_manual_cache_ten_minutes_then_refresh():
    store, provider = ManualStore(), FakeSujin()
    now = pd.Timestamp("2026-08-28T15:00:00Z")
    assert refresh_quotes(store, provider, now)["source"] == "refreshed"
    assert refresh_quotes(store, provider, now + pd.Timedelta(minutes=5))["source"] == "cache"
    assert provider.quote_calls == 1
    assert refresh_quotes(store, provider, now + pd.Timedelta(minutes=15))["source"] == "refreshed"


def test_failure_preserves_old_timestamp_and_haa_documents():
    store, provider = ManualStore(), FakeSujin()
    now = pd.Timestamp("2026-08-28T15:00:00Z")
    store.docs["market/current"] = {"haa": "변경 없음"}
    old = refresh_quotes(store, provider, now)["market"]
    provider.fail = True
    response = refresh_quotes(store, provider, now + pd.Timedelta(minutes=20))
    assert response["source"] == "fallback" and response["market"] == old
    assert store.docs["status/sujinQuotes"]["ok"] is False
    assert store.docs["market/current"] == {"haa": "변경 없음"}


def test_failed_first_request_is_throttled_and_no_fake_cache_created():
    store, provider = ManualStore(), FakeSujin()
    provider.fail = True
    now = pd.Timestamp("2026-08-28T15:00:00Z")
    with pytest.raises(LookupError):
        refresh_quotes(store, provider, now)
    with pytest.raises(RefreshBusy):
        refresh_quotes(store, provider, now + pd.Timedelta(seconds=5))
    assert "market/sujin" not in store.docs and provider.quote_calls == 1


def test_shared_refresh_commits_haa_even_when_sujin_fails():
    store, provider = ManualStore(), FakeSujin()
    provider.fail = True
    result = refresh_with_haa(store, FakeMarket(), provider, "2026-08-28T21:30:00Z", "daily")
    assert result == "updated"
    assert store.get("signals/confirmed")["month"] == "2026-07"
    assert store.get("status/daily")["ok"] is True
    assert store.get("status/sujinQuotes")["ok"] is False
    assert provider.base == store.get("market/current")


def test_closed_or_duplicate_haa_does_not_start_extra_queries():
    store, provider = ManualStore(), FakeSujin()
    assert refresh_with_haa(store, FakeMarket(), provider, "2026-08-29T15:00:00Z", "intraday") == "closed"
    assert provider.quote_calls == 0
    refresh_with_haa(store, FakeMarket(), provider, "2026-08-28T15:00:00Z", "intraday")
    calls = provider.quote_calls
    assert refresh_with_haa(store, FakeMarket(), provider, "2026-08-28T15:00:00Z", "intraday") == "duplicate"
    assert provider.quote_calls == calls


def test_history_cache_and_failure_fallback_preserves_cutoff():
    store, provider = ManualStore(), FakeSujin()
    now = pd.Timestamp("2026-08-28T15:00:00Z")
    first = run_backtest(store, provider, now)
    run_backtest(store, provider, now + pd.Timedelta(minutes=1), start="2026-01-01")
    assert provider.history_calls == 1
    provider.fail = True
    fallback = run_backtest(store, provider, now + pd.Timedelta(days=1))
    assert fallback["source"] == "fallback"
    assert fallback["dataUpdatedAt"] == first["dataUpdatedAt"] and fallback["warnings"]


def test_shared_quotes_download_only_three_extra_tickers(monkeypatch):
    import sujin.market as module

    now = pd.Timestamp("2026-08-28T15:00:00Z")
    common = [t for t in TICKERS if t not in ["QQQM", "JEPQ", "GLD"]]
    base = {"prices": dict.fromkeys(common, 99.0), "priceTimes": dict.fromkeys(common, now.isoformat())}
    calls = []

    def download(tickers, **kwargs):
        calls.append(tickers)
        return pd.concat({"Close": pd.DataFrame(100.0, index=[now], columns=tickers)}, axis=1)

    monkeypatch.setattr(module.yf, "download", download)
    prices, _ = SujinMarket().quotes(now, base)
    assert set(calls[0]) == {"QQQM", "JEPQ", "GLD"}
    assert prices["SPY"] == 99 and prices["JEPQ"] == 100


def test_stale_shared_price_is_not_reused_and_missing_quote_fails(monkeypatch):
    import sujin.market as module

    now = pd.Timestamp("2026-08-28T15:00:00Z")
    base = {
        "prices": dict.fromkeys(TICKERS, 99.0),
        "priceTimes": dict.fromkeys(TICKERS, (now - pd.Timedelta(hours=1)).isoformat()),
    }
    seen = []

    def download(tickers, **kwargs):
        seen.extend(tickers)
        return pd.concat({"Close": pd.DataFrame(np.nan, index=[now], columns=tickers)}, axis=1)

    monkeypatch.setattr(module.yf, "download", download)
    monkeypatch.setattr(module, "retry", lambda fn, **kwargs: fn())
    with pytest.raises(ValueError):
        SujinMarket().quotes(now, base)
    assert set(seen) == set(TICKERS)


def test_history_uses_completed_actual_month_end_and_never_fills_missing_day(monkeypatch):
    import sujin.market as module

    now = pd.Timestamp("2026-08-28T15:00:00Z")
    days = schedule("2025-01-01", "2026-08-28").index
    table = pd.DataFrame(100.0, index=days, columns=TICKERS + ["^IRX"])
    frame = pd.concat({"Close": table, "Adj Close": table}, axis=1)
    monkeypatch.setattr(module.yf, "download", lambda *args, **kwargs: frame)
    monkeypatch.setattr(module, "retry", lambda fn, **kwargs: fn())
    monthly, _ = SujinMarket().history(now)
    assert monthly.index[-1] == pd.Timestamp("2026-07-31")
    frame = frame.drop(pd.Timestamp("2026-06-30"))
    with pytest.raises(ValueError, match="실제 월말"):
        SujinMarket().history(now)


def test_weekend_close_cache_is_reused_without_yahoo(monkeypatch):
    import sujin.market as module

    now = pd.Timestamp("2026-08-30T05:00:00Z")
    base = {
        "prices": dict.fromkeys(TICKERS, 99.0),
        "priceTimes": dict.fromkeys(TICKERS, "2026-08-28T20:00:00Z"),
    }
    monkeypatch.setattr(module.yf, "download", lambda *a, **k: pytest.fail("주말 캐시 재사용"))
    prices, _ = SujinMarket().quotes(now, base)
    assert prices == base["prices"]
    store = ManualStore()
    store.docs["market/sujin"] = {**deepcopy(base), "updatedAt": "2026-08-28T21:30:00Z"}
    assert refresh_quotes(store, FakeSujin(), now)["source"] == "cache"

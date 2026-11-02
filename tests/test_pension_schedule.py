from copy import deepcopy
from unittest.mock import Mock

import pandas as pd
import pytest
from pension import market
from pension.market import TICKERS, PensionMarket, is_recently_closed, quote_payload
from pension.service import refresh_quotes, scheduled_refresh
from test_firebase_sujin import ManualStore


class CountingProvider:
    def __init__(self):
        self.calls = 0
        self.fail = False

    def quotes(self, now):
        self.calls += 1
        if self.fail:
            raise ValueError('조회 실패')
        session = market.current_session(now)
        stamp = min(now, session.market_close)
        return quote_payload(dict.fromkeys(TICKERS, 10000.0), dict.fromkeys(TICKERS, stamp.isoformat()), now, 'intraday')


@pytest.mark.parametrize('now', [
    '2026-11-01T01:00:00Z', # Sunday
    '2026-10-09T01:00:00Z', # KRX holiday
    '2026-11-02T00:00:00Z', # before KRX open (09:00 KST is 00:00 UTC; replaced below)
])
def test_skip_holiday_weekend_and_outside_session_without_fetch(now):
    # The final case is 08:59 KST on the Monday, not 09:00.
    if now == '2026-11-02T00:00:00Z':
        now = '2026-11-01T23:59:00Z'
    provider = CountingProvider()
    class NoStore:
        def get(self, _):
            pytest.fail('outside session must not read Firestore')
    assert scheduled_refresh(NoStore(), provider, pd.Timestamp(now))['source'] == 'skipped'
    assert provider.calls == 0


def test_40_krx_daily_slots_include_open_and_close_but_not_1540_or_1550():
    store, provider = ManualStore(), CountingProvider()
    start = pd.Timestamp('2026-11-02T00:00:00Z')
    result = [scheduled_refresh(store, provider, start + pd.Timedelta(minutes=10*i))['source'] for i in range(42)]
    assert result[:40] == ['refreshed'] * 40
    assert result[40:] == ['skipped', 'skipped']
    assert provider.calls == 40
    assert store.docs['market/pensionV2']['priceTimes']['0072R0'] == '2026-11-02T06:30:00+00:00'


def test_at_close_naver_uses_today_verified_regular_session_trade(monkeypatch):
    now = pd.Timestamp('2026-11-02T06:30:20Z')
    assert is_recently_closed(now) and not market.is_open(now)
    def get(url, **kwargs):
        assert url.endswith('/basic')
        code = url.split('/')[-2]
        return Mock(json=lambda: {'itemCode':code, 'marketStatus':'CLOSE', 'closePrice':'11,185', 'localTradedAt':'2026-11-02T15:30:00+09:00'}, raise_for_status=lambda:None)
    monkeypatch.setattr(market.requests, 'get', get)
    value = PensionMarket()._naver(now)
    assert value['quoteMode'] == 'intraday' and value['marketDate'] == '2026-11-02'
    assert set(value['priceTimes'].values()) == {'2026-11-02T06:30:00+00:00'}


@pytest.mark.parametrize('stamp', ['2026-10-30T15:30:00+09:00','2026-11-02T15:30:01+09:00'])
def test_close_rejects_previous_day_and_after_hours_timestamps(monkeypatch, stamp):
    now = pd.Timestamp('2026-11-02T06:30:20Z')
    def get(url, **kwargs):
        return Mock(json=lambda: {'itemCode':url.split('/')[-2], 'marketStatus':'CLOSE', 'closePrice':'11,185','localTradedAt':stamp}, raise_for_status=lambda:None)
    monkeypatch.setattr(market.requests, 'get', get)
    with pytest.raises(ValueError):
        PensionMarket()._naver(now)


def test_prior_day_fallback_does_not_overwrite_same_day_success():
    store, provider = ManualStore(), CountingProvider()
    now = pd.Timestamp('2026-11-02T03:00:00Z')
    good = scheduled_refresh(store, provider, now)['market']
    before = deepcopy(good)
    class OldClose:
        def quotes(self, when):
            stamp=market.last_close(when)[1]
            return quote_payload(dict.fromkeys(TICKERS, 9999.0), dict.fromkeys(TICKERS,stamp.isoformat()),when,'close')
    later = now + pd.Timedelta(minutes=10)
    result = scheduled_refresh(store, OldClose(), later)
    assert result['source']=='fallback' and store.docs['market/pensionV2']==before
    assert store.docs['status/pensionQuotesV2']['ok'] is False
    assert result['market']['updatedAt']==before['updatedAt']


def test_one_symbol_older_rejects_entire_response_for_manual_and_scheduled():
    store, provider = ManualStore(), CountingProvider()
    now = pd.Timestamp('2026-11-02T03:00:00Z')
    before = refresh_quotes(store, provider, now)['market']
    original = now
    class OneOlder:
        def quotes(self, when):
            stamps=dict.fromkeys(TICKERS,when.isoformat())
            stamps['0072R0']=(original-pd.Timedelta(minutes=1)).isoformat()
            return quote_payload(dict.fromkeys(TICKERS,12000.0),stamps,when,'intraday')
    for fn in [refresh_quotes, scheduled_refresh]:
        now += pd.Timedelta(minutes=10)
        result=fn(store,OneOlder(),now)
        assert result['source']=='fallback' and result['market']==before
        assert store.docs['market/pensionV2']==before
        assert store.docs['status/pensionQuotesV2']['ok'] is False


def test_automatic_and_manual_share_gate_and_failed_update_keeps_success():
    store, provider = ManualStore(), CountingProvider()
    now = pd.Timestamp('2026-11-02T03:00:00Z')
    before=refresh_quotes(store,provider,now)['market']
    assert scheduled_refresh(store,provider,now+pd.Timedelta(seconds=30))['source']=='cache'
    assert provider.calls==1
    provider.fail=True
    assert scheduled_refresh(store,provider,now+pd.Timedelta(minutes=10))['source']=='fallback'
    assert store.docs['market/pensionV2']==before
    assert store.docs['status/pensionQuotesV2']['ok'] is False


def test_invalid_old_postclose_timestamp_does_not_block_valid_session_close():
    now=pd.Timestamp('2026-11-02T06:30:20Z')
    store,provider=ManualStore(),CountingProvider()
    valid=provider.quotes(now)
    cached=deepcopy(valid)
    cached['priceTimes']['0072R0']='2026-11-02T06:30:45+00:00'
    store.docs['market/pensionV2']=cached
    result=scheduled_refresh(store,provider,now)
    assert result['source']=='refreshed'
    assert result['market']['priceTimes']['0072R0']=='2026-11-02T06:30:00+00:00'


def test_close_does_not_reuse_recent_previous_day_closing_cache():
    store,provider=ManualStore(),CountingProvider()
    now=pd.Timestamp('2026-11-02T06:30:20Z')
    before=market.last_close(now)[1]
    yesterday=quote_payload(dict.fromkeys(TICKERS,10000.0),dict.fromkeys(TICKERS,before.isoformat()),now,'close')
    store.docs['market/pensionV2']=yesterday
    result=scheduled_refresh(store,provider,now)
    assert result['source']=='refreshed' and provider.calls==1
    assert result['market']['marketDate']=='2026-11-02'
    assert result['market']['priceTimes']['0072R0']=='2026-11-02T06:30:00+00:00'

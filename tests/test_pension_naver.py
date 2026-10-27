from unittest.mock import Mock

import pandas as pd
import pytest
from pension import market
from pension.market import PensionMarket
from pension.service import refresh_quotes
from test_firebase_sujin import ManualStore


def test_naver_intraday_all_twelve_codes_and_timestamp(monkeypatch):
    now = pd.Timestamp('2026-10-27T01:00:00Z')
    def get(url, **kwargs):
        code = url.split('/')[-2]
        return Mock(json=lambda: {'itemCode':code, 'marketStatus':'OPEN', 'closePrice':'12,345', 'localTradedAt':'2026-10-27T09:59:00+09:00'}, raise_for_status=lambda:None)
    monkeypatch.setattr(market.requests, 'get', get)
    monkeypatch.setattr(market.yf, 'download', lambda *a, **k: pytest.fail('Yahoo should not run after Naver success'))
    result = PensionMarket().quotes(now)
    assert len(result['prices']) == 12 and result['prices']['0072R0'] == 12345
    assert result['source'] == 'Naver Finance'
    assert result['priceTimes']['0085P0'] == '2026-10-27T00:59:00+00:00'
    saved = refresh_quotes(ManualStore(), PensionMarket(), now)
    assert saved['market']['source'] == 'Naver Finance'


def test_preopen_daily_selects_completed_day_not_today_placeholder(monkeypatch):
    response = Mock(json=lambda:[{'localTradedAt':'2026-10-27','closePrice':'999'}, {'localTradedAt':'2026-10-26','closePrice':'12,345'}], raise_for_status=lambda:None)
    monkeypatch.setattr(market.requests,'get',lambda *a,**k:response)
    result = PensionMarket()._naver(pd.Timestamp('2026-10-26T23:55:00Z'))
    assert set(result['prices'].values()) == {12345}
    assert set(result['priceTimes'].values()) == {'2026-10-26T06:30:00+00:00'}


@pytest.mark.parametrize('patch',[{'itemCode':'OTHER'}, {'marketStatus':'PREOPEN'}, {'localTradedAt':'2026-10-26T15:30:00+09:00'}, {'localTradedAt':'2026-10-27T10:05:00+09:00'}, {'closePrice':'0'}])
def test_naver_rejects_wrong_code_old_future_or_invalid_quote(monkeypatch,patch):
    def get(url,**kwargs):
        data={'itemCode':url.split('/')[-2],'marketStatus':'OPEN','closePrice':'12345','localTradedAt':'2026-10-27T09:59:00+09:00',**patch}
        return Mock(json=lambda:data,raise_for_status=lambda:None)
    monkeypatch.setattr(market.requests,'get',get)
    with pytest.raises(ValueError):
        PensionMarket()._naver(pd.Timestamp('2026-10-27T01:00:00Z'))

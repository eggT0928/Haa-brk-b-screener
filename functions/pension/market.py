"""KRX 거래 달력과 실제 시세 시각을 검증한 Yahoo 국내 ETF 원시 가격."""

import logging
from functools import lru_cache

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf
from haa.calendar import utc
from haa.market import field, retry, validate_row

TICKERS = ["379800", "294400", "283580", "453810", "411060", "385560", "308620", "453850", "449170"]
SYMBOLS = [ticker + ".KS" for ticker in TICKERS]
KST = "Asia/Seoul"
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def calendar():
    return mcal.get_calendar("XKRX")


@lru_cache(maxsize=64)
def schedule(start, end):
    return calendar().schedule(start_date=start, end_date=end)


def current_session(now):
    day = utc(now).tz_convert(KST).date().isoformat()
    sessions = schedule(day, day)
    return None if sessions.empty else sessions.iloc[0]


def is_open(now):
    session = current_session(now)
    return bool(session is not None and session.market_open <= utc(now) < session.market_close)


def last_close(now):
    now = utc(now)
    day = now.tz_convert(KST).date()
    sessions = schedule(str(day - pd.Timedelta(days=40)), str(day))
    # 일봉 반영을 위해 폐장 후 60분 기다린다. 전일 종가를 오늘 가격으로 표시하지 않는다.
    closed = sessions.loc[sessions.market_close + pd.Timedelta(minutes=60) <= now]
    if closed.empty:
        raise ValueError("완료된 KRX 거래일을 찾을 수 없습니다.")
    return closed.index[-1], closed.iloc[-1].market_close


def quote_payload(prices, timestamps, now, mode):
    now = utc(now)
    if mode not in {"intraday", "close"} or set(prices) != set(TICKERS) or set(timestamps) != set(TICKERS):
        raise ValueError("연금 필수 종목의 시세 응답이 불완전합니다.")
    session = current_session(now) if mode == "intraday" else None
    close = last_close(now)[1] if mode == "close" else None
    cleaned, times = {}, {}
    for ticker in TICKERS:
        price = prices[ticker]
        if (
            isinstance(price, bool)
            or not isinstance(price, (int, float))
            or not np.isfinite(price)
            or not 0 < price <= 1e12
        ):
            raise ValueError(f"{ticker} 가격이 유효하지 않습니다.")
        stamp = utc(timestamps[ticker])
        if mode == "intraday":
            if (
                not is_open(now)
                or session is None
                or stamp < session.market_open
                or not -60 <= (now - stamp).total_seconds() <= 1500
            ):
                raise ValueError(f"{ticker} 장중 시세가 누락되었거나 오래되었습니다.")
        elif stamp != close:
            raise ValueError(f"{ticker} 시세가 최근 완료된 KRX 종가와 다릅니다.")
        cleaned[ticker], times[ticker] = float(price), stamp.isoformat()
    return {
        "prices": cleaned,
        "priceTimes": times,
        "updatedAt": now.isoformat(),
        "currency": "KRW",
        "quoteMode": mode,
        "marketDate": min(pd.Timestamp(v) for v in times.values()).tz_convert(KST).date().isoformat(),
        "source": "Yahoo Finance",
    }


class PensionMarket:
    def _download(self, opened):
        # 9종목을 한 번에 조회한다. 실패 재시도 2회와 각 요청 제한을 두어 수동 연속 호출 비용을 줄인다.
        return yf.download(
            SYMBOLS,
            period="1d" if opened else "3mo",
            interval="1m" if opened else "1d",
            prepost=False,
            auto_adjust=False,
            progress=False,
            threads=9,
            timeout=5,
        )

    def _intraday(self, now):
        close = field(self._download(True), "Close", SYMBOLS)
        if close.index.tz is None:
            raise ValueError("Yahoo 연금 장중 시세에 시간대가 없습니다.")
        prices, times = {}, {}
        for ticker, symbol in zip(TICKERS, SYMBOLS):
            series = close[symbol].dropna()
            if series.empty:
                raise ValueError(f"{ticker} 장중 시세가 없습니다.")
            prices[ticker], times[ticker] = float(series.iloc[-1]), utc(series.index[-1]).isoformat()
        return quote_payload(prices, times, now, "intraday")

    def _closing(self, now):
        day, stamp = last_close(now)
        close = field(self._download(False), "Close", SYMBOLS)
        close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
        validate_row(close, day, SYMBOLS)
        return quote_payload(
            {t: float(close.loc[day, s]) for t, s in zip(TICKERS, SYMBOLS)},
            dict.fromkeys(TICKERS, stamp.isoformat()),
            now,
            "close",
        )

    def quotes(self, now):
        now = utc(now)
        if is_open(now):
            try:
                return retry(lambda: self._intraday(now), attempts=2)
            except Exception:
                logger.warning("연금 Yahoo 장중 응답 불가. 최근 완료된 KRX 종가만 별도 표시합니다.")
        return retry(lambda: self._closing(now), attempts=2)

"""Yahoo 조회·검증·재시도. 잘못된 부분 응답은 성공값으로 저장하지 않는다."""

import logging
import time

import numpy as np
import pandas as pd
import yfinance as yf

from .calendar import ET, completed_month, latest_closed_session, schedule, utc
from .core import APP_TICKERS, STRATEGY_TICKERS

logger = logging.getLogger(__name__)


def retry(operation, attempts=3, sleep=time.sleep):
    for attempt in range(attempts):
        try:
            return operation()
        except Exception:
            if attempt == attempts - 1:
                raise
            logger.warning("Yahoo 조회 재시도 %s/%s", attempt + 1, attempts)
            sleep(2**attempt)


def field(frame, name, tickers):
    if frame is None or frame.empty or name not in frame.columns.get_level_values(0):
        raise ValueError(f"Yahoo {name} 응답이 비었습니다.")
    result = frame[name]
    if isinstance(result, pd.Series):
        result = result.to_frame(tickers[0])
    return result.reindex(columns=tickers).sort_index()


def validate_row(frame, day, tickers):
    if day not in frame.index:
        raise ValueError(f"{day.date()} 종가가 아직 없습니다.")
    values = frame.loc[day, tickers].to_numpy(dtype=float)
    invalid = [ticker for ticker, value in zip(tickers, values) if not np.isfinite(value) or value <= 0]
    if invalid:
        raise ValueError(f"{day.date()} Yahoo 필수 종가 누락: {', '.join(invalid)}")


class YahooMarket:
    def daily(self, now, full=False):
        tickers = APP_TICKERS + (["DBC", "^IRX"] if full else [])

        def fetch():
            kwargs = {"start": "2000-01-01"} if full else {"period": "2y"}
            frame = yf.download(
                tickers, auto_adjust=False, progress=False, threads=4, timeout=15, interval="1d", **kwargs
            )
            adjusted, raw = field(frame, "Adj Close", tickers), field(frame, "Close", tickers)
            for table in (adjusted, raw):
                table.index = pd.to_datetime(table.index).tz_localize(None).normalize()
            last = latest_closed_session(now)
            required = STRATEGY_TICKERS + (["DBC"] if full else [])
            validate_row(adjusted, last, required)
            validate_month_ends(adjusted, now, required)
            if full:
                # ^IRX 누락은 고정 2.5%로 감추지 않고 마지막 성공 캐시로 복구한다.
                rates = raw["^IRX"].dropna()
                if (
                    rates.empty
                    or not np.isfinite(rates).all()
                    or rates.index[-1] < last - pd.Timedelta(days=7)
                ):
                    raise ValueError("^IRX 수익률 데이터가 누락되었거나 오래되었습니다.")
                return adjusted.loc[:last], raw.loc[:last]
            return adjusted, raw

        return retry(fetch)

    def quotes(self, now):
        def fetch():
            frame = yf.download(
                APP_TICKERS,
                period="1d",
                interval="1m",
                prepost=False,
                auto_adjust=False,
                progress=False,
                threads=4,
                timeout=15,
            )
            close = field(frame, "Close", APP_TICKERS)
            if close.index.tz is None:
                raise ValueError("Yahoo 장중 시각의 시간대가 없습니다.")
            prices, timestamps = {}, {}
            for ticker in APP_TICKERS:
                series = close[ticker].dropna()
                if series.empty:
                    if ticker == "SPYM":
                        continue
                    raise ValueError(f"{ticker}의 장중 가격이 없습니다.")
                stamp = utc(series.index[-1])
                age = (utc(now) - stamp).total_seconds()
                if age < -60 or age > 25 * 60:
                    if ticker == "SPYM":
                        continue
                    raise ValueError(f"{ticker}의 Yahoo 가격 시각이 오래되었습니다.")
                value = float(series.iloc[-1])
                if not np.isfinite(value) or value <= 0:
                    raise ValueError(f"{ticker} 가격이 유효하지 않습니다.")
                prices[ticker], timestamps[ticker] = value, stamp.isoformat()
            return prices, timestamps

        return retry(fetch)


def closing_quotes(raw, now):
    day = latest_closed_session(now)
    prices, timestamps = {}, {}
    close_time = schedule(str(day.date()), str(day.date())).iloc[0].market_close.isoformat()
    for ticker in APP_TICKERS:
        value = raw.loc[day, ticker] if ticker in raw else np.nan
        if np.isfinite(value) and value > 0:
            prices[ticker], timestamps[ticker] = float(value), close_time
    return prices, timestamps


def quote_payload(prices, timestamps, now):
    return {
        "prices": prices,
        "priceTimes": timestamps,
        "updatedAt": utc(now).isoformat(),
        "marketDate": utc(now).tz_convert(ET).date().isoformat(),
        "source": "Yahoo Finance",
    }


def validate_month_ends(adjusted, now, tickers):
    """과거 월말 거래일 전체가 빠진 경우 전일 종가를 월말 종가로 오인하지 않는다."""
    completed, _, _ = completed_month(now)
    sessions = schedule(str(adjusted.index[0].date()), str(completed.end_time.date()))
    month_ends = sessions.groupby(sessions.index.to_period("M")).tail(1).index
    for ticker in tickers:
        first = adjusted[ticker].first_valid_index()
        if first is None:
            raise ValueError(f"{ticker} 가격 데이터가 없습니다.")
        eligible = month_ends[month_ends >= first]
        samples = adjusted[ticker].reindex(eligible)
        invalid = samples.isna() | ~np.isfinite(samples) | (samples <= 0)
        if invalid.any():
            day = samples.index[invalid][0].date()
            raise ValueError(f"{ticker}의 실제 마지막 거래일 {day} 종가가 누락되었습니다.")

"""미국 동부시간과 NYSE 휴장·조기폐장을 반영한 신호 경계."""

from functools import lru_cache

import pandas as pd
import pandas_market_calendars as mcal

ET = "America/New_York"


@lru_cache(maxsize=1)
def calendar():
    return mcal.get_calendar("NYSE")


def utc(value):
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("기준 시각에는 시간대가 필요합니다.")
    return timestamp.tz_convert("UTC")


@lru_cache(maxsize=64)
def schedule(start: str, end: str):
    return calendar().schedule(start_date=start, end_date=end)


def is_open(now):
    now = utc(now)
    day = now.tz_convert(ET).date().isoformat()
    sessions = schedule(day, day)
    return bool(len(sessions) and sessions.iloc[0].market_open <= now < sessions.iloc[0].market_close)


def completed_month(now, settlement_minutes=60):
    now = utc(now)
    month = now.tz_convert(ET).tz_localize(None).to_period("M")
    sessions = schedule(str(month.start_time.date()), str(month.end_time.date()))
    # 최종 거래일 종가 반영을 위해 최소 60분 기다리고, 일봉 존재 여부도 별도 확인한다.
    if now < sessions.iloc[-1].market_close + pd.Timedelta(minutes=settlement_minutes):
        month -= 1
        sessions = schedule(str(month.start_time.date()), str(month.end_time.date()))
    return month, sessions.index[-1], sessions.iloc[-1].market_close


def latest_closed_session(now):
    now = utc(now)
    day = now.tz_convert(ET).date()
    sessions = schedule(str(day - pd.Timedelta(days=15)), str(day))
    closed = sessions.loc[sessions.market_close + pd.Timedelta(minutes=60) <= now]
    if closed.empty:
        raise ValueError("완료된 거래일을 찾을 수 없습니다.")
    return closed.index[-1]

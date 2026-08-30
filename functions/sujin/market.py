"""HAA의 공통 시세를 재사용하고 수진에게 필요한 나머지 종목만 추가 조회한다."""

import numpy as np
import pandas as pd
import yfinance as yf
from haa.calendar import completed_month, is_open, latest_closed_session, schedule, utc
from haa.market import field, retry, validate_row

from .core import TICKERS


class SujinMarket:
    def quotes(self, now, base=None):
        opened = is_open(now)
        prices, times = {}, {}
        day = latest_closed_session(now) if not opened else None
        close_time = (
            schedule(str(day.date()), str(day.date())).iloc[0].market_close if day is not None else None
        )
        # 가격뿐 아니라 실제 기준시각까지 검증한 공통 종목만 재사용한다.
        for ticker in TICKERS:
            try:
                price = float(base["prices"][ticker])
                stamp = utc(base["priceTimes"][ticker])
                valid_time = (
                    -60 <= (utc(now) - stamp).total_seconds() <= 1500 if opened else stamp == close_time
                )
                if np.isfinite(price) and price > 0 and valid_time:
                    prices[ticker], times[ticker] = price, stamp.isoformat()
            except (KeyError, TypeError, ValueError):
                pass
        missing = [t for t in TICKERS if t not in prices]
        if not missing:
            return prices, times

        def fetch():
            frame = yf.download(
                missing,
                period="1d" if opened else "1mo",
                interval="1m" if opened else "1d",
                prepost=False,
                auto_adjust=False,
                progress=False,
                threads=4,
                timeout=10,
            )
            close = field(frame, "Close", missing)
            extra_prices, extra_times = {}, {}
            if not opened:
                close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
                validate_row(close, day, missing)
                return (
                    {**prices, **{t: float(close.loc[day, t]) for t in missing}},
                    {**times, **dict.fromkeys(missing, close_time.isoformat())},
                )
            if close.index.tz is None:
                raise ValueError("장중 시세에 시간대가 없습니다.")
            for ticker in missing:
                series = close[ticker].dropna()
                if series.empty:
                    raise ValueError(f"{ticker} 시세가 없습니다.")
                stamp, price = utc(series.index[-1]), float(series.iloc[-1])
                if (
                    not np.isfinite(price)
                    or price <= 0
                    or not -60 <= (utc(now) - stamp).total_seconds() <= 1500
                ):
                    raise ValueError(f"{ticker} 시세가 유효하지 않거나 오래되었습니다.")
                extra_prices[ticker], extra_times[ticker] = price, stamp.isoformat()
            return {**prices, **extra_prices}, {**times, **extra_times}

        return retry(fetch, attempts=2)

    def history(self, now):
        def fetch():
            _, cutoff, _ = completed_month(now)
            symbols = TICKERS + ["^IRX"]
            frame = yf.download(
                symbols,
                start="2000-01-01",
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=4,
                timeout=10,
            )
            adjusted, raw = field(frame, "Adj Close", symbols), field(frame, "Close", symbols)
            for table in (adjusted, raw):
                table.index = pd.to_datetime(table.index).tz_localize(None).normalize()
            adjusted = adjusted.loc[:cutoff, TICKERS]
            firsts = [adjusted[t].first_valid_index() for t in TICKERS]
            if any(day is None for day in firsts):
                raise ValueError("수진 백테스트 필수 종목을 받지 못했습니다.")
            sessions = schedule(str(max(firsts).date()), str(cutoff.date()))
            if sessions.empty:
                raise ValueError("완료된 공통 월말 이력이 없습니다.")
            last_days = sessions.groupby(sessions.index.to_period("M")).tail(1).index
            monthly = adjusted.reindex(last_days).copy()
            # 실제 마지막 거래일이 빠졌으면 전일 가격을 월말 가격으로 대입하지 않는다.
            if not np.isfinite(monthly.to_numpy()).all() or (monthly <= 0).any().any():
                raise ValueError("필수 종목의 실제 월말 거래일 가격이 누락되었습니다.")
            monthly.index = monthly.index.to_period("M").to_timestamp("M")
            rates = raw["^IRX"].loc[:cutoff].dropna()
            if rates.empty or not np.isfinite(rates).all() or rates.index[-1] < cutoff - pd.Timedelta(days=7):
                raise ValueError("^IRX 데이터가 누락되었거나 오래되었습니다.")
            return monthly, rates

        return retry(fetch, attempts=2)

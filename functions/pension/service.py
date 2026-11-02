"""정규장 자동·수동으로 갱신하는 독립 원화 시세 캐시."""

import logging

import numpy as np
import pandas as pd
from haa.calendar import utc
from sujin.service import RefreshBusy

from .market import TICKERS, current_session, is_open, is_recently_closed, last_close, quote_payload

logger = logging.getLogger(__name__)


def recent_quotes(cached, now):
    if not cached or cached.get("currency") != "KRW":
        return False
    try:
        if any(not np.isfinite(cached["prices"][t]) or cached["prices"][t] <= 0 for t in TICKERS):
            return False
        stamps = [utc(cached["priceTimes"][t]) for t in TICKERS]
        if is_open(now) or is_recently_closed(now):
            recent = 0 <= (now - utc(cached["updatedAt"])).total_seconds() < 60
            if cached.get("quoteMode") == "intraday":
                session = current_session(now)
                return bool(session is not None and recent and all(
                    session.market_open <= stamp <= session.market_close
                    and -60 <= (now - stamp).total_seconds() <= 1500 for stamp in stamps
                ))
            if is_recently_closed(now):
                # 마감 작업에서 60초 이내의 전일 종가를 새 당일값으로 재사용하지 않는다.
                return False
            return (
                recent
                and cached.get("quoteMode") == "close"
                and all(stamp == last_close(now)[1] for stamp in stamps)
            )
        return cached.get("quoteMode") == "close" and all(stamp == last_close(now)[1] for stamp in stamps)
    except (KeyError, TypeError, ValueError):
        return False


def scheduled_refresh(store, provider, now):
    """실제 KRX 세션에만 갱신한다. 15:30 실행 지연은 3분까지 허용한다."""
    now = utc(now)
    session = current_session(now)
    if session is None or not session.market_open <= now <= session.market_close + pd.Timedelta(minutes=3):
        return {"source": "skipped", "message": "KRX 정규장 시간이 아닙니다."}
    try:
        return refresh_quotes(store, provider, now)
    except RefreshBusy:
        # 다른 자동·수동 조회가 진행 중이면 중복 작업을 시작하지 않는다.
        return {"source": "busy", "message": "다른 연금 시세 조회가 진행 중입니다."}


def older_quote(value, cached):
    """한 종목이라도 기존 성공 시각보다 과거면 전체 응답을 버린다."""
    if not cached:
        return False
    for ticker in TICKERS:
        try:
            previous = utc(cached["priceTimes"][ticker])
            received = utc(value["priceTimes"][ticker])
            if received < previous:
                session = current_session(previous)
                # 기존 코드가 허용했던 장 종료 후·미래 시각은 신뢰할 수 없는 캐시다.
                if session is not None and session.market_open <= previous <= session.market_close:
                    return True
        except (KeyError, ValueError, TypeError):
            # 오염되거나 누락된 캐시 필드는 비교하지 않고 새 정상 응답으로 복구한다.
            continue
    return False


def refresh_quotes(store, provider, now):
    now = utc(now)
    path, gate = "market/pensionV2", "pensionQuotesV2"
    cached = store.get(path)
    if recent_quotes(cached, now):
        return {
            "market": cached,
            "source": "cache",
            "message": "저장된 최근 연금 시세를 재사용했습니다. 가격 기준시각을 확인하세요.",
        }
    token = store.acquire_manual(gate, now)
    if not token:
        if cached:
            return {
                "market": cached,
                "source": "fallback",
                "message": "갱신 중이거나 재시도 대기 중입니다. 직전 성공값을 표시합니다.",
            }
        raise RefreshBusy("연금 첫 시세 조회 중이거나 재시도 대기 중입니다. 1분 후 다시 눌러 주세요.")
    try:
        cached = store.get(path) or cached
        value = provider.quotes(now)
        source = value.get("source", "Yahoo Finance")
        value = quote_payload(value["prices"], value["priceTimes"], now, value["quoteMode"])
        value["source"] = source if source in {"Naver Finance", "Yahoo Finance"} else "온라인 시세"
        if older_quote(value, cached):
            raise ValueError("조회한 연금 시세가 마지막 성공값보다 오래되었습니다.")
        store.commit(
            {
                path: value,
                f"status/{gate}": {
                    "ok": True,
                    "lastAttemptAt": now.isoformat(),
                    "message": "연금 시세 갱신 성공",
                },
            }
        )
        message = (
            f"{value['source']} 장중 가격을 받았습니다."
            if value["quoteMode"] == "intraday"
            else "최근 완료된 KRX 거래일 종가를 받았습니다. 실시간 가격은 아닙니다."
        )
        return {"market": value, "source": "refreshed", "message": message}
    except Exception:
        logger.exception("연금 시세 조회 실패. 직전 성공값과 시세 기준시각을 유지합니다.")
        store.commit(
            {
                f"status/{gate}": {
                    "ok": False,
                    "lastAttemptAt": now.isoformat(),
                    "message": "연금 시세 조회 실패. 직전 성공값을 사용합니다.",
                }
            }
        )
        if cached:
            return {
                "market": cached,
                "source": "fallback",
                "message": "온라인 조회 실패로 직전 연금 시세를 표시합니다. 기준시각을 확인하세요.",
            }
        raise LookupError(
            "연금 시세 조회에 실패했고 저장된 성공값도 없습니다. 잠시 후 다시 시도하세요."
        ) from None
    finally:
        store.release_manual(gate, token)

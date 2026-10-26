"""사용자가 요청할 때만 갱신하는 원화 시세 캐시. 보유수량·HAA·수진은 변경하지 않는다."""

import logging

import numpy as np
from haa.calendar import utc
from sujin.service import RefreshBusy

from .market import TICKERS, is_open, last_close, quote_payload

logger = logging.getLogger(__name__)


def recent_quotes(cached, now):
    if not cached or cached.get("currency") != "KRW":
        return False
    try:
        if any(not np.isfinite(cached["prices"][t]) or cached["prices"][t] <= 0 for t in TICKERS):
            return False
        stamps = [utc(cached["priceTimes"][t]) for t in TICKERS]
        if is_open(now):
            recent = 0 <= (now - utc(cached["updatedAt"])).total_seconds() < 600
            if cached.get("quoteMode") == "intraday":
                return recent and all(-60 <= (now - stamp).total_seconds() <= 1500 for stamp in stamps)
            return (
                recent
                and cached.get("quoteMode") == "close"
                and all(stamp == last_close(now)[1] for stamp in stamps)
            )
        return cached.get("quoteMode") == "close" and all(stamp == last_close(now)[1] for stamp in stamps)
    except (KeyError, TypeError, ValueError):
        return False


def refresh_quotes(store, provider, now):
    now = utc(now)
    path, gate = "market/pension", "pensionQuotes"
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
        value = quote_payload(value["prices"], value["priceTimes"], now, value["quoteMode"])
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
            "Yahoo 장중 가격을 받았습니다."
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
                "message": "Yahoo 조회 실패로 직전 연금 시세를 표시합니다. 기준시각을 확인하세요.",
            }
        raise LookupError(
            "연금 시세 조회에 실패했고 저장된 성공값도 없습니다. 잠시 후 다시 시도하세요."
        ) from None
    finally:
        store.release_manual(gate, token)

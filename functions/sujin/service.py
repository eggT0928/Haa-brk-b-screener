"""기존 자동 작업과 수동 요청이 공유하는 수진 캐시. HAA 문서·잠금과 분리한다."""

import io
import json
import logging
import zlib

import pandas as pd
from haa.calendar import completed_month, is_open, latest_closed_session, schedule, utc
from haa.market import quote_payload

from .core import TICKERS, backtest

logger = logging.getLogger(__name__)


class RefreshBusy(Exception):
    pass


def pack(monthly, rates):
    data = {
        "monthly": monthly.to_json(orient="split", date_format="iso"),
        "rates": rates.to_frame("IRX").to_json(orient="split", date_format="iso"),
    }
    blob = zlib.compress(json.dumps(data).encode(), level=6)
    if len(blob) > 900_000:
        raise ValueError("수진 가격 캐시가 안전 한도를 초과했습니다.")
    return blob


def unpack(blob):
    data = json.loads(zlib.decompress(blob))
    monthly = pd.read_json(io.StringIO(data["monthly"]), orient="split")
    rates = pd.read_json(io.StringIO(data["rates"]), orient="split")
    monthly.index, rates.index = pd.to_datetime(monthly.index), pd.to_datetime(rates.index)
    return monthly, rates.IRX


def recent_quotes(cached, now):
    if not cached:
        return False
    try:
        stamps = [utc(cached["priceTimes"][t]) for t in TICKERS]
        if is_open(now):
            return 0 <= (now - utc(cached["updatedAt"])).total_seconds() < 600 and all(
                -60 <= (now - stamp).total_seconds() <= 1500 for stamp in stamps
            )
        day = latest_closed_session(now)
        close = schedule(str(day.date()), str(day.date())).iloc[0].market_close
        return all(stamp == close for stamp in stamps)
    except (KeyError, ValueError):
        return False


def refresh_cached(store, provider, now, kind):
    now = utc(now)
    quotes = kind == "quotes"
    path = "market/sujin" if quotes else "internal/sujinHistory"
    cached = store.get(path)
    fresh = (
        recent_quotes(cached, now)
        if quotes
        else bool(
            cached
            and cached.get("cutoff") == str(completed_month(now)[0])
            and 0 <= (now - utc(cached["updatedAt"])).total_seconds() < 12 * 3600
        )
    )
    if fresh:
        return cached, "cache", "최근 성공 데이터를 재사용했습니다. 시세 기준시각을 확인하세요."
    gate = "sujinQuotes" if quotes else "sujinHistory"
    token = store.acquire_manual(gate, now)
    if not token:
        if cached:
            return cached, "fallback", "갱신 중이거나 잠시 대기해야 합니다. 직전 성공값을 표시합니다."
        raise RefreshBusy("첫 조회가 진행 중이거나 재시도 대기 중입니다. 1분 후 다시 눌러 주세요.")
    try:
        # 잠금 직전에 다른 요청이 갱신했을 수도 있으므로 새 성공값을 재확인한다.
        cached = store.get(path) or cached
        if quotes:
            prices, times = provider.quotes(now, base=store.get("market/current"))
            value = quote_payload(prices, times, now)
        else:
            monthly, rates = provider.history(now)
            backtest(monthly, rates)
            value = {
                "payload": pack(monthly, rates),
                "updatedAt": now.isoformat(),
                "cutoff": str(completed_month(now)[0]),
            }
        store.commit(
            {
                path: value,
                f"status/{gate}": {"ok": True, "lastAttemptAt": now.isoformat(), "message": "갱신 성공"},
            }
        )
        return value, "refreshed", "새 Yahoo 데이터를 받았습니다."
    except Exception:
        logger.exception("수진 %s 갱신 실패: 직전 성공값 유지", kind)
        store.commit(
            {
                f"status/{gate}": {
                    "ok": False,
                    "lastAttemptAt": now.isoformat(),
                    "message": "갱신 실패. 직전 성공값을 사용합니다.",
                }
            }
        )
        if cached:
            return cached, "fallback", "Yahoo 조회 실패로 직전 성공값을 표시합니다. 기준시각을 확인하세요."
        raise LookupError(
            "수진 데이터 조회에 실패했고 저장된 성공값도 없습니다. 잠시 후 다시 시도하세요."
        ) from None
    finally:
        store.release_manual(gate, token)


def refresh_quotes(store, provider, now):
    cached, source, message = refresh_cached(store, provider, now, "quotes")
    return {"market": cached, "source": source, "message": message}


def run_backtest(store, provider, now, start=None, end=None, initial=10000.0):
    cached, source, message = refresh_cached(store, provider, now, "history")
    monthly, rates = unpack(cached["payload"])
    result = backtest(monthly, rates, start, end, initial)
    result["dataUpdatedAt"] = cached["updatedAt"]
    result["source"] = source
    if source == "fallback":
        result["warnings"].append(message)
    return result


def refresh_with_haa(store, haa_provider, sujin_provider, now, job):
    from haa.service import refresh

    outcome = refresh(store, haa_provider, now, job)
    if outcome != "updated":
        return outcome
    # HAA의 성공 발행은 이미 완료됐다. 수진 전용 조회 실패가 이를 되돌리거나 재실행하지 않는다.
    try:
        refresh_quotes(store, sujin_provider, now)
    except Exception:
        logger.exception("HAA 갱신 성공. 수진 추가 갱신만 실패하여 다음 실행에서 재시도합니다.")
    return outcome

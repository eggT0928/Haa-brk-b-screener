"""사용자 조회와 독립적으로 공용 신호를 계산하고 원자적으로 발행한다."""

import logging

from .calendar import completed_month, is_open, utc
from .core import STRATEGY_TICKERS, backtest, preview, signal
from .market import closing_quotes, quote_payload, validate_row
from .store import pack, unpack

logger = logging.getLogger(__name__)


def refresh(store, provider, now, job="intraday"):
    now = utc(now)
    if job == "intraday" and not is_open(now):
        return "closed"
    run_key = f"{job}:{now.floor('15min').isoformat()}"
    token = store.acquire(run_key, now)
    if token is None:
        return "duplicate"
    success = False
    try:
        full = job == "daily"
        adjusted, raw = provider.daily(now, full=full)
        month, session, close_time = completed_month(now)
        docs = {}
        if full:
            validate_row(adjusted, session, STRATEGY_TICKERS)
            confirmed = signal(adjusted.loc[:session], month, "confirmed", close_time.isoformat())
            confirmed.update({"updatedAt": now.isoformat(), "sessionDate": session.date().isoformat()})
            current = store.get("signals/confirmed")
            if current is None or current["month"] < str(month):
                docs["signals/confirmed"] = confirmed
                # 한 달에 한 번만 확정본을 저장하며 일일 갱신으로 과거를 덮어쓰지 않는다.
                if not store.get(f"signalHistory/{month}"):
                    docs[f"signalHistory/{month}"] = confirmed
            docs["internal/history"] = {
                "payload": pack(adjusted, raw),
                "updatedAt": now.isoformat(),
                "cutoff": session.date().isoformat(),
            }
            prices, timestamps = closing_quotes(raw, now)
        else:
            prices, timestamps = provider.quotes(now)
        docs["market/current"] = quote_payload(prices, timestamps, now)
        # 월초 휴장일에는 전월 가격으로 새 달 예상신호를 만들지 않는다.
        quote_months = {
            utc(timestamps[t]).tz_convert("America/New_York").strftime("%Y-%m") for t in STRATEGY_TICKERS
        }
        if quote_months == {now.tz_convert("America/New_York").strftime("%Y-%m")}:
            expected = preview(adjusted, prices, now.isoformat())
            expected.update(
                {
                    "updatedAt": now.isoformat(),
                    "priceTimes": timestamps,
                    "oldestPriceAt": min(timestamps[t] for t in STRATEGY_TICKERS),
                }
            )
            docs["signals/preview"] = expected
        docs[f"status/{job}"] = {
            "lastAttemptAt": now.isoformat(),
            "lastSuccessAt": now.isoformat(),
            "ok": True,
            "message": "갱신 성공",
        }
        store.commit(docs)
        success = True
        return "updated"
    except Exception:
        logger.exception("%s 신호 갱신 실패: 마지막 성공값 유지", job)
        store.failure(job, now)
        # Scheduler도 실패를 인식하여 제한된 재시도를 수행하게 한다.
        raise
    finally:
        store.release(token, run_key, success)


def cached_backtest(store, start=None, end=None, initial=10000.0):
    cached = store.get("internal/history")
    if not cached:
        raise LookupError("장기 가격 캐시가 없습니다. 관리자가 일일 갱신을 먼저 실행해야 합니다.")
    adjusted, raw = unpack(cached["payload"])
    result = backtest(adjusted, raw["^IRX"], cached["cutoff"], start, end, initial)
    result["dataUpdatedAt"] = cached["updatedAt"]
    return result

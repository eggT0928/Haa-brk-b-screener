"""배포 진입점. 예약 작업 2개와 인증된 백테스트 API 1개만 운영한다."""

import json
import logging
import os
import re

import firebase_admin
import pandas as pd
from firebase_admin import auth
from firebase_functions import https_fn, options, scheduler_fn
from haa.market import YahooMarket
from haa.service import cached_backtest, refresh
from haa.store import create_store

firebase_admin.initialize_app()
options.set_global_options(
    region="us-central1",
    min_instances=0,
    max_instances=1,
    memory=options.MemoryOption.MB_512,
    cpu="gcf_gen1",
    concurrency=1,
)


def store():
    return create_store()


@scheduler_fn.on_schedule(
    schedule="*/15 9-15 * * 1-5",
    timezone="America/New_York",
    timeout_sec=480,
    retry_count=1,
    min_backoff_seconds=60,
)
def refresh_intraday(event: scheduler_fn.ScheduledEvent) -> None:
    # 09:00/09:15와 휴장일은 달력 검사 후 Yahoo에 접속하지 않고 종료한다.
    refresh(store(), YahooMarket(), pd.Timestamp.now(tz="UTC"), "intraday")


@scheduler_fn.on_schedule(
    schedule="30 17 * * 1-5",
    timezone="America/New_York",
    timeout_sec=480,
    retry_count=2,
    min_backoff_seconds=120,
)
def refresh_daily(event: scheduler_fn.ScheduledEvent) -> None:
    # 일일 장기 캐시 갱신과 월말 확정을 한 작업으로 결합한다. 다음 평일 재실행도 복구 가능하다.
    refresh(store(), YahooMarket(), pd.Timestamp.now(tz="UTC"), "daily")


def response(payload, status=200):
    return https_fn.Response(
        json.dumps(payload, ensure_ascii=False, allow_nan=False),
        status=status,
        content_type="application/json; charset=utf-8",
        headers={"Cache-Control": "private, no-store"},
    )


@https_fn.on_request(timeout_sec=60, invoker="public")
def api(req: https_fn.Request) -> https_fn.Response:
    # Hosting은 공개 진입점이지만 데이터 요청은 반드시 Firebase 사용자 토큰으로 검증한다.
    if req.method != "GET":
        return response({"error": "GET 요청만 지원합니다."}, 405)
    bearer = req.headers.get("Authorization", "")
    if not bearer.startswith("Bearer "):
        return response({"error": "로그인이 필요합니다."}, 401)
    try:
        decoded = auth.verify_id_token(bearer[7:], check_revoked=True)
    except (ValueError, auth.InvalidIdTokenError, auth.RevokedIdTokenError, auth.UserDisabledError):
        return response({"error": "로그인이 만료되었거나 유효하지 않습니다."}, 401)
    try:
        db = store()
        access = db.get(f"access/{decoded['uid']}")
        if not access or access.get("enabled") is not True:
            return response({"error": "관리자의 사용자 승인이 필요합니다."}, 403)
        if os.getenv("ENFORCE_APP_CHECK", "false").lower() == "true":
            from firebase_admin import app_check

            try:
                app_check.verify_token(req.headers.get("X-Firebase-AppCheck", ""))
            except Exception:
                return response({"error": "앱 검증에 실패했습니다."}, 403)
        if req.path not in ("/api/backtest", "/backtest"):
            return response({"error": "요청한 경로가 없습니다."}, 404)
        args = {key: req.args.get(key) for key in ("start", "end")}
        for date in args.values():
            if date and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
                raise ValueError("날짜는 YYYY-MM-DD 형식이어야 합니다.")
        result = cached_backtest(db, **args, initial=float(req.args.get("initial", "10000")))
        return response(result)
    except ValueError as exc:
        return response({"error": str(exc)}, 400)
    except LookupError as exc:
        return response({"error": str(exc)}, 503)
    except Exception:
        logging.exception("백테스트 API 오류")
        return response({"error": "서버에서 처리하지 못했습니다. 잠시 후 다시 시도하세요."}, 500)

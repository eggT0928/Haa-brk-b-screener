"""배포 진입점. 공유 예약 작업 2개와 인증된 백테스트·수동 갱신 API 1개."""

import json
import logging
import math
import os
import re

import firebase_admin
import pandas as pd
from firebase_admin import auth
from firebase_functions import https_fn, options, scheduler_fn
from haa import access as family_access
from haa.market import YahooMarket
from haa.service import cached_backtest
from haa.store import create_store
from sujin.market import SujinMarket
from sujin.service import RefreshBusy, refresh_quotes, refresh_with_haa
from sujin.service import run_backtest as sujin_backtest

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
    refresh_with_haa(store(), YahooMarket(), SujinMarket(), pd.Timestamp.now(tz="UTC"), "intraday")


@scheduler_fn.on_schedule(
    schedule="30 17 * * 1-5",
    timezone="America/New_York",
    timeout_sec=480,
    retry_count=2,
    min_backoff_seconds=120,
)
def refresh_daily(event: scheduler_fn.ScheduledEvent) -> None:
    # 일일 장기 캐시 갱신과 월말 확정을 한 작업으로 결합한다. 다음 평일 재실행도 복구 가능하다.
    refresh_with_haa(store(), YahooMarket(), SujinMarket(), pd.Timestamp.now(tz="UTC"), "daily")


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
    sujin_route = req.path in (
        "/api/sujin/refresh",
        "/sujin/refresh",
        "/api/sujin/backtest",
        "/sujin/backtest",
    )
    access_route = req.path.removeprefix("/api") in ("/access/session", "/access/request", "/access/manage")
    expected_method = "POST" if sujin_route or (access_route and not req.path.endswith("/manage")) else "GET"
    if req.method != expected_method and not (
        access_route and req.path.endswith("/manage") and req.method == "POST"
    ):
        return response({"error": "이 경로에서 지원하지 않는 요청 방식입니다."}, 405)
    bearer = req.headers.get("Authorization", "")
    if not bearer.startswith("Bearer "):
        return response({"error": "로그인이 필요합니다."}, 401)
    try:
        decoded = auth.verify_id_token(bearer[7:], check_revoked=True)
    except (ValueError, auth.InvalidIdTokenError, auth.RevokedIdTokenError, auth.UserDisabledError):
        return response({"error": "로그인이 만료되었거나 유효하지 않습니다."}, 401)
    try:
        db = store()
        if os.getenv("ENFORCE_APP_CHECK", "false").lower() == "true":
            from firebase_admin import app_check

            try:
                app_check.verify_token(req.headers.get("X-Firebase-AppCheck", ""))
            except Exception:
                return response({"error": "앱 검증에 실패했습니다."}, 403)
        if access_route:
            if req.content_length and req.content_length > 2048:
                return response({"error": "요청 내용이 너무 큽니다."}, 413)
            repo, now = family_access.FamilyRepository(db.db), pd.Timestamp.now(tz="UTC")
            if req.path.endswith("/session"):
                return response(family_access.session(repo, decoded, now))
            if req.path.endswith("/request"):
                return response(family_access.request_access(repo, decoded, now))
            if req.method == "GET":
                return response(family_access.management_list(repo, decoded))
            return response(
                family_access.manage(repo, decoded, req.get_json(silent=True), now, auth.get_user)
            )
        access = db.get(f"access/{decoded['uid']}")
        if not access or access.get("enabled") is not True:
            return response({"error": "관리자의 사용자 승인이 필요합니다."}, 403)
        if sujin_route:
            if req.content_length and req.content_length > 4096:
                return response({"error": "요청 내용이 너무 큽니다."}, 413)
            now = pd.Timestamp.now(tz="UTC")
            if req.path.endswith("/refresh"):
                return response(refresh_quotes(db, SujinMarket(), now))
            data = req.get_json(silent=True)
            if not isinstance(data, dict) or set(data) - {"start", "end", "initial"}:
                raise ValueError("백테스트 요청 형식을 확인하세요.")
            args = {key: data.get(key) for key in ("start", "end")}
            for date in args.values():
                if date is not None and (
                    not isinstance(date, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date)
                ):
                    raise ValueError("날짜는 YYYY-MM-DD 형식이어야 합니다.")
                if date:
                    pd.Timestamp(date)
            if args["start"] and args["end"] and args["start"] > args["end"]:
                raise ValueError("시작일은 종료일보다 늦을 수 없습니다.")
            initial = data.get("initial", 10000)
            if (
                isinstance(initial, bool)
                or not isinstance(initial, (int, float))
                or not math.isfinite(initial)
                or not 0 < initial <= 1e12
            ):
                raise ValueError("초기 금액을 확인하세요.")
            return response(sujin_backtest(db, SujinMarket(), now, **args, initial=initial))
        if req.path not in ("/api/backtest", "/backtest"):
            return response({"error": "요청한 경로가 없습니다."}, 404)
        args = {key: req.args.get(key) for key in ("start", "end")}
        for date in args.values():
            if date and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
                raise ValueError("날짜는 YYYY-MM-DD 형식이어야 합니다.")
        result = cached_backtest(db, **args, initial=float(req.args.get("initial", "10000")))
        return response(result)
    except PermissionError as exc:
        return response({"error": str(exc)}, 403)
    except RefreshBusy as exc:
        result = response({"error": str(exc)}, 429)
        result.headers["Retry-After"] = "60"
        return result
    except ValueError as exc:
        return response({"error": str(exc)}, 400)
    except LookupError as exc:
        return response({"error": str(exc)}, 503)
    except Exception:
        logging.exception("포트폴리오 API 오류")
        return response({"error": "서버에서 처리하지 못했습니다. 잠시 후 다시 시도하세요."}, 500)

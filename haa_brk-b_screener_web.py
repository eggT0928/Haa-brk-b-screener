import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import json
import base64
from datetime import datetime, timedelta
from pandas.tseries.offsets import MonthEnd, YearEnd
import plotly.graph_objects as go
import plotly.express as px
import streamlit.components.v1 as components


STRATEGY_TICKERS = ["SPY", "VEA", "VWO", "IWM", "BIL", "IEF", "TLT", "TIP", "PDBC", "VNQ", "BRK-B"]
APP_TICKERS = STRATEGY_TICKERS + ["SPYM"]
SP500_REBALANCE_OPTIONS = ["SPY", "SPYM"]
MOMENTUM_PERIODS = (1, 3, 6, 12)


def get_rebalance_ticker(asset: str, sp500_rebalance_ticker: str) -> str:
    """리밸런싱 실행용 티커 반환 (신호/백테스트는 SPY 기준 유지)"""
    return sp500_rebalance_ticker if asset == "SPY" else asset


def get_price_with_fallback(pricing_data: pd.DataFrame, strategy_data: pd.DataFrame, target_date: pd.Timestamp,
                            asset: str, sp500_rebalance_ticker: str) -> float:
    """가장 최근 실행용 티커 가격을 사용하고, 없으면 전략 티커 가격으로 대체."""
    exec_ticker = get_rebalance_ticker(asset, sp500_rebalance_ticker)
    if exec_ticker in pricing_data.columns:
        prices = pricing_data[exec_ticker].dropna()
        if not prices.empty:
            return float(prices.iloc[-1])

    prices = strategy_data[asset].dropna()
    if prices.empty:
        raise ValueError(f"{exec_ticker} 가격 데이터가 없습니다.")
    return float(prices.iloc[-1])


def format_selected_asset_label(asset: str, sp500_rebalance_ticker: str) -> str:
    """선택 자산 표시용 라벨 반환"""
    return get_rebalance_ticker(asset, sp500_rebalance_ticker)


def get_last_completed_month_end(as_of: pd.Timestamp = None) -> pd.Timestamp:
    """아직 진행 중인 달을 제외한 가장 최근 달의 말일을 반환."""
    as_of = pd.Timestamp.now() if as_of is None else pd.Timestamp(as_of)
    as_of = as_of.tz_localize(None) if as_of.tzinfo is not None else as_of
    return as_of.normalize().replace(day=1) - pd.Timedelta(days=1)


def get_month_end_prices(data: pd.DataFrame) -> pd.DataFrame:
    """일별 수정주가를 달력상 월말 라벨의 월말 종가로 변환."""
    if data.empty:
        return data.copy()
    return data.sort_index().resample(MonthEnd()).last().dropna(how="all")


def calculate_momentum_returns(data: pd.DataFrame) -> dict:
    """HAA 13612U에 사용하는 월말 기준 1·3·6·12개월 총수익률."""
    monthly_prices = get_month_end_prices(data)
    return {
        months: monthly_prices.pct_change(periods=months, fill_method=None)
        for months in MOMENTUM_PERIODS
    }


def calculate_momentum_scores(data: pd.DataFrame) -> pd.DataFrame:
    """완료된 월말 수정주가로 HAA 13612U 모멘텀 점수를 계산."""
    period_returns = calculate_momentum_returns(data)
    score = period_returns[MOMENTUM_PERIODS[0]].copy()
    for months in MOMENTUM_PERIODS[1:]:
        score = score.add(period_returns[months])

    # 네 기간이 모두 존재하는 달만 유효한 신호로 사용한다.
    return (score / len(MOMENTUM_PERIODS)).dropna(how="all")


def calculate_preview_momentum(data: pd.DataFrame, as_of: pd.Timestamp = None):
    """현재 가격을 당월의 가상 월말 가격으로 사용한 다음 리밸런싱 예상 신호.

    21·63·126·252 거래일 근사값을 사용하지 않고, 당월 월말 라벨에서
    정확히 1·3·6·12개월 전의 월말 수정주가를 비교 기준으로 사용한다.
    실제 월말이 끝나면 이 값은 새 확정 월말 신호와 일치한다.
    """
    if data.empty:
        raise ValueError("예상 신호를 계산할 가격 데이터가 없습니다.")

    clean = data.sort_index().copy()
    clean.index = pd.to_datetime(clean.index).tz_localize(None)
    requested_date = clean.index[-1] if as_of is None else pd.Timestamp(as_of)
    requested_date = requested_date.tz_localize(None) if requested_date.tzinfo is not None else requested_date
    available = clean.loc[clean.index <= requested_date].dropna(how="all")
    if available.empty:
        raise ValueError("예상 기준일 이전의 가격 데이터가 없습니다.")

    preview_date = available.index[-1]
    preview_month_end = preview_date + MonthEnd(0)
    monthly_prices = get_month_end_prices(available)
    current_prices = monthly_prices.loc[preview_month_end]

    period_returns = {}
    for months in MOMENTUM_PERIODS:
        base_month_end = preview_month_end - pd.DateOffset(months=months)
        if base_month_end not in monthly_prices.index:
            raise ValueError(f"{months}개월 예상 신호 기준 월말 데이터가 없습니다.")
        period_returns[months] = current_prices / monthly_prices.loc[base_month_end] - 1

    period_frame = pd.DataFrame(period_returns)
    preview_scores = period_frame.mean(axis=1, skipna=False)
    return period_returns, preview_scores, preview_date, preview_month_end


def build_rank_labels(scores: pd.Series) -> pd.Series:
    """확정·예상 점수를 같은 규칙으로 비교하기 위한 표시용 순위."""
    labels = pd.Series("", index=scores.index, dtype=object)
    offense = ["SPY", "VEA", "VWO", "IWM", "TLT", "PDBC", "VNQ", "IEF"]
    defense = ["IEF", "BIL"]

    for i, ticker in enumerate(scores[offense].nlargest(4).index, 1):
        labels.loc[ticker] = f"공격{i}위"
    for i, ticker in enumerate(scores[defense].nlargest(1).index, 1):
        labels.loc[ticker] = f"방어{i}위"

    labels.loc["TIP"] = "공격" if scores.get("TIP", 0) > 0 else "대피"
    if "BRK-B" in labels.index:
        labels.loc["BRK-B"] = "보유"
    return labels


def select_assets(momentum_scores: pd.DataFrame, data: pd.DataFrame, target_date: pd.Timestamp = None):
    """TIP 기준으로 자산 선택 (offense/defense)"""
    # target_date가 없으면 가장 마지막 인덱스를 사용
    if target_date is None:
        target_date = momentum_scores.index[-1]

    # target_date가 인덱스에 없으면 가장 가까운 이전 날짜를 찾음
    if target_date not in momentum_scores.index:
        available_dates = momentum_scores.index[momentum_scores.index <= target_date]
        if len(available_dates) > 0:
            target_date = available_dates[-1]
        else:
            # 사용 가능한 날짜가 없으면 첫 번째 날짜 사용
            target_date = momentum_scores.index[0]

    scores = momentum_scores.loc[target_date]

    tip_score = scores.get("TIP", 0)

    if tip_score > 0:
        # TIP이 양수면 공격 자산군에서 상위 4개 선택
        # IEF도 공격 자산군에 포함
        offense = ["SPY", "VEA", "VWO", "IWM", "TLT", "PDBC", "VNQ", "IEF"]
        # 해당 자산 중 모멘텀 점수 > 0인 것만 추려 상위 4개
        selected = list(scores[offense][scores[offense] > 0].nlargest(4).items())
    else:
        # TIP이 비양수면 방어 자산군에서 상위 1개 선택
        # IEF도 방어 자산군에 포함
        defense = ["IEF", "BIL"]
        selected = list(scores[defense].nlargest(1).items())

    return selected, target_date


def run_screener(total_balance: float, sp500_rebalance_ticker: str = "SPY"):
    """스크리너 실행
    - 리밸런싱 실행은 SPY 또는 SPYM 중 사용자가 선택
    - 모멘텀 계산과 백테스트는 항상 SPY 기준 유지
    """
    strategy_tickers = STRATEGY_TICKERS.copy()
    download_tickers = strategy_tickers + (["SPYM"] if "SPYM" not in strategy_tickers else [])

    # 1) 과거 데이터 다운로드
    start_date = "2014-11-01"
    end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

    with st.spinner("데이터를 다운로드하는 중..."):
        # 수정주가(Adj Close) 다운로드 - 배당/분할을 반영한 가격
        downloaded = yf.download(
            download_tickers,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False
        )["Adj Close"]
        if isinstance(downloaded, pd.Series):
            downloaded = downloaded.to_frame()
        downloaded.index = downloaded.index.tz_localize(None)
        pricing_data = downloaded.copy()
        pricing_data.index = pricing_data.index.tz_localize(None)

        # 2) 오늘 장중 가격(fast_info)을 마지막 행으로 추가
        today = pd.Timestamp.now().normalize()
        fast_prices = {}
        for t in download_tickers:
            try:
                ticker = yf.Ticker(t)
                fast_prices[t] = ticker.fast_info["last_price"]
            except Exception:
                if t in pricing_data.columns and len(pricing_data[t].dropna()) > 0:
                    fast_prices[t] = pricing_data[t].dropna().iloc[-1]

        if today not in pricing_data.index and fast_prices:
            pricing_data.loc[today] = pd.Series(fast_prices)
            pricing_data.sort_index(inplace=True)

        # 3) 신호에는 완료된 월의 수정주가만 사용한다.
        # 장중/당월 가격은 매수 수량과 보유자산 평가에만 사용한다.
        signal_cutoff = get_last_completed_month_end(today)
        data = downloaded.loc[downloaded.index <= signal_cutoff, strategy_tickers].copy()
        if data.empty:
            raise ValueError("완료된 월말 가격 데이터가 없습니다.")

        # 4) 모멘텀 점수 계산
        momentum_scores = calculate_momentum_scores(data)

        # 5) TIP 기준 자산 선택 (offense/defense) 및 target_date 결정
        selected_assets, target_date = select_assets(momentum_scores, data)

        # 현재 가격을 당월 가상 월말 가격으로 사용한 다음 리밸런싱 예상 신호.
        # 실제 추천과 백테스트에는 사용하지 않고 화면 비교용으로만 전달한다.
        preview_data = pricing_data[strategy_tickers].copy()
        preview_returns, preview_scores, preview_date, preview_month_end = calculate_preview_momentum(
            preview_data, today
        )
        preview_score_frame = preview_scores.to_frame().T
        preview_score_frame.index = pd.DatetimeIndex([preview_month_end])
        preview_selected_assets, _ = select_assets(
            preview_score_frame, preview_data, preview_month_end
        )

        # 6) 백테스트 실행 (항상 SPY 기준)
        portfolio_value, rebalancing_history, performance_metrics, analysis_data = run_backtest(
            data, momentum_scores, total_balance
        )

        # 7) 최근 12개월 리밸런싱 내역 (표시만 선택 ETF 반영)
        recent_rebalancing = get_recent_rebalancing_history(
            data, momentum_scores, months=12, sp500_rebalance_ticker=sp500_rebalance_ticker
        )

        # 8) 결과 요약 및 테이블 생성
        return display_results(
            momentum_scores,
            data,
            pricing_data,
            selected_assets,
            strategy_tickers,
            total_balance,
            target_date,
            preview_returns,
            preview_scores,
            preview_selected_assets,
            preview_date,
            preview_month_end,
            portfolio_value,
            rebalancing_history,
            performance_metrics,
            recent_rebalancing,
            analysis_data,
            sp500_rebalance_ticker=sp500_rebalance_ticker
        )


def display_results(
    momentum_scores: pd.DataFrame,
    data: pd.DataFrame,
    pricing_data: pd.DataFrame,
    selected_assets: list,
    tickers: list,
    total_balance: float,
    target_date: pd.Timestamp,
    preview_returns: dict,
    preview_scores: pd.Series,
    preview_selected_assets: list,
    preview_date: pd.Timestamp,
    preview_month_end: pd.Timestamp,
    portfolio_value: pd.Series = None,
    rebalancing_history: list = None,
    performance_metrics: dict = None,
    recent_rebalancing: list = None,
    analysis_data: dict = None,
    sp500_rebalance_ticker: str = "SPY"
):
    """결과 표시 및 데이터 반환
    - 신호/랭킹/백테스트는 SPY 기준
    - 실제 매수 추천 수량/가격은 SPY 또는 SPYM 중 선택한 티커 기준
    """
    haa_bal = total_balance * 0.8

    # 선택된 자산 데이터 준비
    selected_data = []
    for asset, score in selected_assets:
        execution_asset = get_rebalance_ticker(asset, sp500_rebalance_ticker)
        price = get_price_with_fallback(pricing_data, data, target_date, asset, sp500_rebalance_ticker)
        shares = haa_bal / len(selected_assets) / price
        purchase_amount = haa_bal / len(selected_assets)
        selected_data.append({
            "자산": execution_asset,
            "신호 기준": asset,
            "모멘텀 점수": f"{score:.3f}",
            "현재 가격": f"${price:.2f}",
            "구매 수량": f"{shares:.2f}",
            "구매 금액": f"${purchase_amount:,.2f}"
        })

    # BRK-B 모멘텀 점수 계산
    brk_price = get_price_with_fallback(pricing_data, data, target_date, "BRK-B", sp500_rebalance_ticker)
    brk_shares = total_balance * 0.2 / brk_price
    brk_purchase_amount = total_balance * 0.2
    brk_momentum = momentum_scores.loc[target_date, "BRK-B"]
    selected_data.append({
        "자산": "BRK-B",
        "신호 기준": "BRK-B",
        "모멘텀 점수": f"{brk_momentum:.3f}",
        "현재 가격": f"${brk_price:.2f}",
        "구매 수량": f"{brk_shares:.2f}",
        "구매 금액": f"${brk_purchase_amount:,.2f}"
    })

    # 예상 신호는 실제 매수 지시가 아닌 다음 월말 리밸런싱 참고용이다.
    preview_selected_data = []
    preview_haa_weight = 0.8 / len(preview_selected_assets) if preview_selected_assets else 0
    for asset, score in preview_selected_assets:
        preview_selected_data.append({
            "예상 자산": format_selected_asset_label(asset, sp500_rebalance_ticker),
            "신호 기준": asset,
            "예상 모멘텀": f"{score:.3f}",
            "예상 비중": f"{preview_haa_weight * 100:.0f}%"
        })
    preview_selected_data.append({
        "예상 자산": "BRK-B",
        "신호 기준": "BRK-B",
        "예상 모멘텀": f"{preview_scores['BRK-B']:.3f}",
        "예상 비중": "20%"
    })

    official_names = [format_selected_asset_label(a, sp500_rebalance_ticker) for a, _ in selected_assets]
    preview_names = [format_selected_asset_label(a, sp500_rebalance_ticker) for a, _ in preview_selected_assets]
    added = [asset for asset in preview_names if asset not in official_names]
    removed = [asset for asset in official_names if asset not in preview_names]
    if not added and not removed:
        preview_change_summary = "현재 확정 구성과 예상 구성이 같습니다."
    else:
        changes = []
        if added:
            changes.append(f"편입 예상: {', '.join(added)}")
        if removed:
            changes.append(f"제외 예상: {', '.join(removed)}")
        preview_change_summary = " / ".join(changes)

    # 반환할 데이터 준비
    result_data = {
        "target_date": target_date,
        "price_date": pricing_data.index[-1],
        "preview_date": preview_date,
        "preview_month_end": preview_month_end,
        "total_balance": total_balance,
        "selected_data": selected_data,
        "preview_selected_data": preview_selected_data,
        "preview_change_summary": preview_change_summary,
        "momentum_scores": momentum_scores,
        "preview_scores": preview_scores,
        "preview_returns": preview_returns,
        "data": data,
        "pricing_data": pricing_data,
        "tickers": tickers,
        "selected_assets": selected_assets,
        "haa_bal": haa_bal,
        "brk_shares": brk_shares,
        "portfolio_value": portfolio_value,
        "rebalancing_history": rebalancing_history,
        "performance_metrics": performance_metrics,
        "recent_rebalancing": recent_rebalancing,
        "analysis_data": analysis_data,
        "sp500_rebalance_ticker": sp500_rebalance_ticker
    }

    # ==== 아래쪽: 전체 자산군 테이블 생성 ====
    st.subheader("📈 전체 자산군 분석")
    st.caption(f"리밸런싱 실행 ETF: {sp500_rebalance_ticker} / 신호·백테스트 기준: SPY")
    monthly_prices = get_month_end_prices(data)
    period_returns = calculate_momentum_returns(data)
    recent = monthly_prices.loc[target_date]
    official_scores = momentum_scores.loc[target_date]
    official_ranks = build_rank_labels(official_scores)
    preview_ranks = build_rank_labels(preview_scores)
    current_prices = pricing_data[tickers].ffill().iloc[-1]

    official_df = pd.DataFrame({
        "확정 순위": official_ranks,
        "월말 기준가": recent,
        "확정 모멘텀": official_scores,
        "1M (%)": period_returns[1].loc[target_date] * 100,
        "3M (%)": period_returns[3].loc[target_date] * 100,
        "6M (%)": period_returns[6].loc[target_date] * 100,
        "12M (%)": period_returns[12].loc[target_date] * 100,
    }).loc[tickers]

    preview_df = pd.DataFrame({
        "예상 순위": preview_ranks,
        "현재가": current_prices,
        "예상 모멘텀": preview_scores,
        "1M 예상 (%)": preview_returns[1] * 100,
        "3M 예상 (%)": preview_returns[3] * 100,
        "6M 예상 (%)": preview_returns[6] * 100,
        "12M 예상 (%)": preview_returns[12] * 100,
    }).loc[tickers]

    df = pd.DataFrame({
        "확정 순위": official_ranks,
        "예상 순위": preview_ranks,
        "확정 모멘텀": official_scores,
        "예상 모멘텀": preview_scores,
        "점수 변화": preview_scores - official_scores,
    }).loc[tickers]
    df["구성 변화"] = np.where(df["확정 순위"] == df["예상 순위"], "-", "변경")

    # ---- 구매 수량 계산 ----
    df["구매 수량"] = ""
    for asset, _ in selected_assets:
        price = get_price_with_fallback(pricing_data, data, target_date, asset, sp500_rebalance_ticker)
        shares = haa_bal / len(selected_assets) / price
        df.loc[asset, "구매 수량"] = f"{shares:.2f}"
    df.loc["BRK-B", "구매 수량"] = f"{brk_shares:.2f}"

    # SPY 선택 시 실행 티커 안내용 컬럼 추가
    df["실행 티커"] = ""
    df.loc["SPY", "실행 티커"] = sp500_rebalance_ticker
    df.loc["BRK-B", "실행 티커"] = "BRK-B"
    df = df[["확정 순위", "예상 순위", "실행 티커", "확정 모멘텀", "예상 모멘텀", "점수 변화", "구성 변화", "구매 수량"]]

    for col in ["확정 모멘텀", "예상 모멘텀", "점수 변화"]:
        df[col] = df[col].apply(lambda x: f"{x:.3f}")

    official_df["월말 기준가"] = official_df["월말 기준가"].apply(lambda x: f"${x:,.2f}")
    official_df["확정 모멘텀"] = official_df["확정 모멘텀"].apply(lambda x: f"{x:.3f}")
    for col in ["1M (%)", "3M (%)", "6M (%)", "12M (%)"]:
        official_df[col] = official_df[col].apply(lambda x: f"{x:.2f}%")

    preview_df["현재가"] = preview_df["현재가"].apply(lambda x: f"${x:,.2f}")
    preview_df["예상 모멘텀"] = preview_df["예상 모멘텀"].apply(lambda x: f"{x:.3f}")
    for col in ["1M 예상 (%)", "3M 예상 (%)", "6M 예상 (%)", "12M 예상 (%)"]:
        preview_df[col] = preview_df[col].apply(lambda x: f"{x:.2f}%")

    result_data["df"] = df
    result_data["official_detail_df"] = official_df
    result_data["preview_detail_df"] = preview_df
    return result_data


def run_backtest(data: pd.DataFrame, momentum_scores: pd.DataFrame, initial_balance: float = 10000.0):
    """HAA 전략 백테스트 실행"""
    try:
        # momentum_scores에 데이터가 있는 날짜만 필터링
        if len(momentum_scores.index) > 0:
            first_valid_date = momentum_scores.index[0]
            data_filtered = data[data.index >= first_valid_date].copy()
        else:
            data_filtered = data.copy()

        # 월말 날짜 추출 (문자열 별칭 대신 offset 객체를 사용해 pandas 버전 차이 회피)
        monthly_dates = data_filtered.resample(MonthEnd()).last().index

        # 최소 2개월 데이터 필요
        if len(monthly_dates) < 2:
            return None, None, None, None

        # 포트폴리오 가치 추적
        portfolio_value = pd.Series(index=monthly_dates, dtype=float)
        portfolio_value.iloc[0] = initial_balance

        # 리밸런싱 내역 저장
        rebalancing_history = []

        # 각 월별로 리밸런싱 및 수익률 계산
        for i in range(1, len(monthly_dates)):
            current_date = monthly_dates[i]
            prev_date = monthly_dates[i - 1]

            # 실제 거래일 찾기 (월말 날짜가 주말이면 가장 가까운 이전 거래일 사용)
            if prev_date not in data_filtered.index:
                available_dates = data_filtered.index[data_filtered.index <= prev_date]
                if len(available_dates) > 0:
                    prev_date_actual = available_dates[-1]
                else:
                    continue
            else:
                prev_date_actual = prev_date

            if current_date not in data_filtered.index:
                available_dates = data_filtered.index[data_filtered.index <= current_date]
                if len(available_dates) > 0:
                    current_date_actual = available_dates[-1]
                else:
                    continue
            else:
                current_date_actual = current_date

            # 이전 달 말에 선택된 자산
            try:
                # 모멘텀 인덱스는 달력상 월말 라벨이므로 실제 마지막 거래일이
                # 그보다 이르더라도 해당 월의 신호(prev_date)를 사용한다.
                selected_assets, _ = select_assets(momentum_scores, data_filtered, prev_date)
            except Exception as e:
                st.warning(f"날짜 {prev_date_actual}에서 자산 선택 실패: {e}")
                continue

            # HAA 80% + BRK-B 20% 구성
            haa_weight = 0.8
            brk_weight = 0.2

            # 각 자산의 월간 수익률 계산
            haa_return = 0.0
            haa_count = 0

            # HAA 자산들의 수익률 (균등 비중)
            if selected_assets:
                for asset, _ in selected_assets:
                    if asset in data_filtered.columns:
                        if prev_date_actual in data_filtered.index and current_date_actual in data_filtered.index:
                            prev_price = data_filtered.loc[prev_date_actual, asset]
                            curr_price = data_filtered.loc[current_date_actual, asset]
                            if not pd.isna(prev_price) and not pd.isna(curr_price) and prev_price > 0:
                                asset_return = (curr_price / prev_price) - 1
                                haa_return += asset_return
                                haa_count += 1

            # HAA 평균 수익률 계산
            if haa_count > 0:
                haa_return = haa_return / haa_count
            else:
                haa_return = 0.0

            # BRK-B 수익률
            brk_return = 0.0
            if "BRK-B" in data_filtered.columns:
                if prev_date_actual in data_filtered.index and current_date_actual in data_filtered.index:
                    prev_price = data_filtered.loc[prev_date_actual, "BRK-B"]
                    curr_price = data_filtered.loc[current_date_actual, "BRK-B"]
                    if not pd.isna(prev_price) and not pd.isna(curr_price) and prev_price > 0:
                        brk_return = (curr_price / prev_price) - 1

            # 포트폴리오 수익률 = HAA 80% + BRK-B 20%
            portfolio_return = (haa_weight * haa_return) + (brk_weight * brk_return)

            # 포트폴리오 가치 업데이트 (복리 계산)
            if portfolio_value.iloc[i - 1] > 0:
                portfolio_value.iloc[i] = portfolio_value.iloc[i - 1] * (1 + portfolio_return)
            else:
                portfolio_value.iloc[i] = portfolio_value.iloc[i - 1]

            # 리밸런싱 내역 저장
            haa_assets = len(selected_assets)
            if haa_assets > 0:
                haa_weight_per_asset = 0.8 / haa_assets
                asset_weights = []
                for rank, (asset, score) in enumerate(selected_assets, 1):
                    asset_weights.append(f"{asset}({haa_weight_per_asset*100:.0f}% {rank}위)")
                asset_weights.append("BRK-B(20% 보유)")
                asset_str = ", ".join(asset_weights)
            else:
                asset_str = "BRK-B(20% 보유)"

            rebalancing_history.append({
                "적용 시점": current_date.strftime("%Y-%m-%d"),
                "목표 자산 비중": asset_str
            })

        # 성과 지표 계산
        initial_value = portfolio_value.iloc[0]
        final_value = portfolio_value.iloc[-1]
        total_return = (final_value / initial_value) - 1

        # 기간 계산
        start_date = monthly_dates[0]
        end_date = monthly_dates[-1]
        days_diff = (end_date - start_date).days
        years = days_diff / 365.25

        # CAGR 계산
        if years > 0 and initial_value > 0:
            cagr = ((final_value / initial_value) ** (1 / years)) - 1
        else:
            cagr = 0

        # 검증: 월별 수익률로부터 CAGR 재계산
        monthly_returns_series = portfolio_value.pct_change().dropna()
        if len(monthly_returns_series) > 0:
            monthly_geometric_mean = (1 + monthly_returns_series).prod() ** (1 / len(monthly_returns_series)) - 1
            cagr_from_monthly = ((1 + monthly_geometric_mean) ** 12) - 1

            if abs(cagr - cagr_from_monthly) > 0.01:
                st.warning(f"⚠️ CAGR 계산 검증: 직접 계산={cagr*100:.2f}%, 월별 기하평균={cagr_from_monthly*100:.2f}%")

        # 월별 수익률 계산
        monthly_returns_series = portfolio_value.pct_change().dropna()

        # 연환산 변동성
        if len(monthly_returns_series) > 1:
            volatility = monthly_returns_series.std() * np.sqrt(12)
        else:
            volatility = 0

        # 최대 낙폭 (MDD)
        cumulative = (1 + monthly_returns_series).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        mdd = drawdown.min()

        # 무위험 수익률 조회
        start_date_str = monthly_dates[0].strftime("%Y-%m-%d")
        end_date_str = monthly_dates[-1].strftime("%Y-%m-%d")
        risk_free_rate = get_risk_free_rate(start_date=start_date_str, end_date=end_date_str)

        # 샤프 비율
        sharpe = ((cagr - risk_free_rate) / volatility) if volatility > 0 else 0

        performance_metrics = {
            "총 수익률": f"{total_return*100:.2f}%",
            "CAGR": f"{cagr*100:.2f}%",
            "연환산 변동성": f"{volatility*100:.2f}%",
            "샤프 비율": f"{sharpe:.2f}",
            "무위험 수익률": f"{risk_free_rate*100:.2f}%",
            "무위험 수익률 기준": "미국 13주 단기국채 수익률 (^IRX) 평균",
            "최대 낙폭 (MDD)": f"{mdd*100:.2f}%",
            "시작일": monthly_dates[0].strftime("%Y-%m-%d"),
            "종료일": monthly_dates[-1].strftime("%Y-%m-%d"),
            "기간 (년)": f"{years:.2f}"
        }

        # 추가 분석 데이터 계산
        yearly_returns = calculate_yearly_returns(portfolio_value)
        monthly_returns = calculate_monthly_returns(portfolio_value)
        monthly_heatmap = create_monthly_heatmap_data(monthly_returns)
        drawdown_series, drawdown_events = calculate_drawdown_events(portfolio_value)
        monthly_distribution = create_monthly_distribution(monthly_returns)

        return portfolio_value, rebalancing_history, performance_metrics, {
            "yearly_returns": yearly_returns,
            "monthly_returns": monthly_returns,
            "monthly_heatmap": monthly_heatmap,
            "drawdown_series": drawdown_series,
            "drawdown_events": drawdown_events,
            "monthly_distribution": monthly_distribution
        }

    except Exception as e:
        st.error(f"백테스트 실행 중 오류 발생: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None, None


def get_risk_free_rate(start_date: str = None, end_date: str = None):
    """
    무위험 수익률 조회 (미국 13주 단기국채 수익률, ^IRX)
    start_date와 end_date가 제공되면 해당 기간의 평균을 사용,
    없으면 최근 1개월 값을 사용합니다.

    참고:
    - ^IRX 는 Yahoo Finance에서 13주 T-Bill 수익률 지표로 많이 사용됩니다.
    - 값이 5.25 형태로 내려오면 이는 5.25%를 뜻하므로 100으로 나누어 연율 소수로 변환합니다.
    """
    try:
        # 미국 13주 단기국채 수익률 조회 (^IRX)
        ticker = yf.Ticker("^IRX")

        if start_date and end_date:
            # 백테스트 기간 전체의 평균 단기 무위험 수익률 사용
            hist = ticker.history(start=start_date, end=end_date)
            if not hist.empty:
                avg_rate = hist["Close"].dropna().mean() / 100.0
                if pd.notna(avg_rate):
                    return float(avg_rate)
        else:
            # 최근 1개월 마지막 값 사용
            hist = ticker.history(period="1mo")
            if not hist.empty:
                current_rate = hist["Close"].dropna().iloc[-1] / 100.0
                if pd.notna(current_rate):
                    return float(current_rate)
    except Exception:
        pass

    # 조회 실패 시 기본값
    return 0.025


def calculate_yearly_returns(portfolio_value):
    """연도별 수익률 계산"""
    if portfolio_value is None or len(portfolio_value) < 2:
        return None

    yearly = portfolio_value.resample(YearEnd()).last()
    yearly_returns = yearly.pct_change().dropna() * 100
    return yearly_returns


def calculate_monthly_returns(portfolio_value):
    """월별 수익률 계산"""
    if portfolio_value is None or len(portfolio_value) < 2:
        return None

    monthly_returns = portfolio_value.pct_change().dropna() * 100
    return monthly_returns


def create_monthly_heatmap_data(monthly_returns):
    """월별 수익률 히트맵 데이터 생성 (연도 x 월)"""
    if monthly_returns is None or len(monthly_returns) == 0:
        return None

    # 연도와 월로 분리
    monthly_returns.index = pd.to_datetime(monthly_returns.index)
    monthly_returns_df = monthly_returns.to_frame("return")
    monthly_returns_df["year"] = monthly_returns_df.index.year
    monthly_returns_df["month"] = monthly_returns_df.index.month

    # 피벗 테이블 생성
    heatmap_data = monthly_returns_df.pivot_table(
        values="return",
        index="year",
        columns="month",
        aggfunc="first"
    )

    # 컬럼 이름을 월 이름으로 변경
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    month_dict = {i: month_names[i - 1] for i in range(1, 13)}

    heatmap_data.columns = [month_dict[i] if i in month_dict else f"Month_{i}"
                            for i in heatmap_data.columns]

    # 있는 월만 유지
    heatmap_data = heatmap_data[[col for col in heatmap_data.columns if col in month_names]]

    # 연도 순서 역순
    heatmap_data = heatmap_data.sort_index(ascending=False)

    # 평균 행 계산
    monthly_avg = heatmap_data.mean(axis=0, skipna=True)
    avg_row = pd.DataFrame([monthly_avg.values], index=["평균"], columns=heatmap_data.columns)

    # 평균 행을 맨 앞에 추가
    heatmap_data = pd.concat([avg_row, heatmap_data])

    return heatmap_data


def calculate_drawdown_events(portfolio_value):
    """드로우다운 이벤트 계산"""
    if portfolio_value is None or len(portfolio_value) < 2:
        return None, None

    # 드로우다운 계산
    cumulative = portfolio_value
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max * 100

    # 드로우다운 이벤트 찾기
    drawdown_events = []
    in_drawdown = False
    drawdown_start = None
    max_drawdown = 0

    for date, dd_value in drawdown.items():
        if dd_value < 0 and not in_drawdown:
            in_drawdown = True
            drawdown_start = date
            max_drawdown = dd_value
        elif dd_value < max_drawdown and in_drawdown:
            max_drawdown = dd_value
        elif dd_value >= 0 and in_drawdown:
            drawdown_period = drawdown.loc[drawdown_start:date]
            trough_date = drawdown_period.idxmin()
            trough_value = drawdown_period.min()

            drawdown_events.append({
                "start": drawdown_start,
                "trough": trough_date,
                "end": date,
                "drawdown": trough_value
            })
            in_drawdown = False
            max_drawdown = 0

    # 진행 중인 드로우다운 처리
    if in_drawdown:
        drawdown_period = drawdown.loc[drawdown_start:]
        trough_date = drawdown_period.idxmin()
        trough_value = drawdown_period.min()
        drawdown_events.append({
            "start": drawdown_start,
            "trough": trough_date,
            "end": portfolio_value.index[-1],
            "drawdown": trough_value
        })

    # 드로우다운 크기순 정렬
    drawdown_events.sort(key=lambda x: x["drawdown"])

    return drawdown, drawdown_events


def create_monthly_distribution(monthly_returns):
    """월별 수익률 분포 히스토그램 데이터 생성"""
    if monthly_returns is None or len(monthly_returns) == 0:
        return None

    # 히스토그램 구간 설정 (-10% ~ 10%, 1% 간격)
    bins = np.arange(-10, 11, 1)
    hist, bin_edges = np.histogram(monthly_returns.values, bins=bins)

    # 중간값 계산
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 구간 레이블 생성
    bin_labels = []
    for i in range(len(bin_edges) - 1):
        left = int(bin_edges[i])
        right = int(bin_edges[i + 1])
        if right == 10:
            bin_labels.append(f"[{left}%, {right}%]")
        else:
            bin_labels.append(f"[{left}%, {right}%)")

    return pd.DataFrame({
        "bin_center": bin_centers,
        "bin_label": bin_labels,
        "count": hist
    })


def get_asset_full_name(ticker: str) -> str:
    """티커의 전체 이름 반환"""
    asset_names = {
        "SPY": "SPDR S&P 500 ETF Trust",
        "SPYM": "SPDR Portfolio S&P 500 High Dividend ETF",
        "VEA": "Vanguard FTSE Developed Markets ETF",
        "VWO": "Vanguard FTSE Emerging Markets ETF",
        "IWM": "iShares Russell 2000 ETF",
        "TLT": "iShares 20+ Year Treasury Bond ETF",
        "PDBC": "Invesco Optimum Yield Diversified Commodity Strategy No K-1 ETF",
        "VNQ": "Vanguard Real Estate ETF",
        "IEF": "iShares 7-10 Year Treasury Bond ETF",
        "BIL": "SPDR Bloomberg 1-3 Month T-Bill ETF",
        "TIP": "iShares TIPS Bond ETF",
        "BRK-B": "Berkshire Hathaway Inc. Class B"
    }
    return asset_names.get(ticker, ticker)


def _encode_holdings_payload(holdings: dict) -> str:
    payload = json.dumps(holdings, ensure_ascii=False, separators=(",", ":"))
    return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("utf-8")


def _decode_holdings_payload(payload: str):
    decoded = base64.urlsafe_b64decode(payload.encode("utf-8")).decode("utf-8")
    data = json.loads(decoded)
    if not isinstance(data, dict):
        raise ValueError("Invalid holdings payload")
    return data


def _inject_localstorage_restore_script():
    """브라우저 localStorage → URL 파라미터로 복원 트리거"""
    components.html(
        """
        <script>
        const KEY = "haa_holdings_v1";
        const url = new URL(window.parent.location.href);
        if (!url.searchParams.get("holdings")) {
          const saved = window.localStorage.getItem(KEY);
          if (saved) {
            url.searchParams.set("holdings", saved);
            window.parent.location.replace(url.toString());
          }
        }
        </script>
        """,
        height=0,
    )


def _save_holdings_to_localstorage(holdings: dict):
    payload = _encode_holdings_payload(holdings)
    components.html(
        f"""
        <script>
        const KEY = "haa_holdings_v1";
        window.localStorage.setItem(KEY, "{payload}");
        </script>
        """,
        height=0,
    )


def _clear_holdings_from_localstorage():
    components.html(
        """
        <script>
        const KEY = "haa_holdings_v1";
        window.localStorage.removeItem(KEY);
        </script>
        """,
        height=0,
    )


def restore_holdings_from_query_params(tickers: list):
    """URL의 holdings 파라미터를 읽어 세션 상태에 반영"""
    holdings_payload = st.query_params.get("holdings", "")
    already_restored = st.session_state.get("holdings_restored", False)
    if holdings_payload and not already_restored:
        try:
            restored = _decode_holdings_payload(holdings_payload)
            st.session_state["holdings"] = {t: float(restored.get(t, 0.0)) for t in tickers}
            st.session_state["holdings_restored"] = True
            st.sidebar.success("localStorage에서 보유수량을 자동 복원했습니다.")
        except Exception:
            st.sidebar.warning("localStorage 자동 복원에 실패했습니다. JSON 백업을 사용해주세요.")


def initialize_holdings_state(tickers: list):
    """세션 상태에 보유 수량 딕셔너리 초기화/동기화"""
    if "holdings" not in st.session_state:
        st.session_state["holdings"] = {t: 0.0 for t in tickers}
    else:
        for t in tickers:
            st.session_state["holdings"].setdefault(t, 0.0)


def render_holdings_manager(tickers: list):
    """사이드바에서 현재 보유 수량 입력/저장/불러오기 UI"""
    initialize_holdings_state(tickers)

    with st.sidebar.expander("📦 현재 보유수량 저장", expanded=False):
        uploaded_file = st.file_uploader(
            "보유수량 JSON 불러오기",
            type=["json"],
            key="holdings_json_upload",
            help="티커:수량 형태의 JSON 파일을 업로드하세요."
        )

        if uploaded_file is not None:
            try:
                payload = json.load(uploaded_file)
                if isinstance(payload, dict):
                    for t in tickers:
                        st.session_state["holdings"][t] = float(payload.get(t, 0.0))
                    _save_holdings_to_localstorage(st.session_state["holdings"])
                    st.success("보유수량을 불러왔습니다.")
                else:
                    st.error("JSON 형식이 올바르지 않습니다. (객체 형태 필요)")
            except Exception as e:
                st.error(f"JSON 불러오기 실패: {e}")

        edited_holdings = {}
        for t in tickers:
            edited_holdings[t] = st.number_input(
                f"{t} 수량",
                min_value=0.0,
                value=float(st.session_state["holdings"].get(t, 0.0)),
                step=1.0,
                key=f"holding_input_{t}"
            )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 보유수량 저장", use_container_width=True):
                st.session_state["holdings"] = edited_holdings
                _save_holdings_to_localstorage(st.session_state["holdings"])
                st.success("현재 보유수량이 저장되었습니다.")
        with col2:
            if st.button("↩️ 수량 초기화", use_container_width=True):
                st.session_state["holdings"] = {t: 0.0 for t in tickers}
                _clear_holdings_from_localstorage()
                st.success("보유수량을 0으로 초기화했습니다.")
                st.rerun()

        holdings_json = json.dumps(st.session_state["holdings"], ensure_ascii=False, indent=2)
        st.download_button(
            label="📥 보유수량 JSON 다운로드",
            data=holdings_json,
            file_name="haa_holdings.json",
            mime="application/json",
            use_container_width=True
        )


def get_recent_rebalancing_history(
    data: pd.DataFrame,
    momentum_scores: pd.DataFrame,
    months: int = 12,
    sp500_rebalance_ticker: str = "SPY"
):
    """최근 N개월 리밸런싱 내역 추출
    - 신호 계산은 SPY 기준
    - 표시용 티커만 SPY/SPYM 선택을 반영
    """
    end_date = get_last_completed_month_end()
    start_date = end_date - pd.DateOffset(months=months)

    monthly_dates = momentum_scores.index
    monthly_dates = monthly_dates[(monthly_dates >= start_date) & (monthly_dates <= end_date)]

    if len(monthly_dates) == 0:
        return []

    rebalancing_history = []

    for date in monthly_dates:
        target_date = date
        if target_date not in momentum_scores.index:
            available_dates = momentum_scores.index[momentum_scores.index <= target_date]
            if len(available_dates) > 0:
                target_date = available_dates[-1]
            else:
                continue

        selected_assets, _ = select_assets(momentum_scores, data, target_date)

        haa_assets = len(selected_assets)
        if haa_assets > 0:
            haa_weight_per_asset = 0.8 / haa_assets
            asset_weights = []
            for rank, (asset, score) in enumerate(selected_assets, 1):
                display_asset = format_selected_asset_label(asset, sp500_rebalance_ticker)
                asset_weights.append(f"{display_asset}({haa_weight_per_asset*100:.0f}% {rank}위)")
            asset_weights.append("BRK-B(20% 보유)")
            asset_str = ", ".join(asset_weights)
        else:
            asset_str = "BRK-B(20% 보유)"

        rebalancing_history.append({
            "적용 시점": date.strftime("%Y-%m-%d"),
            "목표 자산 비중": asset_str
        })

    rebalancing_history.reverse()
    return rebalancing_history



# ==== Streamlit 앱 메인 ====
st.set_page_config(
    page_title="HAA 전략 스크리너",
    page_icon="📊",
    layout="wide"
)

st.title("📊 HAA 전략 스크리너")
st.markdown("---")

# 브라우저 localStorage에 보관된 보유수량 자동 복원
_inject_localstorage_restore_script()
restore_holdings_from_query_params(APP_TICKERS)

# 사이드바에 입력 필드
with st.sidebar:
    st.header("⚙️ 설정")
    render_holdings_manager(APP_TICKERS)
    st.markdown("---")
    sp500_rebalance_ticker = st.selectbox(
        "S&P500 리밸런싱 ETF 선택",
        SP500_REBALANCE_OPTIONS,
        index=0,
        help="실제 매수 추천 수량은 SPY 또는 SPYM 중 선택한 티커 기준으로 계산합니다. 백테스트와 모멘텀 신호는 항상 SPY 기준입니다."
    )
    balance_text = st.text_input(
        "보유 금액 입력",
        value="10000",
        help="입력하지 않으면 기본값 10000이 사용됩니다."
    )

    if st.button("🚀 실행", type="primary", use_container_width=True):
        try:
            total_balance = float(balance_text.strip().replace(",", "")) if balance_text else 10000.0
            if total_balance <= 0:
                st.error("보유 금액은 0보다 커야 합니다.")
            else:
                result_data = run_screener(total_balance, sp500_rebalance_ticker=sp500_rebalance_ticker)
                st.session_state["result_data"] = result_data
                st.session_state["balance"] = total_balance
                st.session_state["sp500_rebalance_ticker"] = sp500_rebalance_ticker
        except ValueError:
            st.error("올바른 숫자를 입력해주세요.")
        except Exception as e:
            st.error(f"오류 발생: {e}")

    if st.button("🔄 초기화", use_container_width=True):
        if "result_data" in st.session_state:
            del st.session_state["result_data"]
        if "balance" in st.session_state:
            del st.session_state["balance"]
        st.rerun()

    # ==== 사이드바에 설정 정보 표시 ====
    if "result_data" in st.session_state:
        st.markdown("---")
        st.subheader("📊 설정 정보")
        result_data = st.session_state["result_data"]
        st.metric("확정 신호 월말", result_data["target_date"].strftime("%Y-%m-%d"))
        st.metric("예상 신호 현재가", result_data["preview_date"].strftime("%Y-%m-%d"))
        st.metric("보유 금액", f"${result_data['total_balance']:,.2f}")
        st.caption(f"리밸런싱 ETF: {result_data.get('sp500_rebalance_ticker', 'SPY')} / 백테스트 기준: SPY")
        st.caption("실제 추천은 완료된 월말 신호, 예상 신호는 현재가를 당월 가상 월말 가격으로 사용합니다.")

        st.markdown("---")
        st.subheader("✅ 선택된 자산")
        selected_df = pd.DataFrame(result_data["selected_data"])
        st.dataframe(selected_df, use_container_width=True, hide_index=True)

        st.subheader("🔭 다음 월말 예상")
        st.caption(result_data["preview_change_summary"])
        st.dataframe(
            pd.DataFrame(result_data["preview_selected_data"]),
            use_container_width=True,
            hide_index=True
        )

# 메인 영역에 결과 표시
if "result_data" in st.session_state:
    result_data = st.session_state["result_data"]

    # ==== 본문에 기준 날짜와 투자 금액 표시 ====
    st.subheader("📊 설정 정보")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("확정 신호 월말", result_data["target_date"].strftime("%Y-%m-%d"))
    with col2:
        st.metric("예상 신호 현재가", result_data["preview_date"].strftime("%Y-%m-%d"))
    with col3:
        st.metric("보유 금액", f"${result_data['total_balance']:,.2f}")
    st.caption(f"리밸런싱 ETF: {result_data.get('sp500_rebalance_ticker', 'SPY')} / 백테스트·신호 기준: SPY")
    st.caption(
        f"확정 신호: {result_data['target_date']:%Y-%m-%d} 완료 월말 / "
        f"예상 신호: {result_data['preview_date']:%Y-%m-%d} 현재가를 "
        f"{result_data['preview_month_end']:%Y-%m-%d} 가상 월말 가격으로 사용"
    )
    st.info("예상 신호는 참고용이며 월말 종가가 확정될 때까지 변경될 수 있습니다.")

    st.markdown("---")

    # ==== 확정 자산과 다음 월말 예상 자산 병기 ====
    official_col, preview_col = st.columns(2)
    with official_col:
        st.subheader("✅ 현재 확정 자산")
        st.dataframe(
            pd.DataFrame(result_data["selected_data"]),
            use_container_width=True,
            hide_index=True
        )
    with preview_col:
        st.subheader("🔭 다음 월말 예상 자산")
        st.caption(result_data["preview_change_summary"])
        st.dataframe(
            pd.DataFrame(result_data["preview_selected_data"]),
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")

    # ==== 현재 보유 포지션 평가 ====
    st.subheader("📦 현재 보유 포지션")
    holdings = st.session_state.get("holdings", {})
    price_snapshot = result_data["pricing_data"].ffill().iloc[-1]
    holding_rows = []
    for t, qty in holdings.items():
        if qty > 0 and t in price_snapshot.index:
            current_price = float(price_snapshot[t])
            holding_rows.append({
                "자산": t,
                "자산명": get_asset_full_name(t),
                "보유수량": float(qty),
                "현재가격": current_price,
                "평가금액": current_price * float(qty)
            })

    if holding_rows:
        holdings_df = pd.DataFrame(holding_rows)
        total_holding_value = holdings_df["평가금액"].sum()
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            st.metric("총 평가금액", f"${total_holding_value:,.2f}")
        with col_h2:
            diff = total_holding_value - result_data["total_balance"]
            st.metric("입력 보유금액 대비", f"${diff:,.2f}")

        holdings_view = holdings_df.copy()
        holdings_view["현재가격"] = holdings_view["현재가격"].map(lambda x: f"${x:,.2f}")
        holdings_view["평가금액"] = holdings_view["평가금액"].map(lambda x: f"${x:,.2f}")
        st.dataframe(holdings_view, use_container_width=True, hide_index=True)
    else:
        st.info("사이드바의 '현재 보유수량 저장'에서 수량을 입력하면 평가금액이 표시됩니다.")

    st.markdown("---")

    # ==== 전체 자산군 분석 테이블 ====
    st.subheader("📈 전체 자산군 분석")
    st.caption("구매 수량은 확정 신호에만 적용됩니다. 예상 순위와 예상 모멘텀은 참고용입니다.")
    st.dataframe(
        result_data["df"],
        use_container_width=True,
        height=400
    )

    official_tab, preview_tab = st.tabs(["확정 월말 1·3·6·12개월", "현재가 기준 월말 예상"])
    with official_tab:
        st.caption(f"완료된 월말 {result_data['target_date']:%Y-%m-%d} 기준")
        st.dataframe(result_data["official_detail_df"], use_container_width=True, height=400)
    with preview_tab:
        st.caption(
            f"{result_data['preview_date']:%Y-%m-%d} 현재가를 "
            f"{result_data['preview_month_end']:%Y-%m-%d} 월말 가격으로 가정"
        )
        st.dataframe(result_data["preview_detail_df"], use_container_width=True, height=400)

    # CSV 다운로드 버튼
    csv = result_data["df"].to_csv(index=True)
    st.download_button(
        label="📥 CSV로 다운로드",
        data=csv,
        file_name=f"haa_screener_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

    st.markdown("---")

    # ==== 백테스트 성과 지표 ====
    if result_data.get("performance_metrics"):
        st.subheader("📊 백테스트 성과 지표")
        metrics = result_data["performance_metrics"]
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("CAGR", metrics.get("CAGR", "N/A"))
            st.metric("총 수익률", metrics.get("총 수익률", "N/A"))
        with col2:
            st.metric("연환산 변동성", metrics.get("연환산 변동성", "N/A"))
            st.metric("샤프 비율", metrics.get("샤프 비율", "N/A"))
            st.metric(
                "무위험 수익률",
                metrics.get("무위험 수익률", "N/A"),
                help="백테스트 기간 전체의 미국 13주 단기국채 수익률(^IRX) 평균"
            )
            st.caption(f"기준: {metrics.get('무위험 수익률 기준', '미국 13주 단기국채 수익률 (^IRX) 평균')}")
        with col3:
            st.metric("최대 낙폭 (MDD)", metrics.get("최대 낙폭 (MDD)", "N/A"))
        with col4:
            st.metric("기간", metrics.get("기간 (년)", "N/A") + "년")
            st.metric("시작일", metrics.get("시작일", "N/A"))
        with col5:
            st.metric("종료일", metrics.get("종료일", "N/A"))

    st.markdown("---")

    # ==== 최근 리밸런싱 및 리밸런싱 추이 ====
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📊 최근 리밸런싱")
        current_selected_assets = result_data.get("selected_assets", [])
        if current_selected_assets:
            haa_assets = len(current_selected_assets)
            if haa_assets > 0:
                haa_weight_per_asset = 0.8 / haa_assets
                values = [haa_weight_per_asset * 100] * haa_assets
                labels = [format_selected_asset_label(asset, result_data.get("sp500_rebalance_ticker", "SPY")) for asset, _ in current_selected_assets]
            else:
                values = []
                labels = []

            values.append(20.0)
            labels.append("BRK-B")

            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.5,
                textinfo="label+percent",
                textposition="outside"
            )])
            fig.update_layout(
                height=400,
                showlegend=True,
                margin=dict(t=0, b=0, l=0, r=0)
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📈 리밸런싱 추이")
        if result_data.get("recent_rebalancing"):
            rebal_df = pd.DataFrame(result_data["recent_rebalancing"])
            st.dataframe(
                rebal_df,
                use_container_width=True,
                hide_index=True,
                height=400
            )
        else:
            st.info("리밸런싱 내역이 없습니다.")

    # ==== 백테스트 포트폴리오 가치 차트 ====
    if result_data.get("portfolio_value") is not None:
        st.markdown("---")
        st.subheader("📈 백테스트 포트폴리오 가치 추이")
        portfolio_df = result_data["portfolio_value"].to_frame("포트폴리오 가치")
        fig = px.line(
            portfolio_df,
            x=portfolio_df.index,
            y="포트폴리오 가치",
            title="HAA 전략 백테스트 포트폴리오 가치",
            labels={"value": "포트폴리오 가치 ($)", "index": "날짜"}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # ==== 추가 분석 데이터 표시 ====
        analysis = result_data.get("analysis_data")
        if analysis:
            # 연도별 수익률
            if analysis.get("yearly_returns") is not None and len(analysis["yearly_returns"]) > 0:
                st.markdown("---")
                st.subheader("📊 연도별 수익률 (%)")
                yearly_df = analysis["yearly_returns"].to_frame("수익률")
                yearly_df.index = yearly_df.index.year

                colors = ["#d32f2f" if x < 0 else "#2e7d32" for x in yearly_df["수익률"]]
                year_labels = [f"{int(year)}년" for year in yearly_df.index]

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=year_labels,
                    y=yearly_df["수익률"],
                    marker_color=colors,
                    text=[f"{x:.1f}%" for x in yearly_df["수익률"]],
                    textposition="outside",
                    textfont={"size": 13, "color": "black", "family": "Arial Black"},
                    name="연도별 수익률"
                ))
                fig.update_layout(
                    xaxis_title=dict(text="연도", font=dict(size=14, color="black")),
                    yaxis_title=dict(text="수익률 (%)", font=dict(size=14, color="black")),
                    height=450,
                    showlegend=False,
                    hovermode="x unified",
                    xaxis=dict(
                        tickfont={"size": 12, "color": "black"},
                        title_font={"size": 14, "color": "black"}
                    ),
                    yaxis=dict(
                        tickfont={"size": 12, "color": "black"},
                        title_font={"size": 14, "color": "black"}
                    ),
                    margin=dict(b=60, t=80, l=60, r=40)
                )
                st.plotly_chart(fig, use_container_width=True)

            # 월별 수익률 히트맵
            if analysis.get("monthly_heatmap") is not None and not analysis["monthly_heatmap"].empty:
                st.markdown("---")
                st.subheader("📅 월별 수익률 (%)")

                heatmap_data = analysis["monthly_heatmap"]

                y_labels = []
                y_positions = []
                for pos, idx in enumerate(heatmap_data.index):
                    if idx == "평균":
                        y_labels.append("평균")
                    else:
                        try:
                            year_int = int(float(idx))
                            y_labels.append(str(year_int))
                        except Exception:
                            y_labels.append(str(idx))
                    y_positions.append(pos)

                z_min = np.nanmin(heatmap_data.values)
                z_max = np.nanmax(heatmap_data.values)

                z_normalized = heatmap_data.values.copy()
                abs_min = abs(z_min) if z_min < 0 else 0
                abs_max = abs(z_max) if z_max > 0 else 0

                for i in range(len(z_normalized)):
                    for j in range(len(z_normalized[i])):
                        val = z_normalized[i, j]
                        if not pd.isna(val):
                            if val < 0:
                                if abs_min > 0:
                                    z_normalized[i, j] = 0.5 * (1 - abs(val) / abs_min)
                                else:
                                    z_normalized[i, j] = 0.5
                            elif val > 0:
                                if abs_max > 0:
                                    z_normalized[i, j] = 0.5 + 0.5 * (val / abs_max)
                                else:
                                    z_normalized[i, j] = 0.5
                            else:
                                z_normalized[i, j] = 0.5

                fig = go.Figure(data=go.Heatmap(
                    z=z_normalized,
                    x=heatmap_data.columns,
                    y=y_positions,
                    colorscale=[
                        [0.0, "#b71c1c"],
                        [0.2, "#d32f2f"],
                        [0.4, "#ffcdd2"],
                        [0.5, "#ffffff"],
                        [0.6, "#c8e6c9"],
                        [0.8, "#2e7d32"],
                        [1.0, "#1b5e20"]
                    ],
                    text=[[f"<b>{val:.1f}%</b>" if not pd.isna(val) else "" for val in row]
                          for row in heatmap_data.values],
                    texttemplate="%{text}",
                    textfont={"size": 16, "color": "black", "family": "Arial Black"},
                    colorbar=dict(
                        title=dict(text="수익률 (%)", font=dict(size=13, color="black")),
                        tickmode="array",
                        tickvals=[0.0, 0.25, 0.5, 0.75, 1.0],
                        ticktext=[
                            f"{z_min:.1f}%" if z_min < 0 else "0%",
                            f"{z_min/2:.1f}%" if z_min < 0 else "0%",
                            "0%",
                            f"{z_max/2:.1f}%" if z_max > 0 else "0%",
                            f"{z_max:.1f}%" if z_max > 0 else "0%"
                        ],
                        tickfont={"size": 11, "color": "black"}
                    ),
                    ygap=2,
                    xgap=2
                ))
                fig.update_layout(
                    height=450 + len(heatmap_data) * 30,
                    xaxis_title=dict(text="월", font=dict(size=14, color="black")),
                    yaxis_title=dict(text="연도", font=dict(size=14, color="black")),
                    xaxis=dict(
                        tickfont={"size": 12, "color": "black"},
                        title_font={"size": 14, "color": "black"}
                    ),
                    yaxis=dict(
                        autorange="reversed",
                        tickmode="array",
                        tickvals=y_positions,
                        ticktext=y_labels,
                        tickfont={"size": 12, "color": "black"},
                        title_font={"size": 14, "color": "black"},
                        dtick=None
                    ),
                    margin=dict(b=60, t=60, l=80, r=80)
                )
                st.plotly_chart(fig, use_container_width=True)

            # 월별 수익률 분포
            if analysis.get("monthly_distribution") is not None:
                st.markdown("---")
                st.subheader("📊 월별 수익률 분포")

                dist_data = analysis["monthly_distribution"].copy()
                total_count = dist_data["count"].sum()
                dist_data["percentage"] = (dist_data["count"] / total_count * 100) if total_count > 0 else 0

                colors = ["#d32f2f" if x < 0 else "#2e7d32" for x in dist_data["bin_center"]]
                x_labels = dist_data["bin_label"].tolist()

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=x_labels,
                    y=dist_data["count"],
                    marker_color=colors,
                    name="빈도",
                    text=[f"{pct:.1f}%" if count > 0 else "" for count, pct in zip(dist_data["count"], dist_data["percentage"])],
                    textposition="outside",
                    textfont={"size": 13, "color": "black", "family": "Arial Black"},
                    hovertemplate="구간: %{x}<br>빈도: %{y}회<br>비율: %{customdata:.1f}%<extra></extra>",
                    customdata=dist_data["percentage"]
                ))
                fig.update_layout(
                    xaxis_title=dict(text="수익률 구간 (%)", font=dict(size=14, color="black")),
                    yaxis_title=dict(text="빈도 (회)", font=dict(size=14, color="black")),
                    height=550,
                    showlegend=False,
                    hovermode="x unified",
                    xaxis=dict(
                        tickmode="array",
                        tickvals=x_labels,
                        ticktext=x_labels,
                        tickangle=-45,
                        tickfont={"size": 11, "color": "black"},
                        title_font={"size": 14, "color": "black"}
                    ),
                    yaxis=dict(
                        tickfont={"size": 12, "color": "black"},
                        title_font={"size": 14, "color": "black"}
                    ),
                    margin=dict(b=120, t=80, l=60, r=40)
                )
                st.plotly_chart(fig, use_container_width=True)

            # MDD 차트
            if analysis.get("drawdown_series") is not None:
                st.markdown("---")
                st.subheader("📉 최대 손실폭 (MDD)")

                drawdown = analysis["drawdown_series"]
                col1, col2 = st.columns([3, 1])
                with col1:
                    mdd_value = result_data.get("performance_metrics", {}).get("최대 낙폭 (MDD)", "N/A")
                    st.metric("현재 MDD", mdd_value)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values,
                    mode="lines",
                    fill="tozeroy",
                    fillcolor="rgba(211, 47, 47, 0.3)",
                    line=dict(color="#d32f2f", width=2),
                    name="드로우다운"
                ))
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                fig.update_layout(
                    xaxis_title="날짜",
                    yaxis_title="드로우다운 (%)",
                    height=400,
                    showlegend=False,
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)

            # 드로우다운 이벤트 테이블
            if analysis.get("drawdown_events") is not None and len(analysis["drawdown_events"]) > 0:
                st.markdown("---")
                st.subheader("📋 포트폴리오 드로우다운")

                portfolio_value = result_data.get("portfolio_value")

                events = analysis["drawdown_events"][:10]
                events_data = []
                for i, event in enumerate(events, 1):
                    start_date = event["start"]
                    trough_date = event["trough"]
                    end_date = event["end"]

                    # 회복시점 계산
                    recovery_point = None
                    if portfolio_value is not None and start_date in portfolio_value.index:
                        peak_value = portfolio_value.loc[:start_date].max()

                        if end_date in portfolio_value.index:
                            if portfolio_value.loc[end_date] >= peak_value:
                                recovery_point = end_date
                            elif end_date < portfolio_value.index[-1]:
                                future_dates = portfolio_value.index[portfolio_value.index > end_date]
                                for date in future_dates:
                                    if portfolio_value.loc[date] >= peak_value:
                                        recovery_point = date
                                        break

                    # 손실기간
                    if recovery_point is not None and recovery_point > start_date:
                        loss_months = (recovery_point.year - start_date.year) * 12 + (recovery_point.month - start_date.month)
                    else:
                        loss_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)

                    # 회복기간
                    if recovery_point is not None and recovery_point > trough_date:
                        recovery_months = (recovery_point.year - trough_date.year) * 12 + (recovery_point.month - trough_date.month)
                    else:
                        recovery_months = None

                    events_data.append({
                        "순위": i,
                        "시작": start_date.strftime("%Y/%m"),
                        "종료": end_date.strftime("%Y/%m"),
                        "회복시점": recovery_point.strftime("%Y/%m") if recovery_point is not None else "-",
                        "회복기간": f"{recovery_months} 개월" if recovery_months is not None else "-",
                        "손실기간": f"{loss_months} 개월",
                        "드로우다운": f"{event['drawdown']:.1f}%"
                    })

                events_df = pd.DataFrame(events_data)
                st.dataframe(events_df, use_container_width=True, hide_index=True)
else:
    st.info("👈 왼쪽 사이드바에서 보유 금액을 입력하고 '실행' 버튼을 클릭하세요.")

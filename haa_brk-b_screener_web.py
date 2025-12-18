import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px


def calculate_momentum_scores(data: pd.DataFrame) -> pd.DataFrame:
    """모멘텀 점수 계산"""
    aligned_returns = []
    # 1개월, 3개월, 6개월, 12개월을 각각 약 21, 63, 126, 252 거래일로 계산
    for months in [1, 3, 6, 12]:
        returns = data.pct_change(periods=months * 21)
        # 컬럼을 MultiIndex(티커, 기간) 형태로 변경
        returns.columns = pd.MultiIndex.from_tuples(
            [(col, f"{months}M") for col in data.columns],
            names=["Ticker", "Period"]
        )
        aligned_returns.append(returns)

    # 네 개의 수익률 테이블을 가로로 붙임
    aligned_data = pd.concat(aligned_returns, axis=1)
    # 티커별 수익률 평균 = 모멘텀 점수
    return aligned_data.T.groupby(level="Ticker").mean().T


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


def run_screener(total_balance: float):
    """스크리너 실행"""
    tickers = [
        "SPY", "VEA", "VWO", "IWM",
        "BIL", "IEF", "TLT", "TIP",
        "PDBC", "VNQ", "BRK-B"
    ]

    # 1) 과거 데이터 다운로드
    start_date = "2014-11-01"
    end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
    
    with st.spinner("데이터를 다운로드하는 중..."):
        # 수정주가(Adj Close) 다운로드 - 배당/분할을 반영한 가격
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False
        )["Adj Close"]  # 수정주가 사용 (모멘텀 점수 계산에 사용)
        data.index = data.index.tz_localize(None)

        # 2) 오늘 장중 가격(fast_info)을 마지막 행으로 추가
        today = pd.Timestamp.now().normalize()
        fast_prices = {}
        for t in tickers:
            try:
                ticker = yf.Ticker(t)
                # 장중 가격 직접 접근 (실패 시 예외 발생)
                fast_prices[t] = ticker.fast_info["last_price"]
            except:
                # fast_info 실패 시 마지막 가격 사용 (장중 가격이 아닐 수 있음)
                fast_prices[t] = data[t].iloc[-1]
        
        if today not in data.index:
            data.loc[today] = pd.Series(fast_prices)
            data.sort_index(inplace=True)

        # 3) 모멘텀 점수 계산
        momentum_scores = calculate_momentum_scores(data)

        # 4) TIP 기준 자산 선택 (offense/defense) 및 target_date 결정
        selected_assets, target_date = select_assets(momentum_scores, data)

        # 5) 백테스트 실행
        portfolio_value, rebalancing_history, performance_metrics, analysis_data = run_backtest(
            data, momentum_scores, total_balance
        )
        
        # 6) 최근 12개월 리밸런싱 내역
        recent_rebalancing = get_recent_rebalancing_history(data, momentum_scores, months=12)

        # 7) 결과 요약 및 테이블 생성
        return display_results(
            momentum_scores,
            data,
            selected_assets,
            tickers,
            total_balance,
            target_date,
            portfolio_value,
            rebalancing_history,
            performance_metrics,
            recent_rebalancing,
            analysis_data
        )


def display_results(
    momentum_scores: pd.DataFrame,
    data: pd.DataFrame,
    selected_assets: list,
    tickers: list,
    total_balance: float,
    target_date: pd.Timestamp,
    portfolio_value: pd.Series = None,
    rebalancing_history: list = None,
    performance_metrics: dict = None,
    recent_rebalancing: list = None,
    analysis_data: dict = None
):
    """결과 표시 및 데이터 반환"""
    haa_bal = total_balance * 0.8
    
    # 선택된 자산 데이터 준비
    selected_data = []
    for asset, score in selected_assets:
        price = data.loc[target_date, asset]
        shares = haa_bal / len(selected_assets) / price
        purchase_amount = haa_bal / len(selected_assets)
        selected_data.append({
            "자산": asset,
            "모멘텀 점수": f"{score:.3f}",
            "현재 가격": f"${price:.2f}",
            "구매 수량": f"{shares:.2f}",
            "구매 금액": f"${purchase_amount:,.2f}"
        })
    
    # BRK-B 모멘텀 점수 계산
    brk_price = data.loc[target_date, "BRK-B"]
    brk_shares = total_balance * 0.2 / brk_price
    brk_purchase_amount = total_balance * 0.2
    brk_momentum = momentum_scores.loc[target_date, "BRK-B"]
    selected_data.append({
        "자산": "BRK-B",
        "모멘텀 점수": f"{brk_momentum:.3f}",
        "현재 가격": f"${brk_price:.2f}",
        "구매 수량": f"{brk_shares:.2f}",
        "구매 금액": f"${brk_purchase_amount:,.2f}"
    })
    
    # 반환할 데이터 준비
    result_data = {
        "target_date": target_date,
        "total_balance": total_balance,
        "selected_data": selected_data,
        "momentum_scores": momentum_scores,
        "data": data,
        "tickers": tickers,
        "selected_assets": selected_assets,
        "haa_bal": haa_bal,
        "brk_shares": brk_shares,
        "portfolio_value": portfolio_value,
        "rebalancing_history": rebalancing_history,
        "performance_metrics": performance_metrics,
        "recent_rebalancing": recent_rebalancing,
        "analysis_data": analysis_data
    }

    # ==== 아래쪽: 전체 자산군 테이블 생성 ====
    st.subheader("📈 전체 자산군 분석")
    recent = data.loc[target_date]
    df = pd.DataFrame({
        "Recent Price": recent,
        "Momentum Score": momentum_scores.loc[target_date],
        "1M (%)": data.pct_change(21).loc[target_date] * 100,
        "3M (%)": data.pct_change(63).loc[target_date] * 100,
        "6M (%)": data.pct_change(126).loc[target_date] * 100,
        "12M (%)": data.pct_change(252).loc[target_date] * 100,
    })
    df = df.loc[tickers]

    # ---- 순위 설정 ----
    # 공격 자산군: SPY, VEA, VWO, IWM, TLT, PDBC, VNQ, IEF
    off_idx = ["SPY", "VEA", "VWO", "IWM", "TLT", "PDBC", "VNQ", "IEF"]
    # 방어 자산군: IEF, BIL
    def_idx = ["IEF", "BIL"]

    # Rank 컬럼 초기화
    df["Rank"] = ""
    
    # 공격군 중 상위 4개
    for i, t in enumerate(df.loc[off_idx].nlargest(4, "Momentum Score").index, 1):
        df.loc[t, "Rank"] = f"공격{i}위"
    # 방어군 중 상위 1개
    for i, t in enumerate(df.loc[def_idx].nlargest(1, "Momentum Score").index, 1):
        df.loc[t, "Rank"] = f"방어{i}위"

    # TIP: 공격/대피 로직
    tip_val = momentum_scores.loc[target_date, "TIP"]
    df.loc["TIP", "Rank"] = "공격" if tip_val >= 0 else "대피"

    # BRK-B: 항상 보유
    df.loc["BRK-B", "Rank"] = "보유"

    # ---- 구매 수량 계산 ----
    df["Shares to Buy"] = ""
    for asset, _ in selected_assets:
        price = recent[asset]
        shares = haa_bal / len(selected_assets) / price
        df.loc[asset, "Shares to Buy"] = f"{shares:.2f}"
    df.loc["BRK-B", "Shares to Buy"] = f"{brk_shares:.2f}"

    # 컬럼 순서 재정렬
    df = df[["Rank", "Recent Price", "Momentum Score", "1M (%)", "3M (%)", "6M (%)", "12M (%)", "Shares to Buy"]]
    
    # 숫자 포맷팅
    df["Recent Price"] = df["Recent Price"].apply(lambda x: f"${x:,.2f}")
    df["Momentum Score"] = df["Momentum Score"].apply(lambda x: f"{x:.3f}")
    for col in ["1M (%)", "3M (%)", "6M (%)", "12M (%)"]:
        df[col] = df[col].apply(lambda x: f"{x:.2f}%")

    result_data["df"] = df
    return result_data


def run_backtest(data: pd.DataFrame, momentum_scores: pd.DataFrame, initial_balance: float = 10000.0):
    """HAA 전략 백테스트 실행"""
    try:
        # momentum_scores에 데이터가 있는 날짜만 필터링
        # momentum_scores는 모멘텀 계산을 위해 최소 252일(12개월) 데이터가 필요하므로
        # 첫 번째 날짜는 momentum_scores의 첫 번째 날짜 이후여야 함
        if len(momentum_scores.index) > 0:
            first_valid_date = momentum_scores.index[0]
            # first_valid_date 이후의 데이터만 사용
            data_filtered = data[data.index >= first_valid_date].copy()
        else:
            data_filtered = data.copy()
        
        # 월말 날짜 추출 (매월 마지막 거래일)
        monthly_dates = data_filtered.resample('M').last().index
        
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
            prev_date = monthly_dates[i-1]
            
            # 실제 거래일 찾기 (월말 날짜가 주말이면 가장 가까운 이전 거래일 사용)
            if prev_date not in data_filtered.index:
                available_dates = data_filtered.index[data_filtered.index <= prev_date]
                if len(available_dates) > 0:
                    prev_date_actual = available_dates[-1]
                else:
                    continue  # 데이터가 없으면 스킵
            else:
                prev_date_actual = prev_date
            
            if current_date not in data_filtered.index:
                available_dates = data_filtered.index[data_filtered.index <= current_date]
                if len(available_dates) > 0:
                    current_date_actual = available_dates[-1]
                else:
                    continue  # 데이터가 없으면 스킵
            else:
                current_date_actual = current_date
            
            # 이전 달 말에 선택된 자산 (리밸런싱 시점)
            # 리밸런싱은 prev_date_actual에 이루어지고, prev_date_actual부터 current_date_actual까지 보유
            try:
                selected_assets, _ = select_assets(momentum_scores, data_filtered, prev_date_actual)
            except Exception as e:
                # 자산 선택 실패 시 이전 자산 유지 또는 스킵
                st.warning(f"날짜 {prev_date_actual}에서 자산 선택 실패: {e}")
                continue
            
            # HAA 80% + BRK-B 20% 구성
            haa_weight = 0.8
            brk_weight = 0.2
            
            # 각 자산의 월간 수익률 계산 (prev_date_actual부터 current_date_actual까지)
            haa_return = 0.0
            haa_count = 0
            
            # HAA 자산들의 수익률 (균등 비중)
            if selected_assets:
                for asset, _ in selected_assets:
                    if asset in data_filtered.columns:
                        # prev_date_actual와 current_date_actual의 가격 확인
                        if prev_date_actual in data_filtered.index and current_date_actual in data_filtered.index:
                            prev_price = data_filtered.loc[prev_date_actual, asset]
                            curr_price = data_filtered.loc[current_date_actual, asset]
                            if not pd.isna(prev_price) and not pd.isna(curr_price) and prev_price > 0:
                                asset_return = (curr_price / prev_price) - 1
                                haa_return += asset_return
                                haa_count += 1
            
            # HAA 평균 수익률 계산 (데이터가 있는 자산만)
            if haa_count > 0:
                haa_return = haa_return / haa_count
            else:
                # 자산이 선택되었지만 데이터가 없는 경우 - 이전 달 수익률 유지 또는 0
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
            
            # 디버깅: 월별 수익률이 0인 경우 확인
            if abs(portfolio_return) < 0.0001:  # 거의 0인 경우
                selected_asset_names = [asset for asset, _ in selected_assets] if selected_assets else []
                st.warning(f"⚠️ {current_date.strftime('%Y-%m')}: 포트폴리오 수익률 0% (선택된 자산: {selected_asset_names}, HAA 수익률: {haa_return*100:.2f}%, BRK-B 수익률: {brk_return*100:.2f}%, 데이터 있는 자산 수: {haa_count})")
            
            # 포트폴리오 가치 업데이트 (복리 계산)
            if portfolio_value.iloc[i-1] > 0:
                portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1 + portfolio_return)
            else:
                portfolio_value.iloc[i] = portfolio_value.iloc[i-1]
            
            # 리밸런싱 내역 저장 (간단한 형식)
            haa_assets = len(selected_assets)
            if haa_assets > 0:
                haa_weight_per_asset = 0.8 / haa_assets
                asset_weights = []
                # selected_assets는 이미 모멘텀 점수 순서대로 정렬되어 있음
                for rank, (asset, score) in enumerate(selected_assets, 1):
                    asset_weights.append(f"{asset}({haa_weight_per_asset*100:.0f}% {rank}위)")
                asset_weights.append(f"BRK-B(20% 보유)")
                asset_str = ", ".join(asset_weights)
            else:
                asset_str = "BRK-B(20% 보유)"
            
            rebalancing_history.append({
                "적용 시점": current_date.strftime('%Y-%m-%d'),
                "목표 자산 비중": asset_str
            })
        
        # 성과 지표 계산
        initial_value = portfolio_value.iloc[0]
        final_value = portfolio_value.iloc[-1]
        total_return = (final_value / initial_value) - 1
        
        # 기간 계산 (정확한 연수)
        start_date = monthly_dates[0]
        end_date = monthly_dates[-1]
        days_diff = (end_date - start_date).days
        years = days_diff / 365.25
        
        # CAGR 계산: (최종값/초기값)^(1/년수) - 1
        if years > 0 and initial_value > 0:
            # 복리 수익률 계산
            cagr = ((final_value / initial_value) ** (1 / years)) - 1
        else:
            cagr = 0
        
        # 검증: 월별 수익률로부터 CAGR 재계산
        monthly_returns_series = portfolio_value.pct_change().dropna()
        if len(monthly_returns_series) > 0:
            # 월별 수익률의 기하평균을 연환산
            # (1 + r1) * (1 + r2) * ... * (1 + rn) = 최종값/초기값
            # 월별 기하평균 = (최종값/초기값)^(1/n) - 1
            # 연환산 CAGR = (1 + 월별기하평균)^12 - 1
            total_return_from_monthly = (1 + monthly_returns_series).prod() - 1
            monthly_geometric_mean = (1 + monthly_returns_series).prod() ** (1 / len(monthly_returns_series)) - 1
            cagr_from_monthly = ((1 + monthly_geometric_mean) ** 12) - 1
            
            # 디버깅 정보 (필요시 주석 해제)
            # st.info(f"디버깅: 월별 수익률 개수={len(monthly_returns_series)}, 총 수익률(월별)={total_return_from_monthly*100:.2f}%, CAGR(월별)={cagr_from_monthly*100:.2f}%")
            
            # 두 방법의 차이가 크면 경고
            if abs(cagr - cagr_from_monthly) > 0.01:  # 1% 이상 차이
                st.warning(f"⚠️ CAGR 계산 검증: 직접 계산={cagr*100:.2f}%, 월별 기하평균={cagr_from_monthly*100:.2f}%")
        
        # 월별 수익률 계산
        monthly_returns_series = portfolio_value.pct_change().dropna()
        
        # 연환산 변동성: 월별 수익률의 표준편차 * sqrt(12)
        if len(monthly_returns_series) > 1:
            volatility = monthly_returns_series.std() * np.sqrt(12)
        else:
            volatility = 0
        
        # 최대 낙폭 (MDD)
        cumulative = (1 + monthly_returns_series).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        mdd = drawdown.min()
        
        # 샤프 비율 (무위험 수익률 0% 가정)
        sharpe = (cagr / volatility) if volatility > 0 else 0
        
        performance_metrics = {
            "총 수익률": f"{total_return*100:.2f}%",
            "CAGR": f"{cagr*100:.2f}%",
            "연환산 변동성": f"{volatility*100:.2f}%",
            "샤프 비율": f"{sharpe:.2f}",
            "최대 낙폭 (MDD)": f"{mdd*100:.2f}%",
            "시작일": monthly_dates[0].strftime('%Y-%m-%d'),
            "종료일": monthly_dates[-1].strftime('%Y-%m-%d'),
            "기간 (년)": f"{years:.2f}"
        }
        
        # 추가 분석 데이터 계산
        yearly_returns = calculate_yearly_returns(portfolio_value)
        monthly_returns = calculate_monthly_returns(portfolio_value)
        monthly_heatmap = create_monthly_heatmap_data(monthly_returns)
        drawdown_series, drawdown_events = calculate_drawdown_events(portfolio_value)
        monthly_distribution = create_monthly_distribution(monthly_returns)
        
        return portfolio_value, rebalancing_history, performance_metrics, {
            'yearly_returns': yearly_returns,
            'monthly_returns': monthly_returns,
            'monthly_heatmap': monthly_heatmap,
            'drawdown_series': drawdown_series,
            'drawdown_events': drawdown_events,
            'monthly_distribution': monthly_distribution
        }
    except Exception as e:
        st.error(f"백테스트 실행 중 오류 발생: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None, None


def calculate_yearly_returns(portfolio_value):
    """연도별 수익률 계산"""
    if portfolio_value is None or len(portfolio_value) < 2:
        return None
    
    yearly = portfolio_value.resample("YE").last()
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
    monthly_returns_df['year'] = monthly_returns_df.index.year
    monthly_returns_df['month'] = monthly_returns_df.index.month
    
    # 피벗 테이블 생성 (연도 x 월)
    heatmap_data = monthly_returns_df.pivot_table(
        values='return',
        index='year',
        columns='month',
        aggfunc='first'
    )
    
    # 컬럼 이름을 월 이름으로 변경 (있는 월만)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_dict = {i: month_names[i-1] for i in range(1, 13)}
    
    # 실제 데이터가 있는 월만 선택
    available_months = [month_dict[i] for i in heatmap_data.columns if i in month_dict]
    heatmap_data.columns = [month_dict[i] if i in month_dict else f"Month_{i}" 
                          for i in heatmap_data.columns]
    
    # 있는 월만 유지
    heatmap_data = heatmap_data[[col for col in heatmap_data.columns if col in month_names]]
    
    # 연도 순서 역순 (최신 연도가 아래로)
    heatmap_data = heatmap_data.sort_index(ascending=False)
    
    # 평균 행 계산 (NaN 값 제외하고 계산)
    monthly_avg = heatmap_data.mean(axis=0, skipna=True)
    avg_row = pd.DataFrame([monthly_avg.values], index=['평균'], columns=heatmap_data.columns)
    
    # 평균 행을 맨 앞에 추가 (Y축 역순이므로 맨 앞이 차트 하단에 표시됨)
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
    
    # 드로우다운 이벤트 찾기 (새로운 드로우다운 시작)
    drawdown_events = []
    in_drawdown = False
    drawdown_start = None
    drawdown_start_value = None
    max_drawdown = 0
    
    for i, (date, dd_value) in enumerate(drawdown.items()):
        if dd_value < 0 and not in_drawdown:
            # 드로우다운 시작
            in_drawdown = True
            drawdown_start = date
            drawdown_start_value = cumulative.loc[date]
            max_drawdown = dd_value
        elif dd_value < max_drawdown and in_drawdown:
            # 더 깊은 드로우다운
            max_drawdown = dd_value
        elif dd_value >= 0 and in_drawdown:
            # 드로우다운 종료
            # 최대 드로우다운 시점 찾기
            drawdown_period = drawdown.loc[drawdown_start:date]
            trough_date = drawdown_period.idxmin()
            trough_value = drawdown_period.min()
            
            drawdown_events.append({
                'start': drawdown_start,
                'trough': trough_date,
                'end': date,
                'drawdown': trough_value
            })
            in_drawdown = False
            max_drawdown = 0
    
    # 진행 중인 드로우다운 처리
    if in_drawdown:
        drawdown_period = drawdown.loc[drawdown_start:]
        trough_date = drawdown_period.idxmin()
        trough_value = drawdown_period.min()
        drawdown_events.append({
            'start': drawdown_start,
            'trough': trough_date,
            'end': portfolio_value.index[-1],
            'drawdown': trough_value
        })
    
    # 드로우다운 크기순으로 정렬
    drawdown_events.sort(key=lambda x: x['drawdown'])
    
    return drawdown, drawdown_events


def create_monthly_distribution(monthly_returns):
    """월별 수익률 분포 히스토그램 데이터 생성"""
    if monthly_returns is None or len(monthly_returns) == 0:
        return None
    
    # 히스토그램 구간 설정 (-10% ~ 10%, 1% 간격으로 세분화)
    bins = np.arange(-10, 11, 1)  # -10, -9, -8, ..., 9, 10
    hist, bin_edges = np.histogram(monthly_returns.values, bins=bins)
    
    # 중간값 계산 (각 구간의 중간값)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # 구간 레이블 생성 (예: "[-10%, -9%)", "[-9%, -8%)", ..., "[0%, 1%)", "[1%, 2%)")
    bin_labels = []
    for i in range(len(bin_edges) - 1):
        left = int(bin_edges[i])
        right = int(bin_edges[i + 1])
        if right == 10:
            bin_labels.append(f"[{left}%, {right}%]")
        else:
            bin_labels.append(f"[{left}%, {right}%)")
    
    return pd.DataFrame({
        'bin_center': bin_centers,
        'bin_label': bin_labels,
        'count': hist
    })


def get_asset_full_name(ticker: str) -> str:
    """티커의 전체 이름 반환"""
    asset_names = {
        "SPY": "SPDR S&P 500 ETF Trust",
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


def get_recent_rebalancing_history(data: pd.DataFrame, momentum_scores: pd.DataFrame, months: int = 12):
    """최근 N개월 리밸런싱 내역 추출"""
    # 현재 날짜 기준으로 과거 N개월 (현재 월 포함)
    end_date = pd.Timestamp.now().normalize()
    start_date = end_date - pd.DateOffset(months=months)
    
    # 월말 날짜 추출
    monthly_dates = data.resample('M').last().index
    monthly_dates = monthly_dates[(monthly_dates >= start_date) & (monthly_dates <= end_date)]
    
    # 현재 날짜가 포함된 월의 마지막 거래일도 추가
    current_month_end = data.resample('M').last().index[-1] if len(data) > 0 else None
    if current_month_end and current_month_end not in monthly_dates and current_month_end >= start_date:
        monthly_dates = pd.Index(list(monthly_dates) + [current_month_end]).sort_values()
    
    if len(monthly_dates) == 0:
        return []
    
    rebalancing_history = []
    
    for date in monthly_dates:
        # momentum_scores에 없으면 가장 가까운 이전 날짜 사용
        target_date = date
        if target_date not in momentum_scores.index:
            available_dates = momentum_scores.index[momentum_scores.index <= target_date]
            if len(available_dates) > 0:
                target_date = available_dates[-1]
            else:
                continue
            
        selected_assets, _ = select_assets(momentum_scores, data, target_date)
        
        # 비중 계산 및 순위 표시 (모멘텀 점수 순서대로)
        haa_assets = len(selected_assets)
        if haa_assets > 0:
            haa_weight_per_asset = 0.8 / haa_assets
            asset_weights = []
            # selected_assets는 이미 모멘텀 점수 순서대로 정렬되어 있음
            for rank, (asset, score) in enumerate(selected_assets, 1):
                asset_weights.append(f"{asset}({haa_weight_per_asset*100:.0f}% {rank}위)")
            asset_weights.append(f"BRK-B(20% 보유)")
            asset_str = ", ".join(asset_weights)
        else:
            asset_str = "BRK-B(20% 보유)"
        
        rebalancing_history.append({
            "적용 시점": date.strftime('%Y-%m-%d'),
            "목표 자산 비중": asset_str
        })
    
    # 최신순으로 정렬
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

# 사이드바에 입력 필드
with st.sidebar:
    st.header("⚙️ 설정")
    balance_text = st.text_input(
        "보유 금액 입력",
        value="10000",
        help="입력하지 않으면 기본값 10000이 사용됩니다."
    )
    
    if st.button("🚀 실행", type="primary", use_container_width=True):
        try:
            total_balance = float(balance_text) if balance_text else 10000.0
            if total_balance <= 0:
                st.error("보유 금액은 0보다 커야 합니다.")
            else:
                result_data = run_screener(total_balance)
                st.session_state['result_data'] = result_data
                st.session_state['balance'] = total_balance
        except ValueError:
            st.error("올바른 숫자를 입력해주세요.")
        except Exception as e:
            st.error(f"오류 발생: {e}")
    
    if st.button("🔄 초기화", use_container_width=True):
        if 'result_data' in st.session_state:
            del st.session_state['result_data']
        if 'balance' in st.session_state:
            del st.session_state['balance']
        st.rerun()
    
    # ==== 사이드바에 설정 정보 표시 ====
    if 'result_data' in st.session_state:
        st.markdown("---")
        st.subheader("📊 설정 정보")
        result_data = st.session_state['result_data']
        st.metric("기준 날짜", result_data['target_date'].strftime('%Y-%m-%d'))
        st.metric("보유 금액", f"${result_data['total_balance']:,.2f}")
        
        st.markdown("---")
        st.subheader("✅ 선택된 자산")
        selected_df = pd.DataFrame(result_data['selected_data'])
        st.dataframe(selected_df, use_container_width=True, hide_index=True)

# 메인 영역에 결과 표시
if 'result_data' in st.session_state:
    result_data = st.session_state['result_data']
    
    # ==== 본문에 기준 날짜와 투자 금액 표시 ====
    st.subheader("📊 설정 정보")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("기준 날짜", result_data['target_date'].strftime('%Y-%m-%d'))
    with col2:
        st.metric("보유 금액", f"${result_data['total_balance']:,.2f}")
    
    st.markdown("---")
    
    # ==== 선택된 자산 표시 ====
    st.subheader("✅ 선택된 자산")
    st.dataframe(
        pd.DataFrame(result_data['selected_data']),
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    
    # ==== 전체 자산군 분석 테이블 ====
    st.subheader("📈 전체 자산군 분석")
    st.dataframe(
        result_data['df'],
        use_container_width=True,
        height=400
    )
    
    # CSV 다운로드 버튼
    csv = result_data['df'].to_csv(index=True)
    st.download_button(
        label="📥 CSV로 다운로드",
        data=csv,
        file_name=f"haa_screener_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # ==== 백테스트 성과 지표 ====
    if result_data.get('performance_metrics'):
        st.subheader("📊 백테스트 성과 지표")
        metrics = result_data['performance_metrics']
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("CAGR", metrics.get("CAGR", "N/A"))
            st.metric("총 수익률", metrics.get("총 수익률", "N/A"))
        with col2:
            st.metric("연환산 변동성", metrics.get("연환산 변동성", "N/A"))
            st.metric("샤프 비율", metrics.get("샤프 비율", "N/A"))
        with col3:
            st.metric("최대 낙폭 (MDD)", metrics.get("최대 낙폭 (MDD)", "N/A"))
            st.metric("기간", metrics.get("기간 (년)", "N/A") + "년")
        with col4:
            st.metric("시작일", metrics.get("시작일", "N/A"))
            st.metric("종료일", metrics.get("종료일", "N/A"))
    
    st.markdown("---")
    
    # ==== 최근 리밸런싱 및 리밸런싱 추이 ====
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 최근 리밸런싱")
        # 도넛 차트 생성
        current_selected_assets = result_data.get('selected_assets', [])
        if current_selected_assets:
            asset_names = [asset for asset, _ in current_selected_assets]
            asset_names.append("BRK-B")
            
            # 비중 계산
            haa_assets = len(current_selected_assets)
            if haa_assets > 0:
                haa_weight_per_asset = 0.8 / haa_assets
                values = [haa_weight_per_asset * 100] * haa_assets
                labels = [asset for asset, _ in current_selected_assets]
            else:
                values = []
                labels = []
            
            values.append(20.0)  # BRK-B 20%
            labels.append("BRK-B")
            
            # 도넛 차트
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.5,
                textinfo='label+percent',
                textposition='outside'
            )])
            fig.update_layout(
                height=400,
                showlegend=True,
                margin=dict(t=0, b=0, l=0, r=0)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 리밸런싱 추이")
        if result_data.get('recent_rebalancing'):
            rebal_df = pd.DataFrame(result_data['recent_rebalancing'])
            st.dataframe(
                rebal_df,
                use_container_width=True,
                hide_index=True,
                height=400
            )
        else:
            st.info("리밸런싱 내역이 없습니다.")
    
    # ==== 백테스트 포트폴리오 가치 차트 ====
    if result_data.get('portfolio_value') is not None:
        st.markdown("---")
        st.subheader("📈 백테스트 포트폴리오 가치 추이")
        portfolio_df = result_data['portfolio_value'].to_frame("포트폴리오 가치")
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
        analysis = result_data.get('analysis_data')
        if analysis:
            # 연도별 수익률
            if analysis.get('yearly_returns') is not None and len(analysis['yearly_returns']) > 0:
                st.markdown("---")
                st.subheader("📊 연도별 수익률 (%)")
                yearly_df = analysis['yearly_returns'].to_frame("수익률")
                yearly_df.index = yearly_df.index.year
                
                # 색상 설정 (양수: 초록, 음수: 빨강)
                colors = ['#d32f2f' if x < 0 else '#2e7d32' for x in yearly_df['수익률']]
                
                # 연도 레이블을 "2022년" 형식으로 변경
                year_labels = [f"{int(year)}년" for year in yearly_df.index]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=year_labels,
                    y=yearly_df['수익률'],
                    marker_color=colors,
                    text=[f"{x:.1f}%" for x in yearly_df['수익률']],
                    textposition='outside',
                    textfont={"size": 13, "color": "black", "family": "Arial Black"},
                    name='연도별 수익률'
                ))
                fig.update_layout(
                    xaxis_title=dict(text="연도", font=dict(size=14, color="black")),
                    yaxis_title=dict(text="수익률 (%)", font=dict(size=14, color="black")),
                    height=450,
                    showlegend=False,
                    hovermode='x unified',
                    xaxis=dict(
                        tickfont={"size": 12, "color": "black"},
                        title_font={"size": 14, "color": "black"}
                    ),
                    yaxis=dict(
                        tickfont={"size": 12, "color": "black"},
                        title_font={"size": 14, "color": "black"}
                    ),
                    margin=dict(b=60, t=80, l=60, r=40)  # 상하좌우 여백 증가 (텍스트가 잘리지 않도록)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # 월별 수익률 히트맵
            if analysis.get('monthly_heatmap') is not None and not analysis['monthly_heatmap'].empty:
                st.markdown("---")
                st.subheader("📅 월별 수익률 (%)")
                
                heatmap_data = analysis['monthly_heatmap']
                
                # Y축 레이블 생성 (연도는 정수로만, 평균은 그대로)
                y_labels = []
                y_positions = []
                for pos, idx in enumerate(heatmap_data.index):
                    if idx == '평균':
                        y_labels.append('평균')
                    else:
                        try:
                            year_int = int(float(idx))
                            y_labels.append(str(year_int))
                        except:
                            y_labels.append(str(idx))
                    y_positions.append(pos)
                
                # 최소값과 최대값 계산 (NaN 제외)
                z_min = np.nanmin(heatmap_data.values)
                z_max = np.nanmax(heatmap_data.values)
                
                # z 값을 정규화하여 음수는 0~0.5, 양수는 0.5~1로 매핑
                # 이렇게 하면 0을 기준으로 색상이 확실히 분리됨
                z_normalized = heatmap_data.values.copy()
                abs_min = abs(z_min) if z_min < 0 else 0
                abs_max = abs(z_max) if z_max > 0 else 0
                max_abs = max(abs_min, abs_max)
                
                for i in range(len(z_normalized)):
                    for j in range(len(z_normalized[i])):
                        val = z_normalized[i, j]
                        if not pd.isna(val):
                            if val < 0:
                                # 음수: 0~0.5 범위로 정규화
                                if abs_min > 0:
                                    z_normalized[i, j] = 0.5 * (1 - abs(val) / abs_min)
                                else:
                                    z_normalized[i, j] = 0.5
                            elif val > 0:
                                # 양수: 0.5~1 범위로 정규화
                                if abs_max > 0:
                                    z_normalized[i, j] = 0.5 + 0.5 * (val / abs_max)
                                else:
                                    z_normalized[i, j] = 0.5
                            else:
                                # 0
                                z_normalized[i, j] = 0.5
                
                # 색상 스케일 설정: 음수는 빨강 계열, 양수는 초록 계열
                fig = go.Figure(data=go.Heatmap(
                    z=z_normalized,
                    x=heatmap_data.columns,
                    y=y_positions,
                    colorscale=[
                        [0.0, '#b71c1c'],      # 진한 빨강 (최소값)
                        [0.2, '#d32f2f'],     # 빨강
                        [0.4, '#ffcdd2'],     # 연한 빨강
                        [0.5, '#ffffff'],     # 흰색 (0)
                        [0.6, '#c8e6c9'],     # 연한 초록
                        [0.8, '#2e7d32'],     # 초록
                        [1.0, '#1b5e20']      # 진한 초록 (최대값)
                    ],
                    text=[[f"<b>{val:.1f}%</b>" if not pd.isna(val) else "" for val in row] 
                          for row in heatmap_data.values],
                    texttemplate='%{text}',
                    textfont={"size": 16, "color": "black", "family": "Arial Black"},
                    colorbar=dict(
                        title=dict(text="수익률 (%)", font=dict(size=13, color="black")),
                        tickmode='array',
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
                        autorange='reversed',
                        tickmode='array',
                        tickvals=y_positions,
                        ticktext=y_labels,
                        tickfont={"size": 12, "color": "black"},
                        title_font={"size": 14, "color": "black"},
                        dtick=None
                    ),
                    margin=dict(b=60, t=60, l=80, r=80)  # 상하좌우 여백 증가 (텍스트가 잘리지 않도록)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # 월별 수익률 분포
            if analysis.get('monthly_distribution') is not None:
                st.markdown("---")
                st.subheader("📊 월별 수익률 분포")
                
                dist_data = analysis['monthly_distribution']
                total_count = dist_data['count'].sum()
                
                # 비율 계산
                dist_data['percentage'] = (dist_data['count'] / total_count * 100) if total_count > 0 else 0
                
                # 색상 설정 (음수: 빨강, 양수: 초록, 0은 초록)
                colors = ['#d32f2f' if x < 0 else '#2e7d32' for x in dist_data['bin_center']]
                
                # X축 레이블 생성 (구간 형식: "[-10%, -9%)", "[-9%, -8%)", ..., "[0%, 1%)", "[1%, 2%)")
                x_labels = dist_data['bin_label'].tolist()
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=x_labels,
                    y=dist_data['count'],
                    marker_color=colors,
                    name='빈도',
                    text=[f"{pct:.1f}%" if count > 0 else "" for count, pct in zip(dist_data['count'], dist_data['percentage'])],
                    textposition='outside',
                    textfont={"size": 13, "color": "black", "family": "Arial Black"},
                    hovertemplate='구간: %{x}<br>빈도: %{y}회<br>비율: %{customdata:.1f}%<extra></extra>',
                    customdata=dist_data['percentage']
                ))
                fig.update_layout(
                    xaxis_title=dict(text="수익률 구간 (%)", font=dict(size=14, color="black")),
                    yaxis_title=dict(text="빈도 (회)", font=dict(size=14, color="black")),
                    height=550,
                    showlegend=False,
                    hovermode='x unified',
                    xaxis=dict(
                        tickmode='array',
                        tickvals=x_labels,
                        ticktext=x_labels,
                        tickangle=-45,  # 레이블 회전
                        tickfont={"size": 11, "color": "black"},  # X축 레이블 크기 및 색상
                        title_font={"size": 14, "color": "black"}
                    ),
                    yaxis=dict(
                        tickfont={"size": 12, "color": "black"},  # Y축 레이블 크기 및 색상
                        title_font={"size": 14, "color": "black"}
                    ),
                    margin=dict(b=120, t=80, l=60, r=40)  # 상하좌우 여백 증가 (텍스트가 잘리지 않도록)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # MDD 차트
            if analysis.get('drawdown_series') is not None:
                st.markdown("---")
                st.subheader("📉 최대 손실폭 (MDD)")
                
                drawdown = analysis['drawdown_series']
                col1, col2 = st.columns([3, 1])
                with col1:
                    mdd_value = result_data.get('performance_metrics', {}).get('최대 낙폭 (MDD)', 'N/A')
                    st.metric("현재 MDD", mdd_value)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values,
                    mode='lines',
                    fill='tozeroy',
                    fillcolor='rgba(211, 47, 47, 0.3)',
                    line=dict(color='#d32f2f', width=2),
                    name='드로우다운'
                ))
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                fig.update_layout(
                    xaxis_title="날짜",
                    yaxis_title="드로우다운 (%)",
                    height=400,
                    showlegend=False,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # 드로우다운 이벤트 테이블
            if analysis.get('drawdown_events') is not None and len(analysis['drawdown_events']) > 0:
                st.markdown("---")
                st.subheader("📋 포트폴리오 드로우다운")
                
                # portfolio_value를 가져와서 회복시점 계산에 사용
                portfolio_value = result_data.get('portfolio_value')
                
                events = analysis['drawdown_events'][:10]  # 상위 10개만
                events_data = []
                for i, event in enumerate(events, 1):
                    start_date = event['start']
                    trough_date = event['trough']
                    end_date = event['end']
                    
                    # 회복시점: start 이전의 최고점을 다시 회복한 시점
                    recovery_point = None
                    if portfolio_value is not None and start_date in portfolio_value.index:
                        # start 이전의 최고점 찾기
                        peak_value = portfolio_value.loc[:start_date].max()
                        peak_date = portfolio_value.loc[:start_date].idxmax()
                        
                        # end_date에서 이미 peak_value를 회복했는지 확인
                        if end_date in portfolio_value.index:
                            if portfolio_value.loc[end_date] >= peak_value:
                                recovery_point = end_date
                            elif end_date < portfolio_value.index[-1]:
                                # end_date 이후의 모든 날짜 확인
                                future_dates = portfolio_value.index[portfolio_value.index > end_date]
                                
                                for date in future_dates:
                                    if portfolio_value.loc[date] >= peak_value:
                                        recovery_point = date
                                        break
                        else:
                            # 진행 중인 드로우다운
                            recovery_point = None
                    
                    # 손실기간: start부터 recovery_point까지의 기간 (개월) - 전체 손실 구간
                    if recovery_point is not None and recovery_point > start_date:
                        loss_months = (recovery_point.year - start_date.year) * 12 + (recovery_point.month - start_date.month)
                    else:
                        # 아직 회복하지 못한 경우, start부터 현재까지
                        loss_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
                    
                    # 회복기간: trough부터 recovery_point까지의 기간 (개월) - 최저점부터 원래 최고점 회복까지
                    if recovery_point is not None and recovery_point > trough_date:
                        recovery_months = (recovery_point.year - trough_date.year) * 12 + (recovery_point.month - trough_date.month)
                    else:
                        # 아직 회복하지 못한 경우
                        recovery_months = None
                    
                    events_data.append({
                        '순위': i,
                        '시작': start_date.strftime('%Y/%m'),
                        '종료': end_date.strftime('%Y/%m'),
                        '회복시점': recovery_point.strftime('%Y/%m') if recovery_point is not None else '-',
                        '회복기간': f"{recovery_months} 개월" if recovery_months is not None else '-',
                        '손실기간': f"{loss_months} 개월",
                        '드로우다운': f"{event['drawdown']:.1f}%"
                    })
                
                events_df = pd.DataFrame(events_data)
                st.dataframe(events_df, use_container_width=True, hide_index=True)
else:
    st.info("👈 왼쪽 사이드바에서 보유 금액을 입력하고 '실행' 버튼을 클릭하세요.")


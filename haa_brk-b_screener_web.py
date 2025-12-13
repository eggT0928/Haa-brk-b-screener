import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime


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


def select_assets(momentum_scores: pd.DataFrame, data: pd.DataFrame):
    """TIP 기준으로 자산 선택 (offense/defense)"""
    # 가장 마지막 인덱스(=오늘 날짜)를 기준으로 삼음
    target_date = momentum_scores.index[-1]
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

        # 5) 결과 요약 및 테이블 생성
        return display_results(
            momentum_scores,
            data,
            selected_assets,
            tickers,
            total_balance,
            target_date
        )


def display_results(
    momentum_scores: pd.DataFrame,
    data: pd.DataFrame,
    selected_assets: list,
    tickers: list,
    total_balance: float,
    target_date: pd.Timestamp
):
    """결과 표시 및 데이터 반환"""
    haa_bal = total_balance * 0.8
    
    # 선택된 자산 데이터 준비
    selected_data = []
    for asset, score in selected_assets:
        price = data.loc[target_date, asset]
        shares = haa_bal / len(selected_assets) / price
        selected_data.append({
            "자산": asset,
            "모멘텀 점수": f"{score:.3f}",
            "현재 가격": f"${price:.2f}",
            "구매 수량": f"{shares:.2f}"
        })
    
    # BRK-B 모멘텀 점수 계산
    brk_price = data.loc[target_date, "BRK-B"]
    brk_shares = total_balance * 0.2 / brk_price
    brk_momentum = momentum_scores.loc[target_date, "BRK-B"]
    selected_data.append({
        "자산": "BRK-B",
        "모멘텀 점수": f"{brk_momentum:.3f}",
        "현재 가격": f"${brk_price:.2f}",
        "구매 수량": f"{brk_shares:.2f}"
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
        "brk_shares": brk_shares
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
else:
    st.info("👈 왼쪽 사이드바에서 보유 금액을 입력하고 '실행' 버튼을 클릭하세요.")


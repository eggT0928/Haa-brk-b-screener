from pathlib import Path

app_path = Path("haa_brk-b_screener_web.py")
test_path = Path("tests/test_momentum.py")
s = app_path.read_text(encoding="utf-8")


def replace_once(old: str, new: str, label: str):
    global s
    if old not in s:
        raise RuntimeError(f"patch marker not found: {label}")
    s = s.replace(old, new, 1)


replace_once(
    'SP500_REBALANCE_OPTIONS = ["SPY", "SPYM"]\nMOMENTUM_PERIODS = (1, 3, 6, 12)\n',
    'SP500_REBALANCE_OPTIONS = ["SPY", "SPYM"]\nMOMENTUM_PERIODS = (1, 3, 6, 12)\n\n# 백테스트는 Yahoo에서 실제 ETF 데이터를 확보할 수 있는 최대 구간을 사용합니다.\n# 실전 리밸런싱 종목은 PDBC를 유지하되, PDBC 상장 이전 구간은 HAA 원 논문의\n# 원자재 자산인 DBC를 백테스트 전용 프록시로 사용합니다.\nBACKTEST_DOWNLOAD_START = "2000-01-01"\nBACKTEST_COMMODITY_PROXY = "DBC"\nBACKTEST_UI_MIN_DATE = pd.Timestamp("2008-08-01").date()\n',
    "constants",
)

marker = 'def calculate_preview_momentum(data: pd.DataFrame, as_of: pd.Timestamp = None):\n'
helpers = '''def prepare_backtest_data(downloaded: pd.DataFrame, signal_cutoff: pd.Timestamp) -> pd.DataFrame:
    """백테스트 전용 가격 데이터를 만듭니다.

    실전 스크리너에서는 PDBC를 그대로 사용하지만, 장기 백테스트에서는
    HAA 원 논문이 사용한 DBC를 PDBC 자리에 넣어 PDBC 상장 이전까지
    백테스트 구간을 확장합니다. 나머지 자산은 실제 ETF 가격을 그대로 사용합니다.
    """
    required = [t for t in STRATEGY_TICKERS if t != "PDBC"] + [BACKTEST_COMMODITY_PROXY]
    missing = [t for t in required if t not in downloaded.columns]
    if missing:
        raise ValueError(f"백테스트 필수 데이터가 없습니다: {', '.join(missing)}")

    base = downloaded.loc[downloaded.index <= signal_cutoff, required].copy()
    backtest_data = base[[t for t in STRATEGY_TICKERS if t != "PDBC"]].copy()
    backtest_data["PDBC"] = base[BACKTEST_COMMODITY_PROXY]
    return backtest_data[STRATEGY_TICKERS].sort_index()


def get_backtest_available_range(momentum_scores: pd.DataFrame):
    """모든 HAA 구성자산의 13612U 점수가 존재하는 최대 백테스트 구간을 반환합니다."""
    complete_scores = momentum_scores.dropna(subset=STRATEGY_TICKERS, how="any")
    if complete_scores.empty:
        raise ValueError("모든 HAA 자산의 모멘텀 점수가 동시에 존재하는 백테스트 구간이 없습니다.")
    return complete_scores.index[0], complete_scores.index[-1], complete_scores


def select_backtest_range(momentum_scores: pd.DataFrame, requested_start=None, requested_end=None):
    """사용자 요청일을 실제 가능한 월말 신호 구간 안으로 맞춰 백테스트 점수를 반환합니다."""
    available_start, available_end, complete_scores = get_backtest_available_range(momentum_scores)

    start = available_start if requested_start is None else pd.Timestamp(requested_start)
    end = available_end if requested_end is None else pd.Timestamp(requested_end)
    if start > end:
        raise ValueError("백테스트 시작일은 종료일보다 늦을 수 없습니다.")

    start = max(start, available_start)
    end = min(end, available_end)

    selected = complete_scores.loc[(complete_scores.index >= start) & (complete_scores.index <= end)]
    if len(selected) < 2:
        raise ValueError("선택한 백테스트 기간에는 최소 2개의 월말 데이터가 필요합니다.")

    actual_start = selected.index[0]
    actual_end = selected.index[-1]
    return selected, available_start, available_end, actual_start, actual_end


'''
if marker not in s:
    raise RuntimeError("patch marker not found: preview function")
s = s.replace(marker, helpers + marker, 1)

replace_once(
    'def run_screener(total_balance: float, sp500_rebalance_ticker: str = "SPY"):\n',
    'def run_screener(total_balance: float, sp500_rebalance_ticker: str = "SPY",\n                 backtest_start_date=None, backtest_end_date=None):\n',
    "run_screener signature",
)

replace_once(
    '    strategy_tickers = STRATEGY_TICKERS.copy()\n    download_tickers = strategy_tickers + (["SPYM"] if "SPYM" not in strategy_tickers else [])\n\n    # 1) 과거 데이터 다운로드\n    start_date = "2014-11-01"\n    end_date = pd.Timestamp.now().strftime("%Y-%m-%d")\n',
    '    strategy_tickers = STRATEGY_TICKERS.copy()\n    # SPYM은 실전 리밸런싱 가격용, DBC는 장기 백테스트 프록시용입니다.\n    download_tickers = list(dict.fromkeys(strategy_tickers + ["SPYM", BACKTEST_COMMODITY_PROXY]))\n\n    # 1) 가능한 한 긴 과거 데이터를 다운로드합니다.\n    start_date = BACKTEST_DOWNLOAD_START\n    # yfinance의 end는 미포함이므로 다음 날을 넣어 최신 일자를 빠뜨리지 않게 합니다.\n    end_date = (pd.Timestamp.now().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")\n',
    "download range",
)

replace_once(
    '''        # 6) 백테스트 실행 (항상 SPY 기준)
        portfolio_value, rebalancing_history, performance_metrics, analysis_data = run_backtest(
            data, momentum_scores, total_balance
        )
''',
    '''        # 6) 장기 백테스트 데이터 구성
        # 실전 신호는 PDBC를 그대로 사용하지만, 백테스트에서는 DBC를 PDBC 프록시로 사용해
        # HAA 원 논문 구성에 가깝게 만들고 백테스트 가능 기간을 크게 늘립니다.
        backtest_data = prepare_backtest_data(downloaded, signal_cutoff)
        backtest_scores_full = calculate_momentum_scores(backtest_data)
        (
            backtest_scores,
            backtest_available_start,
            backtest_available_end,
            backtest_actual_start,
            backtest_actual_end,
        ) = select_backtest_range(
            backtest_scores_full,
            requested_start=backtest_start_date,
            requested_end=backtest_end_date,
        )

        # 백테스트는 항상 SPY 기준이며, 선택한 기간에 해당하는 월말 신호만 사용합니다.
        portfolio_value, rebalancing_history, performance_metrics, analysis_data = run_backtest(
            backtest_data, backtest_scores, total_balance
        )
''',
    "backtest preparation",
)

old_return = '''        # 8) 결과 요약 및 테이블 생성
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
'''
new_return = '''        # 8) 결과 요약 및 테이블 생성
        result_data = display_results(
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
        result_data["backtest_available_start"] = backtest_available_start
        result_data["backtest_available_end"] = backtest_available_end
        result_data["backtest_actual_start"] = backtest_actual_start
        result_data["backtest_actual_end"] = backtest_actual_end
        result_data["backtest_proxy_note"] = "백테스트의 PDBC 구간은 HAA 원 논문의 DBC를 프록시로 사용합니다."
        return result_data
'''
replace_once(old_return, new_return, "display return")

replace_once(
    '''        if len(momentum_scores.index) > 0:
            first_valid_date = momentum_scores.index[0]
            data_filtered = data[data.index >= first_valid_date].copy()
        else:
            data_filtered = data.copy()
''',
    '''        if len(momentum_scores.index) > 0:
            first_valid_date = momentum_scores.index[0]
            last_valid_date = momentum_scores.index[-1]
            # 사용자 선택 구간 바깥의 가격을 제거해 종료일 이후 수익이 섞이지 않게 합니다.
            data_filtered = data[(data.index >= first_valid_date) & (data.index <= last_valid_date)].copy()
        else:
            data_filtered = data.copy()
''',
    "run_backtest date filtering",
)

replace_once(
    '        "SPYM": "SPDR Portfolio S&P 500 High Dividend ETF",\n',
    '        "SPYM": "SPDR Portfolio S&P 500 ETF",\n',
    "SPYM name",
)

sidebar_marker = '''    sp500_rebalance_ticker = st.selectbox(
        "S&P500 리밸런싱 ETF 선택",
        SP500_REBALANCE_OPTIONS,
        index=0,
        help="실제 매수 추천 수량은 SPY 또는 SPYM 중 선택한 티커 기준으로 계산합니다. 백테스트와 모멘텀 신호는 항상 SPY 기준입니다."
    )
    balance_text = st.text_input(
'''
sidebar_new = '''    sp500_rebalance_ticker = st.selectbox(
        "S&P500 리밸런싱 ETF 선택",
        SP500_REBALANCE_OPTIONS,
        index=0,
        help="실제 매수 추천 수량은 SPY 또는 SPYM 중 선택한 티커 기준으로 계산합니다. 백테스트와 모멘텀 신호는 항상 SPY 기준입니다."
    )

    # 기본값은 백테스트 가능한 최대 기간입니다.
    # 실제 데이터의 첫 유효 월말은 실행 후 자동으로 다시 표시됩니다.
    backtest_default_end = get_last_completed_month_end().date()
    backtest_date_range = st.date_input(
        "백테스트 기간",
        value=(BACKTEST_UI_MIN_DATE, backtest_default_end),
        min_value=BACKTEST_UI_MIN_DATE,
        max_value=backtest_default_end,
        format="YYYY-MM-DD",
        help="기본값은 가능한 최대기간입니다. 백테스트는 SPY 기준이며, PDBC의 과거 구간은 DBC 프록시를 사용합니다."
    )
    if isinstance(backtest_date_range, (tuple, list)) and len(backtest_date_range) == 2:
        backtest_start_date, backtest_end_date = backtest_date_range
    else:
        backtest_start_date, backtest_end_date = BACKTEST_UI_MIN_DATE, backtest_default_end

    balance_text = st.text_input(
'''
replace_once(sidebar_marker, sidebar_new, "sidebar date input")

replace_once(
    '                result_data = run_screener(total_balance, sp500_rebalance_ticker=sp500_rebalance_ticker)\n',
    '                result_data = run_screener(\n                    total_balance,\n                    sp500_rebalance_ticker=sp500_rebalance_ticker,\n                    backtest_start_date=backtest_start_date,\n                    backtest_end_date=backtest_end_date,\n                )\n',
    "run button",
)

replace_once(
    '''        st.caption(f"리밸런싱 ETF: {result_data.get('sp500_rebalance_ticker', 'SPY')} / 백테스트 기준: SPY")
        st.caption("실제 추천은 완료된 월말 신호, 예상 신호는 현재가를 당월 가상 월말 가격으로 사용합니다.")
''',
    '''        st.caption(f"리밸런싱 ETF: {result_data.get('sp500_rebalance_ticker', 'SPY')} / 백테스트 기준: SPY")
        st.caption(
            f"백테스트 적용: {result_data['backtest_actual_start']:%Y-%m-%d} ~ "
            f"{result_data['backtest_actual_end']:%Y-%m-%d} / "
            f"최대 가능: {result_data['backtest_available_start']:%Y-%m-%d} ~ "
            f"{result_data['backtest_available_end']:%Y-%m-%d}"
        )
        st.caption("실제 추천은 완료된 월말 신호, 예상 신호는 현재가를 당월 가상 월말 가격으로 사용합니다.")
''',
    "sidebar backtest caption",
)

replace_once(
    '''    if result_data.get("performance_metrics"):
        st.subheader("📊 백테스트 성과 지표")
        metrics = result_data["performance_metrics"]
''',
    '''    if result_data.get("performance_metrics"):
        st.subheader("📊 백테스트 성과 지표")
        metrics = result_data["performance_metrics"]
        st.caption(
            f"선택 적용기간: {result_data['backtest_actual_start']:%Y-%m-%d} ~ "
            f"{result_data['backtest_actual_end']:%Y-%m-%d} | "
            f"최대 가능기간: {result_data['backtest_available_start']:%Y-%m-%d} ~ "
            f"{result_data['backtest_available_end']:%Y-%m-%d}"
        )
        st.info(result_data.get("backtest_proxy_note", ""))
''',
    "performance caption",
)

app_path.write_text(s, encoding="utf-8")

# 기존 테스트는 유지하고, 장기 백테스트/기간 선택용 회귀 테스트를 추가합니다.
tests = test_path.read_text(encoding="utf-8")
extra = r'''


def test_prepare_backtest_data_uses_dbc_as_pdbc_proxy():
    dates = pd.date_range("2008-01-01", periods=3, freq="D")
    cols = [t for t in haa_app.STRATEGY_TICKERS if t != "PDBC"] + ["DBC"]
    data = pd.DataFrame(100.0, index=dates, columns=cols)
    data["DBC"] = [10.0, 11.0, 12.0]

    result = haa_app.prepare_backtest_data(data, dates[-1])

    assert list(result.columns) == haa_app.STRATEGY_TICKERS
    assert result["PDBC"].tolist() == [10.0, 11.0, 12.0]


def test_select_backtest_range_clamps_and_uses_month_end_dates():
    dates = pd.date_range("2008-08-31", periods=6, freq="ME")
    scores = pd.DataFrame(0.1, index=dates, columns=haa_app.STRATEGY_TICKERS)

    selected, available_start, available_end, actual_start, actual_end = haa_app.select_backtest_range(
        scores,
        requested_start="2008-01-01",
        requested_end="2009-12-31",
    )

    assert available_start == dates[0]
    assert available_end == dates[-1]
    assert actual_start == dates[0]
    assert actual_end == dates[-1]
    assert selected.index.equals(dates)


def test_select_backtest_range_rejects_reversed_dates():
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    scores = pd.DataFrame(0.1, index=dates, columns=haa_app.STRATEGY_TICKERS)

    with pytest.raises(ValueError, match="시작일"):
        haa_app.select_backtest_range(scores, "2020-03-01", "2020-01-01")
'''
if "test_prepare_backtest_data_uses_dbc_as_pdbc_proxy" not in tests:
    test_path.write_text(tests.rstrip() + extra + "\n", encoding="utf-8")

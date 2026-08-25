from pathlib import Path

app_path = Path('haa_brk-b_screener_web.py')
test_path = Path('tests/test_momentum.py')
s = app_path.read_text(encoding='utf-8')

old = 'BACKTEST_UI_MIN_DATE = pd.Timestamp("2008-08-01").date()'
new = 'BACKTEST_UI_MIN_DATE = pd.Timestamp("2008-07-31").date()'
if old not in s:
    raise RuntimeError('UI min date marker not found')
s = s.replace(old, new, 1)

old = '''        if len(momentum_scores.index) > 0:
            first_valid_date = momentum_scores.index[0]
            last_valid_date = momentum_scores.index[-1]
            # 사용자 선택 구간 바깥의 가격을 제거해 종료일 이후 수익이 섞이지 않게 합니다.
            data_filtered = data[(data.index >= first_valid_date) & (data.index <= last_valid_date)].copy()
        else:
            data_filtered = data.copy()
'''
new = '''        if len(momentum_scores.index) > 0:
            first_valid_date = momentum_scores.index[0]
            last_valid_date = momentum_scores.index[-1]
            # 신호 인덱스는 달력상 월말(주말 포함)입니다. 시작 월말이 일요일이어도
            # 그 달의 마지막 실제 거래일이 빠지지 않도록 월초~월말 전체를 포함합니다.
            start_month = first_valid_date.to_period("M").start_time
            end_month = last_valid_date.to_period("M").end_time
            data_filtered = data[(data.index >= start_month) & (data.index <= end_month)].copy()
        else:
            data_filtered = data.copy()
'''
if old not in s:
    raise RuntimeError('run_backtest filter marker not found')
s = s.replace(old, new, 1)

old = '''        except ValueError:
            st.error("올바른 숫자를 입력해주세요.")
'''
new = '''        except ValueError as e:
            st.error(str(e) if str(e) else "올바른 숫자를 입력해주세요.")
'''
if old not in s:
    raise RuntimeError('ValueError marker not found')
s = s.replace(old, new, 1)

app_path.write_text(s, encoding='utf-8')

tests = test_path.read_text(encoding='utf-8')
extra = r'''


def test_backtest_keeps_first_month_when_signal_month_end_is_weekend(monkeypatch):
    # 2020-02-29는 토요일이므로 실제 마지막 거래일(2/28)이 포함되어야 한다.
    dates = pd.to_datetime(["2020-02-28", "2020-03-31", "2020-04-30"])
    data = pd.DataFrame(100.0, index=dates, columns=haa_app.STRATEGY_TICKERS)
    data.loc[pd.Timestamp("2020-03-31")] = 101.0
    data.loc[pd.Timestamp("2020-04-30")] = 102.0

    signal_dates = pd.to_datetime(["2020-02-29", "2020-03-31", "2020-04-30"])
    scores = pd.DataFrame(0.1, index=signal_dates, columns=haa_app.STRATEGY_TICKERS)

    monkeypatch.setattr(haa_app, "get_risk_free_rate", lambda *args, **kwargs: 0.0)
    portfolio_value, _, metrics, _ = haa_app.run_backtest(data, scores, 10000.0)

    assert portfolio_value.index[0] == pd.Timestamp("2020-02-29")
    assert metrics["시작일"] == "2020-02-29"
'''
if 'test_backtest_keeps_first_month_when_signal_month_end_is_weekend' not in tests:
    test_path.write_text(tests.rstrip() + extra + '\n', encoding='utf-8')

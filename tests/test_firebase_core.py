import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from haa.calendar import completed_month, is_open
from haa.core import (
    STRATEGY_TICKERS,
    backtest,
    momentum,
    month_prices,
    prepare_backtest_data,
    preview,
    select,
    select_range,
    signal,
    weights,
)


def frame(start="2024-01-31", periods=32):
    dates = pd.date_range(start, periods=periods, freq="ME")
    return pd.DataFrame(
        {t: 100 * (1 + 0.002 * (i + 1)) ** np.arange(periods) for i, t in enumerate(STRATEGY_TICKERS)},
        index=dates,
    )


def legacy():
    # UI를 실행하지 않고 원본의 순수 함수만 읽어 결과를 대조한다.
    source = Path(__file__).parents[1] / "haa_brk-b_screener_web.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names = {
        "get_month_end_prices",
        "calculate_momentum_returns",
        "calculate_momentum_scores",
        "select_assets",
        "prepare_backtest_data",
    }
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {
        "pd": pd,
        "MonthEnd": pd.offsets.MonthEnd,
        "MOMENTUM_PERIODS": (1, 3, 6, 12),
        "STRATEGY_TICKERS": STRATEGY_TICKERS,
        "BACKTEST_COMMODITY_PROXY": "DBC",
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


def test_existing_momentum_selection_and_proxy_parity():
    old = legacy()
    data = frame()
    np.testing.assert_allclose(momentum(data), old["calculate_momentum_scores"](data), equal_nan=True)
    for day, row in momentum(data).dropna().iterrows():
        expected, _ = old["select_assets"](momentum(data), data, day)
        assert select(row) == [t for t, _ in expected]
    data["DBC"] = data.PDBC * 2
    pd.testing.assert_frame_equal(
        prepare_backtest_data(data, data.index[-1]), old["prepare_backtest_data"](data, data.index[-1])
    )


@pytest.mark.parametrize("month", ["2026-02", "2026-04", "2026-06", "2026-08"])
def test_preview_uses_calendar_month_not_day_of_month(month):
    data = frame()
    period = pd.Period(month, "M")
    quotes = {t: 150.0 for t in STRATEGY_TICKERS}
    result = preview(data, quotes, f"{month}-12T17:00:00Z")
    for n in (1, 3, 6, 12):
        base = (period - n).to_timestamp("M")
        assert result["returns"][str(n)]["SPY"] == pytest.approx(150 / data.loc[base, "SPY"] - 1)


def test_preview_recomputes_tip_and_selection_without_changing_confirmed():
    data = frame()
    official = signal(data, "2026-07", "confirmed", "2026-07-31T20:00:00Z")
    quotes = data.loc["2026-08-31"].to_dict()
    quotes["TIP"] = 1
    expected = preview(data, quotes, "2026-08-12T17:00:00Z")
    assert official["regime"] == "공격"
    assert expected["regime"] == "방어"
    assert expected["selected"] == ["IEF"]
    assert expected["weights"] == {"IEF": 0.8, "BRK-B": 0.2}


def test_preview_matches_confirmed_at_same_month_end():
    data = frame()
    stamp = "2026-08-31T20:00:00Z"
    a = preview(data, data.iloc[-1].to_dict(), stamp)
    b = signal(data, "2026-08", "confirmed", stamp)
    assert a["scores"] == b["scores"]
    assert a["selected"] == b["selected"]


@pytest.mark.parametrize("tip", [0, -0.001])
def test_nonpositive_tip_is_defense(tip):
    row = pd.Series(0.1, index=STRATEGY_TICKERS)
    row["TIP"] = tip
    assert select(row) == ["IEF"]


def test_offense_short_list_and_cash_preserve_original_rule():
    row = pd.Series(-0.1, index=STRATEGY_TICKERS)
    row["TIP"], row["SPY"] = 0.1, 0.2
    assert weights(select(row)) == {"SPY": 0.8, "BRK-B": 0.2}
    row["SPY"] = -0.2
    assert weights(select(row)) == {"BRK-B": 0.2, "CASH": 0.8}


def test_missing_month_and_missing_quote_fail_closed():
    data = frame().drop(pd.Timestamp("2026-05-31"))
    with pytest.raises(ValueError):
        preview(data, dict.fromkeys(STRATEGY_TICKERS, 150), "2026-08-12T17:00:00Z")
    with pytest.raises(ValueError):
        preview(frame(), {"SPY": 150}, "2026-08-12T17:00:00Z")
    with pytest.raises(ValueError):
        select(pd.Series({"TIP": float("nan")}))


def test_missing_final_price_is_not_filled_by_prior_day():
    daily = pd.DataFrame({"SPY": [100, np.nan]}, index=pd.to_datetime(["2026-07-30", "2026-07-31"]))
    assert pd.isna(month_prices(daily).iloc[-1, 0])


@pytest.mark.parametrize(
    ("stamp", "expected"),
    [
        ("2026-07-31T19:59:00Z", "2026-06"),
        ("2026-07-31T20:30:00Z", "2026-06"),
        ("2026-07-31T21:01:00Z", "2026-07"),
        ("2026-08-01T01:00:00Z", "2026-07"),
        ("2024-11-29T18:30:00Z", "2024-10"),
        ("2024-11-29T19:01:00Z", "2024-11"),
    ],
)
def test_confirmed_month_handles_close_delay_weekend_and_early_close(stamp, expected):
    assert str(completed_month(stamp)[0]) == expected


@pytest.mark.parametrize(
    ("stamp", "expected"),
    [
        ("2026-07-03T16:00:00Z", False),
        ("2026-08-29T16:00:00Z", False),
        ("2026-03-06T14:29:00Z", False),
        ("2026-03-06T14:30:00Z", True),
        ("2026-03-09T13:30:00Z", True),
        ("2024-11-29T18:01:00Z", False),
    ],
)
def test_market_hours_holidays_and_dst(stamp, expected):
    assert is_open(stamp) == expected


def test_backtest_no_lookahead_and_spym_independence():
    data = frame()
    data["DBC"] = data.PDBC
    data["SPYM"] = 1
    irx = pd.Series(4.0, index=data.index)
    baseline = backtest(data, irx, "2026-07-31")
    changed = data.copy()
    changed.loc["2026-08-31", :] = 99999
    changed["SPYM"] = np.nan
    assert backtest(changed, irx, "2026-07-31") == baseline
    assert baseline["metrics"]["riskFree"] == pytest.approx(0.04)
    assert baseline["history"][0]["signalMonth"] < baseline["history"][0]["returnMonth"]


def test_first_loss_is_included_in_mdd_and_weekend_month_not_lost():
    data = frame("2019-02-28", 15)
    data[:] = 100.0
    data.loc["2020-03-31":] = 80.0
    data.index = pd.DatetimeIndex(
        [d if d != pd.Timestamp("2020-02-29") else pd.Timestamp("2020-02-28") for d in data.index]
    )
    data["DBC"] = data.PDBC
    result = backtest(data, pd.Series(2.0, index=data.index), "2020-04-30")
    assert result["actualStart"] == "2020-02-29"
    assert result["metrics"]["mdd"] == pytest.approx(-0.2)
    assert result["curve"][0]["value"] == 10000


def test_requested_range_clamping_and_insufficient_data():
    scores = momentum(frame())
    selected, first, last = select_range(scores, "2000-01-01", "2099-12-31")
    assert selected.index[0] == first and selected.index[-1] == last
    with pytest.raises(ValueError, match="시작일"):
        select_range(scores, "2026-06-01", "2025-01-01")
    with pytest.raises(ValueError, match="최소"):
        select_range(scores, "2026-08-01", "2026-08-31")
    with pytest.raises(ValueError, match="누락"):
        select_range(scores.drop(pd.Timestamp("2026-06-30")))


def test_missing_irx_is_not_replaced_with_fixed_rate():
    data = frame()
    data["DBC"] = data.PDBC
    result = backtest(data, pd.Series(dtype=float, index=pd.DatetimeIndex([])), "2026-07-31")
    assert result["metrics"]["sharpe"] is None
    assert result["metrics"]["riskFree"] is None
    assert result["warnings"]


def test_irx_uses_only_selected_interval():
    data = frame()
    data["DBC"] = data.PDBC
    irx = pd.Series(99.0, index=data.index)
    irx.loc["2026-01-31":"2026-07-31"] = 3
    result = backtest(data, irx, "2026-07-31", "2026-01-01", "2026-07-31")
    assert result["metrics"]["riskFree"] == pytest.approx(0.03)

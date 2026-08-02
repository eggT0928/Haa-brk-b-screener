import importlib.util
from pathlib import Path

import pandas as pd
import pytest


APP_PATH = Path(__file__).parents[1] / "haa_brk-b_screener_web.py"
SPEC = importlib.util.spec_from_file_location("haa_app", APP_PATH)
haa_app = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(haa_app)


def test_last_completed_month_excludes_the_current_month():
    assert haa_app.get_last_completed_month_end("2026-08-02") == pd.Timestamp("2026-07-31")


def test_momentum_uses_exact_month_end_lookbacks():
    month_ends = pd.date_range("2025-01-31", periods=13, freq="ME")
    monthly_values = pd.Series(
        [100, 102, 101, 104, 106, 108, 110, 109, 112, 115, 117, 119, 121],
        index=month_ends,
    )

    # 월중 값은 크게 흔들리게 만들어도 신호에는 각 월의 마지막 값만 들어가야 한다.
    rows = []
    for date, value in monthly_values.items():
        rows.extend([(date - pd.Timedelta(days=10), value * 1.5), (date, value)])
    data = pd.DataFrame(rows, columns=["Date", "SPY"]).set_index("Date")

    score = haa_app.calculate_momentum_scores(data).loc[month_ends[-1], "SPY"]
    expected = sum(
        monthly_values.iloc[-1] / monthly_values.iloc[-1 - months] - 1
        for months in (1, 3, 6, 12)
    ) / 4

    assert score == pytest.approx(expected)


def test_rank_uses_month_end_scores_not_21_trading_day_approximations():
    tickers = ["SPY", "VEA", "VWO", "IWM", "BIL", "IEF", "TLT", "TIP", "PDBC", "VNQ"]
    dates = pd.date_range("2025-07-31", periods=13, freq="ME")
    final_prices = {
        "PDBC": 140,
        "IWM": 130,
        "VEA": 120,
        "VNQ": 111,
        "SPY": 110,
        "VWO": 108,
        "IEF": 102,
        "TLT": 99,
        "TIP": 105,
        "BIL": 103,
    }
    data = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    for ticker in tickers:
        data[ticker] = pd.Series(
            [100 + (final_prices[ticker] - 100) * i / 12 for i in range(13)],
            index=dates,
        )

    scores = haa_app.calculate_momentum_scores(data)
    selected, target_date = haa_app.select_assets(scores, data)

    assert target_date == dates[-1]
    assert [ticker for ticker, _ in selected] == ["PDBC", "IWM", "VEA", "VNQ"]

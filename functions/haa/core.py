"""기존 스크리너의 13612U·자산선택·DBC 프록시 로직을 순수 함수로 이식.

SPYM은 이 모듈의 신호 및 백테스트에 절대 포함하지 않는다.
"""

import math

import numpy as np
import pandas as pd

STRATEGY_TICKERS = ["SPY", "VEA", "VWO", "IWM", "BIL", "IEF", "TLT", "TIP", "PDBC", "VNQ", "BRK-B"]
APP_TICKERS = STRATEGY_TICKERS + ["SPYM"]
OFFENSE = ["SPY", "VEA", "VWO", "IWM", "TLT", "PDBC", "VNQ", "IEF"]
PERIODS = (1, 3, 6, 12)
VERSION = "haa80-brk20-13612u-dbc-all-v1"


def month_prices(data: pd.DataFrame) -> pd.DataFrame:
    # 월말 값 누락을 앞선 일자의 가격으로 조용히 채우지 않는다.
    data = data.sort_index()
    monthly = data.resample(pd.offsets.MonthEnd()).last(skipna=False)
    return monthly


def momentum(data: pd.DataFrame) -> pd.DataFrame:
    monthly = month_prices(data)
    returns = [monthly.pct_change(periods=n, fill_method=None) for n in PERIODS]
    return (sum(returns) / len(PERIODS)).dropna(how="all")


def select(scores: pd.Series) -> list[str]:
    required = list(dict.fromkeys(OFFENSE + ["TIP", "BIL"]))
    if not np.isfinite(scores.reindex(required).to_numpy(dtype=float)).all():
        raise ValueError("필수 모멘텀 점수가 누락되어 신호를 계산할 수 없습니다.")
    if scores["TIP"] > 0:
        positive = scores[OFFENSE][scores[OFFENSE] > 0]
        return positive.nlargest(4).index.tolist()
    return scores[["IEF", "BIL"]].nlargest(1).index.tolist()


def weights(selected: list[str]) -> dict[str, float]:
    # 기존 코드처럼 양수 자산 수가 4개 미만이면 그 수로 80%를 균등 배분한다.
    result = {ticker: 0.8 / len(selected) for ticker in selected}
    result["BRK-B"] = 0.2
    if not selected:
        result["CASH"] = 0.8
    return result


def signal(monthly: pd.DataFrame, target_month, kind: str, as_of: str) -> dict:
    target = pd.Period(target_month, freq="M")
    table = month_prices(monthly).copy()
    table.index = table.index.to_period("M")
    required_months = [target] + [target - n for n in PERIODS]
    if any(month not in table.index for month in required_months):
        raise ValueError("1·3·6·12개월 비교에 필요한 월말 가격이 없습니다.")
    sample = table.loc[required_months, STRATEGY_TICKERS]
    if not np.isfinite(sample.to_numpy()).all() or (sample <= 0).any().any():
        raise ValueError("필수 월말 가격이 누락되었거나 유효하지 않습니다.")
    # Period 산술로 4월30일 - 1개월이 3월30일이 되는 월말 오류를 방지한다.
    returns = {str(n): (table.loc[target] / table.loc[target - n] - 1) for n in PERIODS}
    scores = sum(returns.values()) / len(PERIODS)
    chosen = select(scores)
    return {
        "kind": kind,
        "month": str(target),
        "asOf": as_of,
        "version": VERSION,
        "basis": "SPY",
        "commodity": "PDBC",
        "regime": "공격" if scores["TIP"] > 0 else "방어",
        "selected": chosen,
        "weights": weights(chosen),
        "scores": {t: float(scores[t]) for t in STRATEGY_TICKERS},
        "returns": {n: {t: float(r[t]) for t in STRATEGY_TICKERS} for n, r in returns.items()},
    }


def preview(adjusted: pd.DataFrame, raw_quotes: dict[str, float], as_of: str) -> dict:
    now = pd.Timestamp(as_of).tz_convert("America/New_York").tz_localize(None)
    month = now.to_period("M")
    # 같은 Yahoo 조회 묶음의 과거 수정주가와 현재 원시가격을 사용한다.
    # 최신 시점의 수정계수는 1이며 과거 가격이 배당·분할에 맞춰 조정된다.
    table = adjusted.loc[adjusted.index < month.start_time, STRATEGY_TICKERS].copy()
    table.loc[month.to_timestamp("M")] = pd.Series(raw_quotes).reindex(STRATEGY_TICKERS)
    return signal(table, month, "preview", as_of)


def prepare_backtest_data(downloaded: pd.DataFrame, cutoff) -> pd.DataFrame:
    # 현재 main과 동일: 상장 이전뿐 아니라 전 기간 DBC를 PDBC 슬롯에 대입한다.
    needed = [t for t in STRATEGY_TICKERS if t != "PDBC"] + ["DBC"]
    if not set(needed).issubset(downloaded.columns):
        raise ValueError("백테스트 필수 데이터가 없습니다.")
    data = downloaded.loc[:cutoff, needed].copy().rename(columns={"DBC": "PDBC"})
    return data[STRATEGY_TICKERS]


def select_range(scores, start=None, end=None):
    complete = scores.dropna(subset=STRATEGY_TICKERS)
    if len(complete) < 2:
        raise ValueError("모든 자산에 12개월 이력이 있는 월말이 최소 2개 필요합니다.")
    first, last = complete.index[0], complete.index[-1]
    start = first if not start else pd.Timestamp(start)
    end = last if not end else pd.Timestamp(end)
    if start > end:
        raise ValueError("백테스트 시작일은 종료일보다 늦을 수 없습니다.")
    selected = complete.loc[max(first, start) : min(last, end)]
    if len(selected) < 2:
        raise ValueError("선택한 기간에는 최소 2개의 월말 데이터가 필요합니다.")
    expected = pd.date_range(selected.index[0], selected.index[-1], freq="ME")
    if not selected.index.equals(expected):
        raise ValueError("백테스트 중간에 누락된 월이 있습니다. 데이터 갱신이 필요합니다.")
    return selected, first, last


def backtest(downloaded, irx, cutoff, start=None, end=None, initial=10000.0):
    if not math.isfinite(initial) or not 0 < initial <= 1e12:
        raise ValueError("초기 금액은 0 초과, 1조 달러 이하로 입력하세요.")
    data = prepare_backtest_data(downloaded, cutoff)
    scores, first, last = select_range(momentum(data), start, end)
    monthly = month_prices(data).loc[scores.index]
    values = [float(initial)]
    history = []
    for prev, current in zip(scores.index[:-1], scores.index[1:]):
        selected = select(scores.loc[prev])
        allocation = weights(selected)
        month_return = 0.0
        for ticker, weight in allocation.items():
            if ticker == "CASH":
                continue
            old, new = monthly.loc[prev, ticker], monthly.loc[current, ticker]
            if not np.isfinite([old, new]).all() or min(old, new) <= 0:
                raise ValueError(f"{ticker}의 백테스트 가격이 누락되었습니다.")
            month_return += weight * (new / old - 1)
        values.append(values[-1] * (1 + month_return))
        history.append(
            {
                "signalMonth": prev.strftime("%Y-%m"),
                "returnMonth": current.strftime("%Y-%m"),
                "weights": allocation,
                "return": float(month_return),
            }
        )
    curve = pd.Series(values, index=scores.index)
    returns = curve.pct_change().dropna()
    years = (curve.index[-1] - curve.index[0]).days / 365.25
    cagr = (curve.iloc[-1] / initial) ** (1 / years) - 1
    volatility = float(returns.std() * np.sqrt(12)) if len(returns) > 1 else 0.0
    # 초기 원금을 포함해 첫 손실 월도 MDD에 반영한다.
    drawdown = curve / curve.cummax() - 1
    rates = irx.loc[curve.index[0] : curve.index[-1]].dropna()
    rf = float(rates.mean() / 100) if len(rates) else None
    yearly = (1 + returns).groupby(returns.index.year).prod() - 1
    return {
        "version": VERSION,
        "basis": "SPY",
        "commodityProxy": "DBC (전 기간)",
        "availableStart": first.date().isoformat(),
        "availableEnd": last.date().isoformat(),
        "actualStart": curve.index[0].date().isoformat(),
        "actualEnd": curve.index[-1].date().isoformat(),
        "metrics": {
            "totalReturn": float(curve.iloc[-1] / initial - 1),
            "cagr": float(cagr),
            "volatility": volatility,
            "mdd": float(drawdown.min()),
            "riskFree": rf,
            "sharpe": float((cagr - rf) / volatility) if rf is not None and volatility > 0 else None,
        },
        "curve": [
            {"date": d.date().isoformat(), "value": float(v), "drawdown": float(drawdown[d])}
            for d, v in curve.items()
        ],
        "monthly": [{"month": d.strftime("%Y-%m"), "return": float(v)} for d, v in returns.items()],
        "yearly": [{"year": int(y), "return": float(v)} for y, v in yearly.items()],
        "history": history,
        "warnings": []
        if rf is not None
        else ["선택 기간의 ^IRX가 없어 무위험수익률·샤프를 표시하지 않습니다."],
    }
